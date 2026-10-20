"""REUSE 3->1, masked slot semantics and changed-input PDL graph contracts."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_qb_kv as producer
from rtp_llm.models_py.modules.dsv41.fp8.decode.ready_qkv import ReadyIndexPlan


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ReadyQKVCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (10, 3):
            raise unittest.SkipTest("qualified SM103 device required")
        if not getattr(
            producer._ops(), "dsv41_flashmla_pdl_available", lambda: False
        )():
            raise unittest.SkipTest("native PDL consumer not built")
        import deep_gemm

        cls.dg = deep_gemm
        cls.old_pdl = deep_gemm.get_pdl()
        deep_gemm.set_pdl(False)
        torch.manual_seed(20261021)
        cls.mega = SimpleNamespace(
            query_weight=torch.randint(-2, 3, (32768, 1280), device="cuda").to(
                torch.float8_e4m3fn
            ),
            query_scale=torch.full(
                (10, 32768), 0x7F7F7F7F, device="cuda", dtype=torch.int32
            ).T,
            heads=64,
            padded_heads=64,
            groups=8,
            merged_qb=None,
            sink=torch.linspace(-1, 1, 64, device="cuda"),
            softmax_scale=512**-0.5,
        )
        cls.environment = patch.dict(os.environ, {"DSV41_FUSED_READY_QKV": "1"})
        cls.environment.start()

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        cls.dg.set_pdl(cls.old_pdl)

    def case(self, m, k=512, entries=64, metadata_dtype=torch.int64, padded_table=True):
        data = torch.randint(-2, 3, (m, 1, 1280), device="cuda").to(torch.float8_e4m3fn)
        sf = torch.full((10, m), 0x7F7F7F7F, device="cuda", dtype=torch.int32).T
        raw = torch.randn(m, 1, 1792, device="cuda", dtype=torch.bfloat16)[..., -512:]
        gamma = (1 + torch.randn(512, device="cuda") * 0.1).bfloat16()
        positions = torch.arange(m, device="cuda", dtype=metadata_dtype) + 1
        angle = torch.randn(2048, 32, device="cuda")
        frequencies = torch.polar(torch.ones_like(angle), angle)
        self.mega.cos_sin = torch.cat((frequencies.real, frequencies.imag), dim=-1)
        # Whole-token page padding exercises the native TMA cache contract.
        pool = torch.empty_strided(
            (12, 128, 528), (129 * 528, 528, 1), device="cuda", dtype=torch.uint8
        )
        pool.zero_()
        pool[..., 512:].fill_(127)
        slots = torch.arange(m, device="cuda", dtype=metadata_dtype) + 128
        slots[0] = -1
        req = (torch.arange(m, device="cuda", dtype=torch.int32) * 7) % 13
        if padded_table:
            table = (
                torch.arange(13 * 8, device="cuda", dtype=torch.int32).view(13, 8) % 9
                + 1
            )[:, :5]
            table[:, 0] = 0
            table[0, 1] = -1
        else:
            table = (
                torch.arange(13 * 5, device="cuda", dtype=torch.int32).view(13, 5) % 9
                + 1
            )
            table[:, 0] = 0
        local = torch.randint(
            0, 5 * entries, (m, k + 128), device="cuda", dtype=torch.int32
        )[:, :k]
        local[:, 0] = -1
        local[:, 1] = 5 * entries
        local[:, 2] = 0
        plan = ReadyIndexPlan(req, table, local, entries, entries)
        extra = torch.empty_strided(
            (12, entries, 288),
            ((entries + 1) * 288, 288, 1),
            device="cuda",
            dtype=torch.uint8,
        )
        extra[..., :256].random_(0, 256)
        extra[..., 256:].fill_(48)
        swa_indices = (
            torch.arange(128, device="cuda", dtype=torch.int32).expand(m, -1).clone()
        )
        swa_indices[:, -1] = slots.int()
        return (
            (data, sf, raw, gamma, positions, frequencies, pool, slots),
            plan,
            extra,
            swa_indices,
        )

    def oracle(self, case, plan):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_qkv_rope_cache_triton import (
            try_fused_qkv_rope_cache,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            translate_local_to_global_slots,
        )

        data, sf, raw, gamma, positions, frequencies, pool, slots = case
        m = data.shape[0]
        q = torch.empty(m, 32768, device=data.device, dtype=torch.bfloat16)
        self.dg.fp8_fp4_gemm_nt(
            (data.view(m, 1280), sf),
            (self.mega.query_weight, self.mega.query_scale),
            q,
            recipe=(1, 1, 32),
        )
        expected_pool = torch.empty_strided(
            pool.shape, pool.stride(), device=pool.device, dtype=pool.dtype
        )
        expected_pool.copy_(pool)
        ref = try_fused_qkv_rope_cache(
            q.view(m, 1, 64, 512),
            raw,
            gamma,
            positions,
            frequencies,
            expected_pool,
            slots,
            eps=1e-6,
            freqs_cis=frequencies[positions.long()],
            apply_q_rope=False,
            pad_query_heads=64,
        )
        assert ref is not None
        global_slots = translate_local_to_global_slots(
            plan.req,
            plan.table,
            plan.local,
            entries_per_block=plan.entries,
            tokens_per_block_for_block_table=plan.logical_entries,
        )
        return q, ref.kv, expected_pool, global_slots

    def fused(self, case, plan, pdl=False):
        return producer.try_project_kv_index(
            self.mega, *case, plan, eps=1e-6, trigger_pdl=pdl
        )

    def consumer(self, q, pool, swa_indices, positions, extra, indices, pdl=False):
        m = q.shape[0]
        output = torch.empty((m, 8, 4096), device="cuda", dtype=torch.float8_e4m3fn)
        sf = torch.empty((8, 32, m), device="cuda", dtype=torch.int32).permute(2, 0, 1)
        lse = torch.empty((m, 64), device="cuda", dtype=torch.float32)
        producer._ops().dsv41_flashmla_decode_out(
            q.view(m, 64, 512),
            pool,
            swa_indices,
            extra,
            indices,
            positions,
            self.mega.cos_sin,
            self.mega.sink,
            output,
            sf,
            lse,
            self.mega.softmax_scale,
            pdl,
        )
        return output, sf, lse

    @torch.inference_mode()
    def test_three_to_one_all_shapes_changed_input_graph(self):
        for m in (128, 192, 256, 384, 512):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(m=m, dtype=dtype):
                    case, plan, _, _ = self.case(
                        m, 512 if m < 256 else 2048, 64 if m < 256 else 128, dtype
                    )
                    self.fused(case, plan)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = self.fused(case, plan)
                    local_before = plan.local.clone()
                    for phase in range(3):
                        case[0].copy_(
                            torch.randint(-2, 3, case[0].shape, device="cuda").to(
                                case[0].dtype
                            )
                        )
                        case[2].normal_()
                        case[4].add_(1)
                        case[7][1:].add_(1)
                        self.mega.query_weight.view(torch.uint8).bitwise_xor_(128)
                        plan.req.add_(1).remainder_(13)
                        plan.local[:, 3:].add_(7).remainder_(plan.logical_entries * 5)
                        plan.table[:, 1:].add_(1).remainder_(9).add_(1)
                        case[6].zero_()
                        case[6][..., 512:].fill_(127)
                        expected = self.oracle(case, plan)
                        actual[0].fill_(float("nan"))
                        actual[1].fill_(float("nan"))
                        actual[2].fill_(-173)
                        before = plan.local.clone()
                        graph.replay()
                        torch.cuda.synchronize()
                        for value, target in zip(
                            (actual[0].view(m, 32768), actual[1], case[6], actual[2]),
                            expected,
                        ):
                            torch.testing.assert_close(value, target, rtol=0, atol=0)
                        torch.testing.assert_close(plan.local, before, rtol=0, atol=0)
                    self.assertTrue(torch.equal(local_before[:, :3], plan.local[:, :3]))

    @torch.inference_mode()
    def test_pdl_consumer_changed_inputs(self):
        for m in (128, 256, 512):
            for pdl in (False, True):
                with self.subTest(m=m, pdl=pdl):
                    case, plan, extra, swa_indices = self.case(
                        m, metadata_dtype=torch.int32
                    )

                    def pipeline():
                        q, kv, slots = self.fused(case, plan, pdl)
                        return self.consumer(
                            q, case[6], swa_indices, case[4], extra, slots, pdl
                        )

                    pipeline()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph):
                        actual = pipeline()
                    # The exact graph-edge audit is performed by the standalone
                    # qualifier's native graph inspector, not inferred from flags.
                    for phase in range(4):
                        case[0].copy_(
                            torch.randint(-2, 3, case[0].shape, device="cuda").to(
                                case[0].dtype
                            )
                        )
                        case[2].normal_()
                        case[4].add_(1)
                        plan.local[:, 3:].add_(3).remainder_(plan.logical_entries * 5)
                        case[6].zero_()
                        case[6][..., 512:].fill_(127)
                        q, _, expected_pool, slots = self.oracle(case, plan)
                        expected = self.consumer(
                            q, expected_pool, swa_indices, case[4], extra, slots, False
                        )
                        actual[0].fill_(float("nan"))
                        actual[1].fill_(-173)
                        actual[2].fill_(float("nan"))
                        graph.replay()
                        torch.cuda.synchronize()
                        # Same arithmetic and source-pinned consumer on both
                        # sides. PDL changes only launch/wait, so require exactness.
                        for value, target in zip(actual, expected):
                            torch.testing.assert_close(
                                (
                                    value.view(torch.uint8)
                                    if value.dtype == torch.float8_e4m3fn
                                    else value
                                ),
                                (
                                    target.view(torch.uint8)
                                    if target.dtype == torch.float8_e4m3fn
                                    else target
                                ),
                                rtol=0,
                                atol=0,
                            )

    @torch.inference_mode()
    def test_unsupported_metadata_preserves_cache(self):
        case, plan, _, _ = self.case(128)
        original = case[6].clone()
        for invalid in (
            ReadyIndexPlan(plan.req.long(), plan.table, plan.local, 64, 64),
            ReadyIndexPlan(plan.req, plan.table, plan.local[:, :511], 64, 64),
        ):
            self.assertIsNone(self.fused(case, invalid))
            torch.testing.assert_close(case[6], original, rtol=0, atol=0)

    @torch.inference_mode()
    def test_real_decode_and_mega_pdl_route(self):
        from rtp_llm.models_py.modules.dsv41 import _v41_query_quant
        from rtp_llm.models_py.modules.dsv41.fp8._v41_mega_attention import (
            MegaAttention,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv import (
            decode_compute_qkv,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.ready_qkv import make_ready_plan

        for batch, span, ratio in ((128, 1, 1), (64, 2, 2)):
            case, source_plan, extra, swa_indices = self.case(
                128, metadata_dtype=torch.int32
            )
            source_plan.local = source_plan.local.contiguous()
            data, sf, raw, gamma, pos, frequencies, pool, write_slots = case
            mega = MegaAttention(
                SimpleNamespace(),
                self.mega.query_weight,
                self.mega.query_scale,
                torch.empty(0, device="cuda"),
                torch.empty(0, device="cuda"),
                self.mega.cos_sin,
                self.mega.sink,
                512**-0.5,
                64,
                8,
            )
            attn = SimpleNamespace(
                compress_ratio=ratio,
                is_kv_source=False,
                is_index_source=False,
                n_heads=64,
                head_dim=512,
                rope_head_dim=64,
                skip_post_q_norm=True,
                eps=1e-6,
                kv_norm=gamma,
                freqs_cis=frequencies,
                _cp_ctx=None,
                index_source_layer_id=3,
                _shared_attention={"topk": {3: source_plan.local}},
                _global_region=lambda: "global_kv_2",
                _source_pool=lambda _: extra,
                _block_tables_by_type={"global_kv_2": source_plan.table},
                _kv_cache=SimpleNamespace(
                    group_tags=["global_kv_2"],
                    get_kernel_seq_size_per_block=lambda _: ratio * source_plan.entries,
                ),
                _mega_position_chunks=lambda *args: (pos,),
            )
            metadata = SimpleNamespace(
                position_ids=pos, req_id_per_token=source_plan.req
            )
            qr = torch.randn(batch, span, 1280, device="cuda", dtype=torch.bfloat16)
            selected_freq = frequencies[pos.long()].contiguous()
            with patch.dict(os.environ, {"DSV41_READY_FLASHMLA_PDL": "1"}):
                plan = make_ready_plan(attn, mega, metadata, 128, pool, swa_indices)
            self.assertIsNotNone(plan)
            self.assertTrue(plan.pdl)
            self.assertIs(plan.local, source_plan.local)
            self.assertIs(plan.positions, pos)
            original_topk = source_plan.local.clone()
            with patch.object(
                _v41_query_quant,
                "try_project_quantized_qkv",
                return_value=(
                    qr,
                    raw.view(batch, span, 512),
                    (data.view(batch, span, 1280), sf),
                ),
            ):

                def pipeline():
                    qkv = decode_compute_qkv(
                        attn,
                        qr,
                        pos,
                        selected_freq,
                        swa_pool=pool,
                        swa_slots=write_slots,
                        mega_attention=mega,
                        ready_plan=plan,
                    )
                    self.assertTrue(qkv.swa_written and qkv.flashmla_pdl)
                    self.assertIs(qkv.qr, qr)
                    self.assertIs(qkv.freqs_cis, selected_freq)
                    self.assertIsNone(qkv.index_q)
                    return mega.decode(
                        qkv.q,
                        pool,
                        swa_indices,
                        plan.positions,
                        extra_pool=extra,
                        extra_indices=qkv.global_slots,
                        wait_pdl=qkv.flashmla_pdl,
                    )

                pipeline()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph):
                    actual = pipeline()
                inspect = getattr(producer._ops(), "inspect_graph", None)
                if inspect is not None:
                    audit = inspect(graph.raw_cuda_graph())
                    self.assertEqual(audit["node_count"], 2)
                    self.assertEqual(sum(e["type"] == 1 for e in audit["edges"]), 1)
            q, _, expected_pool, slots = self.oracle(case, source_plan)
            expected = self.consumer(q, expected_pool, swa_indices, pos, extra, slots)
            graph.replay()
            torch.cuda.synchronize()
            for value, target in zip(actual, expected):
                torch.testing.assert_close(
                    (
                        value.view(torch.uint8)
                        if value.dtype == torch.float8_e4m3fn
                        else value
                    ),
                    (
                        target.view(torch.uint8)
                        if target.dtype == torch.float8_e4m3fn
                        else target
                    ),
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(source_plan.local, original_topk, rtol=0, atol=0)
            for kv_source, index_source in ((True, True), (False, True)):
                attn.is_kv_source, attn.is_index_source = kv_source, index_source
                self.assertIsNone(
                    make_ready_plan(attn, mega, metadata, 128, pool, swa_indices)
                )


if __name__ == "__main__":
    unittest.main()
