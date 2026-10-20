"""Merged-QB/SWA output, masking, fallback and changed-input graph contracts."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_qb_kv as producer


class QBKVCPU(unittest.TestCase):
    def test_disabled_and_cpu_do_not_import_backend(self):
        data = torch.empty(128, 1280)
        with patch.object(
            producer, "_ops", side_effect=AssertionError("backend imported")
        ):
            for enabled in ("0", "1"):
                with patch.dict(os.environ, {"DSV41_CUSTOM_QB": enabled}):
                    self.assertIsNone(producer.try_project(None, data, None))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class QBKVCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (10, 3):
            raise unittest.SkipTest("qualified SM103 device required")
        if not getattr(producer._ops(), "dsv41_qb_kv_available", lambda: False)():
            raise unittest.SkipTest("native QB/KV not built")
        import deep_gemm

        cls.dg = deep_gemm
        cls.previous_pdl = deep_gemm.get_pdl()
        deep_gemm.set_pdl(False)
        torch.manual_seed(20261020)
        cls.merged = SimpleNamespace(
            main_width=32768,
            index_width=4096,
            weight=torch.randint(-4, 5, (36864, 1280), device="cuda").to(
                torch.float8_e4m3fn
            ),
            weight_scales=torch.full(
                (10, 36864), 0x7F7F7F7F, device="cuda", dtype=torch.int32
            ).T,
        )

    @classmethod
    def tearDownClass(cls):
        cls.dg.set_pdl(cls.previous_pdl)

    def case(self, m, dtype=torch.int64):
        data = torch.randint(-4, 5, (m, 1, 1280), device="cuda").to(torch.float8_e4m3fn)
        scales = torch.full((10, m), 0x7F7F7F7F, device="cuda", dtype=torch.int32).T
        raw = torch.randn(m, 1, 1792, device="cuda", dtype=torch.bfloat16)[..., -512:]
        gamma = (1 + torch.randn(512, device="cuda") * 0.1).bfloat16()
        positions = torch.arange(m, device="cuda", dtype=dtype) % 1000
        angle = torch.randn(1031, 32, device="cuda")
        frequencies = torch.polar(torch.ones_like(angle), angle)
        pool = torch.empty_strided(
            (6, 128, 528), (128 * 528 + 128, 528, 1), device="cuda", dtype=torch.uint8
        )
        pool.fill_(173)
        slots = torch.arange(m, device="cuda", dtype=dtype) + 128
        slots[0] = -1
        return data, scales, raw, gamma, positions, frequencies, pool, slots

    def oracle(self, case):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_qkv_rope_cache_triton import (
            try_fused_qkv_rope_cache,
        )

        data, scales, raw, gamma, pos, frequencies, pool, slots = case
        m = data.shape[0]
        out = torch.empty(m, 36864, device="cuda", dtype=torch.bfloat16)
        self.dg.fp8_fp4_gemm_nt(
            (data.view(m, 1280), scales),
            (self.merged.weight, self.merged.weight_scales),
            out,
            recipe=(1, 1, 32),
        )
        expected_pool = torch.empty_strided(
            pool.shape, pool.stride(), device=pool.device, dtype=pool.dtype
        )
        expected_pool.copy_(pool)
        reference = try_fused_qkv_rope_cache(
            out[:, :32768].view(m, 1, 64, 512),
            raw,
            gamma,
            pos,
            frequencies,
            expected_pool,
            slots,
            eps=1e-6,
            freqs_cis=frequencies[pos.long()],
            apply_q_rope=False,
            pad_query_heads=64,
        )
        assert reference is not None
        return out, reference.kv, expected_pool

    @torch.inference_mode()
    def test_outputs_and_changed_input_graph(self):
        with patch.dict(os.environ, {"DSV41_FUSED_QB_KV": "1", "DSV41_CUSTOM_QB": "0"}):
            for m in (128, 192, 256, 512):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(m=m, dtype=dtype):
                        case = self.case(m, dtype)
                        data, sf, raw, gamma, pos, frequencies, pool, slots = case

                        def call():
                            return producer.try_project_kv(self.merged, *case, eps=1e-6)

                        expected_q, expected_kv, expected_pool = self.oracle(case)
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                call()
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            actual = call()
                        for phase in range(3):
                            data.copy_(
                                torch.randint(-4, 5, data.shape, device="cuda").to(
                                    data.dtype
                                )
                            )
                            raw.normal_()
                            pos.add_(3)
                            slots[1:].add_(1)
                            pool.fill_(173)
                            expected_q, expected_kv, expected_pool = self.oracle(case)
                            for t in actual:
                                t.fill_(float("nan"))
                            graph.replay()
                            torch.cuda.synchronize()
                            q, iq, kv = actual
                            torch.testing.assert_close(
                                q.reshape(m, 32768),
                                expected_q[:, :32768],
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                iq.reshape(m, 4096),
                                expected_q[:, 32768:],
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(kv, expected_kv, rtol=0, atol=0)
                            torch.testing.assert_close(
                                pool, expected_pool, rtol=0, atol=0
                            )
                            self.assertEqual(q.stride(0), 36864)
                            self.assertEqual(
                                iq.untyped_storage().data_ptr(),
                                q.untyped_storage().data_ptr(),
                            )

    @torch.inference_mode()
    def test_qnorm_pdl_changed_input_graph(self):
        from rtp_llm.models_py.modules.dsv41._v41_query_quant import query_norm_quant

        with patch.dict(os.environ, {"DSV41_FUSED_QB_KV": "1"}):
            for m in (128, 512):
                case = list(self.case(m))
                latent = torch.randn(m, 1, 1280, device="cuda", dtype=torch.bfloat16)
                weight = torch.ones(1280, device="cuda", dtype=torch.bfloat16)
                for pdl in (False, True):

                    def pipeline():
                        _, (data, scales) = query_norm_quant(
                            latent, weight, 1e-6, pdl=pdl
                        )
                        return producer.try_project_kv(
                            self.merged, data, scales, *case[2:], eps=1e-6, wait_pdl=pdl
                        )

                    pipeline()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = pipeline()
                    for phase in range(3):
                        latent.normal_()
                        case[2].normal_()
                        case[4].add_(1)
                        case[7][1:].add_(1)
                        case[6].fill_(173)
                        _, (case[0], case[1]) = query_norm_quant(latent, weight, 1e-6)
                        q, kv, pool = self.oracle(case)
                        for out in actual:
                            out.fill_(float("nan"))
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            actual[0].view(m, 32768), q[:, :32768], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual[1].view(m, 4096), q[:, 32768:], rtol=0, atol=0
                        )
                        torch.testing.assert_close(actual[2], kv, rtol=0, atol=0)
                        torch.testing.assert_close(case[6], pool, rtol=0, atol=0)

    @torch.inference_mode()
    def test_decode_contract_and_batched_span(self):
        from rtp_llm.models_py.modules.dsv41 import _v41_query_quant
        from rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv import (
            decode_compute_qkv,
        )

        for batch, span in ((128, 1), (64, 2)):
            case = list(self.case(128))
            case[0] = case[0].view(batch, span, 1280)
            case[2] = case[2].view(batch, span, 512)
            qr = torch.randn(batch, span, 1280, device="cuda", dtype=torch.bfloat16)
            frequencies = case[5][case[4].long()].contiguous()
            attn = SimpleNamespace(
                rope_head_dim=64,
                n_heads=64,
                head_dim=512,
                skip_post_q_norm=True,
                eps=1e-6,
                kv_norm=case[3],
                freqs_cis=case[5],
            )
            mega = SimpleNamespace(merged_qb=self.merged, padded_heads=64)
            # The projection is held fixed; exercise the actual new decode
            # branch, output views, frequency identity and SWA-written flag.
            with patch.object(
                _v41_query_quant,
                "try_project_quantized_qkv",
                return_value=(qr, case[2], (case[0], case[1])),
                create=True,
            ), patch.dict(os.environ, {"DSV41_FUSED_QB_KV": "1"}):
                actual = decode_compute_qkv(
                    attn,
                    qr,
                    case[4],
                    frequencies,
                    swa_pool=case[6],
                    swa_slots=case[7],
                    mega_attention=mega,
                )
            oracle_case = list(case)
            oracle_case[0] = case[0].view(128, 1, 1280)
            flat_kv = case[2].view(128, 512)
            # The old Triton admission requires the size-1 dimension to carry
            # the row stride too; both views refer to identical physical rows.
            oracle_case[2] = flat_kv.as_strided(
                (128, 1, 512), (flat_kv.stride(0), flat_kv.stride(0), 1)
            )
            q, kv, pool = self.oracle(oracle_case)
            torch.testing.assert_close(
                actual.q.reshape(128, 32768), q[:, :32768], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual.index_q.reshape(128, 4096), q[:, 32768:], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual.kv.reshape(128, 1, 512), kv, rtol=0, atol=0
            )
            torch.testing.assert_close(case[6], pool, rtol=0, atol=0)
            self.assertIs(actual.qr, qr)
            self.assertIs(actual.freqs_cis, frequencies)
            self.assertTrue(actual.swa_written)
            self.assertEqual(actual.q.view(128, 32768).stride(0), 36864)

    @torch.inference_mode()
    def test_reindex_prepared_path_is_preserved(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_front_gemm as front
        from rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv import (
            decode_compute_qkv,
        )

        data, sf, raw, gamma, pos, frequencies, pool, slots = self.case(128)
        qr = torch.randn(128, 1, 1280, device="cuda", dtype=torch.bfloat16)
        head = torch.ones(128, 1, 32, device="cuda", dtype=torch.bfloat16)
        prepared = tuple(torch.empty(1, device="cuda") for _ in range(3))
        q = torch.zeros(128, 1, 32768, device="cuda", dtype=torch.bfloat16)
        attn = SimpleNamespace(
            is_index_source=True,
            is_kv_source=False,
            rope_head_dim=64,
            n_heads=64,
            head_dim=512,
            skip_post_q_norm=True,
            eps=1e-6,
            kv_norm=gamma,
            freqs_cis=frequencies,
        )
        mega = SimpleNamespace(merged_qb=self.merged, padded_heads=64)
        with patch.dict(os.environ, {"DSV41_FUSED_QB_KV": "1"}), patch.object(
            front, "project", return_value=(qr, raw, (data, sf), head)
        ), patch.object(
            producer, "try_project_kv", side_effect=AssertionError("preempted Reindex")
        ), patch.object(
            self.merged, "try_forward_prepared", return_value=(q, prepared), create=True
        ) as existing:
            actual = decode_compute_qkv(
                attn,
                qr,
                pos,
                frequencies[pos.long()],
                swa_pool=pool,
                swa_slots=slots,
                mega_attention=mega,
            )
        existing.assert_called_once()
        self.assertIs(actual.index_prepared, prepared)
        self.assertIs(actual.head_weights, head)
        self.assertIsNone(actual.index_q)
        self.assertTrue(actual.swa_written)

    @torch.inference_mode()
    def test_unsupported_shapes_and_layout_have_no_write(self):
        with patch.dict(os.environ, {"DSV41_FUSED_QB_KV": "1"}):
            for m in (127, 129, 191, 193, 255, 257, 384, 511):
                case = self.case(m)
                before = case[-2].clone()
                self.assertIsNone(producer.try_project_kv(self.merged, *case, eps=1e-6))
                torch.testing.assert_close(case[-2], before, rtol=0, atol=0)

    @torch.inference_mode()
    def test_unsupported_metadata_does_not_write(self):
        with patch.dict(os.environ, {"DSV41_FUSED_QB_KV": "1"}):
            for index, replacement in (
                (3, lambda t: t.float()),
                (4, lambda t: t.float()),
                (5, lambda t: t.to(torch.complex128)),
                (7, lambda t: t.float()),
            ):
                case = list(self.case(128))
                before = case[6].clone()
                case[index] = replacement(case[index])
                self.assertIsNone(producer.try_project_kv(self.merged, *case, eps=1e-6))
                torch.testing.assert_close(case[6], before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
