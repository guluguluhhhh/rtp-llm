"""PDL launch policy, generated instructions, and graph dependency regression."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from rtp_llm.models_py.modules.dsv41 import _pdl
from rtp_llm.models_py.modules.dsv41._v41_query_quant import (
    _query_norm_quant_kernel,
    query_norm_quant,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_norm_quant import (
    _v41_norm_quant_kernel,
    rmsnorm_group32_quant,
)


class PDLPolicyTest(unittest.TestCase):
    def test_query_norm_preload_and_tail_ptx(self):
        signature = dict(
            zip(
                _query_norm_quant_kernel.arg_names,
                ("*bf16", "*bf16", "*bf16", "*fp8e4nv", "*i32", "i32", "i32")
                + ("constexpr",) * 5,
            )
        )
        for use_pdl in (False, True):
            for producer_pdl in (False, True):
                with self.subTest(use_pdl=use_pdl, producer_pdl=producer_pdl):
                    kernel = triton.compile(
                        ASTSource(
                            _query_norm_quant_kernel,
                            signature,
                            dict(
                                D=1280,
                                EPS=1e-6,
                                BLOCK=2048,
                                PDL=producer_pdl,
                                USE_PDL=use_pdl,
                            ),
                        ),
                        target=GPUTarget("cuda", 103, 32),
                        options=dict(
                            num_warps=8, enable_fp_fusion=False, launch_pdl=use_pdl
                        ),
                    )
                    self.assertEqual(kernel.metadata.launch_pdl, use_pdl)
                    ptx = kernel.asm["ptx"]
                    self.assertEqual("griddepcontrol.wait" in ptx, use_pdl)
                    if use_pdl:
                        wait = ptx.index("griddepcontrol.wait")
                        self.assertLess(ptx.index("ld.global"), wait)
                        self.assertGreater(ptx.rindex("ld.global"), wait)
                    signal = "griddepcontrol.launch_dependents"
                    self.assertEqual(ptx.count(signal), int(use_pdl or producer_pdl))
                    if use_pdl or producer_pdl:
                        stores = [
                            i
                            for i, line in enumerate(ptx.splitlines())
                            if "st.global" in line
                        ]
                        trigger = next(
                            i
                            for i, line in enumerate(ptx.splitlines())
                            if signal in line
                        )
                        self.assertGreaterEqual(len(stores), 3)
                        self.assertLess(stores[-2], trigger)
                        self.assertLess(trigger, stores[-1])

    def test_device_gates_and_overrides(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _pdl, "_device_supports_pdl", return_value=True
        ) as capability:
            self.assertFalse(_pdl.enabled(torch.device("cpu")))
            capability.assert_not_called()
            device = torch.device("cuda", 3)
            self.assertTrue(_pdl.enabled(device))
            capability.assert_called_once_with(3)
            tensor = SimpleNamespace(device=device)
            self.assertEqual(
                _pdl.launch_options(tensor), {"USE_PDL": True, "launch_pdl": True}
            )
            for setting in ({"DSV41_ENABLE_PDL": "0"}, {"DISABLE_PDL": "1"}):
                with patch.dict(os.environ, setting):
                    self.assertEqual(_pdl.launch_options(tensor), {"USE_PDL": False})
            with patch.object(torch.version, "hip", "test"):
                self.assertEqual(_pdl.launch_options(tensor), {"USE_PDL": False})
            capability.return_value = False
            self.assertEqual(_pdl.launch_options(tensor), {"USE_PDL": False})

    def test_norm_quant_ptx(self):
        signature = dict(
            zip(
                _v41_norm_quant_kernel.arg_names,
                (
                    "*bf16",
                    "*bf16",
                    "*bf16",
                    "*fp8e4nv",
                    "*i32",
                    "i32",
                    "fp32",
                    "i1",
                    "constexpr",
                ),
            )
        )
        for use_pdl in (False, True):
            kernel = triton.compile(
                ASTSource(_v41_norm_quant_kernel, signature, {"USE_PDL": use_pdl}),
                target=GPUTarget("cuda", 103, 32),
                options={
                    "num_warps": 4,
                    "enable_fp_fusion": False,
                    "launch_pdl": use_pdl,
                },
            )
            self.assertEqual(kernel.metadata.launch_pdl, use_pdl)
            ptx = kernel.asm["ptx"]
            self.assertEqual("griddepcontrol.wait" in ptx, use_pdl)
            if use_pdl:
                self.assertLess(
                    ptx.index("griddepcontrol.wait"), ptx.index("ld.global")
                )


@triton.jit
def _delayed_producer(source, output, N: tl.constexpr, USE_PDL: tl.constexpr):
    _pdl.pdl_sync(USE_PDL)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    col = tl.program_id(0) * 256 + tl.arange(0, 256)
    # Publish the launch signal well before the writes, exposing missing waits.
    tl.inline_asm_elementwise(
        "nanosleep.u32 100000; // $0", "=r", [], tl.int32, is_pure=False, pack=1
    )
    value = tl.load(source + col, col < N, 0)
    tl.store(output + col, value, col < N)


@triton.jit
def _query_output_consumer(
    norm,
    quant,
    scales,
    out_norm,
    out_quant,
    out_scales,
    N: tl.constexpr,
    SF_N: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    _pdl.pdl_sync(USE_PDL)
    col = tl.program_id(0) * 256 + tl.arange(0, 256)
    quant = quant.to(tl.pointer_type(tl.uint8))
    out_quant = out_quant.to(tl.pointer_type(tl.uint8))
    tl.store(out_norm + col, tl.load(norm + col, col < N, 0), col < N)
    tl.store(out_quant + col, tl.load(quant + col, col < N, 0), col < N)
    tl.store(out_scales + col, tl.load(scales + col, col < SF_N, 0), col < SF_N)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PDLGraphTest(unittest.TestCase):
    def test_query_norm_producer_and_consumer_dependencies(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("V4.1 query quant requires SM10x")
        for rows in (96, 384):
            source = torch.randn(rows, 1280, device="cuda", dtype=torch.bfloat16)
            scratch = torch.empty_like(source)
            weight = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
            outputs = (
                torch.empty_like(source),
                torch.empty_like(source, dtype=torch.float8_e4m3fn),
                torch.empty((10, rows), device="cuda", dtype=torch.int32).T,
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            for setting in ("0", "1"):
                for producer_pdl in (False, True):
                    with self.subTest(
                        rows=rows, setting=setting, pdl=producer_pdl
                    ), patch.dict(
                        os.environ, {"DSV41_ENABLE_PDL": setting, "DISABLE_PDL": "0"}
                    ):

                        def chain():
                            _delayed_producer[(triton.cdiv(source.numel(), 256),)](
                                source,
                                scratch,
                                source.numel(),
                                **_pdl.launch_options(source)
                            )
                            norm, (quant, scales) = query_norm_quant(
                                scratch, weight, 1e-6, pdl=producer_pdl
                            )
                            _query_output_consumer[(triton.cdiv(source.numel(), 256),)](
                                norm,
                                quant,
                                scales,
                                *outputs,
                                source.numel(),
                                scales.numel(),
                                **_pdl.launch_options(source)
                            )

                        with torch.cuda.stream(stream):
                            chain()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                chain()
                            for magnitude in (0.0, 1e-8, 1.0, 100.0):
                                source.normal_(std=magnitude)
                                graph.replay()
                                with patch.dict(os.environ, {"DSV41_ENABLE_PDL": "0"}):
                                    norm, (quant, scales) = query_norm_quant(
                                        source, weight, 1e-6
                                    )
                                for actual, expected in zip(
                                    outputs, (norm, quant, scales)
                                ):
                                    if actual.dtype == torch.float8_e4m3fn:
                                        actual = actual.view(torch.uint8)
                                        expected = expected.view(torch.uint8)
                                    torch.testing.assert_close(
                                        actual, expected, rtol=0, atol=0
                                    )
                        torch.cuda.current_stream().wait_stream(stream)

    def test_precompiled_swa_refresh_policy_and_graph_dependencies(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("PDL requires SM90 or newer")
        from rtp_llm.models_py.modules.dsv41.fp8 import (
            _v41_swa_refresh_triton as refresh,
        )

        source = torch.randn((7, 512), device="cuda", dtype=torch.bfloat16)
        scratch = torch.empty_like(source)
        output = torch.full((13, 512), 7, device="cuda", dtype=torch.bfloat16)
        layout = torch.tensor(
            [[0, 2, 4, 1], [4, 10, 3, 2]], device="cuda", dtype=torch.int64
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for setting in ("0", "1"):
            with patch.dict(
                os.environ, {"DSV41_ENABLE_PDL": setting, "DISABLE_PDL": "0"}
            ):
                refresh.warmup(source.device)
                kernel = refresh._READY[refresh._key(source.device)]
                self.assertEqual(kernel.metadata.launch_pdl, setting == "1")

                def chain():
                    _delayed_producer[(triton.cdiv(source.numel(), 256),)](
                        source, scratch, source.numel(), **_pdl.launch_options(source)
                    )
                    refresh.copy_rows(scratch, output, layout, 2)

                with torch.cuda.stream(stream):
                    chain()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        chain()
                    for _ in range(12):
                        source.normal_()
                        graph.replay()
                        expected = torch.full_like(output, 7)
                        expected[2:6] = source[:4]
                        expected[10:13] = source[4:]
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.cuda.current_stream().wait_stream(stream)

    def test_native_quant_and_deepselect_graph_dependencies(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("DeepSelect requires SM100/SM103")
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        source = torch.randn((17, 5120), device="cuda", dtype=torch.bfloat16)
        scratch = torch.empty_like(source)
        bounds = torch.full((17,), 5120, device="cuda", dtype=torch.int32)
        ends = torch.empty_like(bounds)
        quant = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        scales = torch.empty((40, 20), device="cuda", dtype=torch.int32).T[:17]
        indices = torch.empty((17, 512), device="cuda", dtype=torch.int32)
        ref_quant = torch.empty_like(quant)
        ref_scales = torch.empty((40, 20), device="cuda", dtype=torch.int32).T[:17]
        ref_indices = torch.empty_like(indices)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for use_pdl in (False, True):

            def chain():
                for src, dst in ((source, scratch), (bounds, ends)):
                    _delayed_producer[(triton.cdiv(src.numel(), 256),)](
                        src, dst, src.numel(), USE_PDL=use_pdl, launch_pdl=use_pdl
                    )
                rtp_llm_ops.per_token_group_quant_fp8_v3(
                    scratch, quant, scales, enable_pdl=use_pdl
                )
                rtp_llm_ops.deepselect_bf16_dense(
                    scratch, ends, indices, True, enable_pdl=use_pdl
                )

            with torch.cuda.stream(stream):
                chain()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    chain()
                for iteration in range(12):
                    source.normal_(mean=iteration / 10)
                    # Alternate short-row scalar loads and the TMA selection path.
                    bounds.fill_(300 if iteration % 2 else 5120)
                    graph.replay()
                    rtp_llm_ops.per_token_group_quant_fp8_v3(
                        source, ref_quant, ref_scales
                    )
                    rtp_llm_ops.deepselect_bf16_dense(source, bounds, ref_indices, True)
                    torch.testing.assert_close(
                        quant.view(torch.uint8),
                        ref_quant.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0)

                    def selected_scores(index):
                        values = source.float().gather(1, index.long().clamp_min(0))
                        return (
                            values.masked_fill(index < 0, -float("inf"))
                            .sort(dim=1)
                            .values
                        )

                    torch.testing.assert_close(
                        selected_scores(indices),
                        selected_scores(ref_indices),
                        rtol=0,
                        atol=0,
                    )
        torch.cuda.current_stream().wait_stream(stream)

    def test_dynamic_inputs_and_inplace_norm_on_nondefault_stream(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("V4.1 norm quant requires SM10x")
        source = torch.randn((17, 5120), device="cuda", dtype=torch.bfloat16)
        scratch = torch.empty_like(source)
        weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for setting in ("0", "1"):
            with patch.dict(
                os.environ, {"DSV41_ENABLE_PDL": setting, "DISABLE_PDL": "0"}
            ):

                def chain():
                    _delayed_producer[(triton.cdiv(source.numel(), 256),)](
                        source, scratch, source.numel(), **_pdl.launch_options(source)
                    )
                    first = rmsnorm_group32_quant(scratch, weight, out_norm=scratch)
                    second = rmsnorm_group32_quant(first[0], weight)
                    return first[1:], second

                with torch.cuda.stream(stream):
                    chain()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = chain()
                    for iteration in range(12):
                        source.normal_(mean=iteration / 10)
                        graph.replay()
                        with patch.dict(os.environ, {"DSV41_ENABLE_PDL": "0"}):
                            first = rmsnorm_group32_quant(source, weight)
                            expected = (
                                first[1:],
                                rmsnorm_group32_quant(first[0], weight),
                            )
                        for left_group, right_group in zip(actual, expected):
                            for left, right in zip(left_group, right_group):
                                if left.dtype == torch.float8_e4m3fn:
                                    left, right = left.view(torch.uint8), right.view(
                                        torch.uint8
                                    )
                                torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == "__main__":
    unittest.main()
