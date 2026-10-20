"""Exact finite publication + slot lookup, including changed-address Graph state."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as selector
from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
    triton_convert_req_index_to_global_index,
)


class SlotPlanCPU(unittest.TestCase):
    def test_disabled_and_cpu_fallback(self):
        req = torch.zeros(3, dtype=torch.int32)
        table = torch.ones(3, 5, dtype=torch.int32)
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {"DSV41_FUSED_TOPK_SLOTS": enabled}):
                self.assertIsNone(
                    selector.make_slot_plan(
                        req, table, entries=128, logical_entries=128
                    )
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TopKSlotsCUDA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261020)
        env = patch.dict(os.environ, {"DSV41_FUSED_TOPK_SLOTS": "1"})
        env.start()
        self.addCleanup(env.stop)

    def inputs(self, k, dtype):
        rows, width = 9, 4096
        logits = torch.randn(rows, width + 256, device="cuda", dtype=dtype)[:, :width]
        logits[:, 9] = float("inf")
        logits[:, 17] = float("nan")
        logits[:, 21] = -float("inf")
        lengths = torch.tensor(
            [0, 1, k - 1, k, k + 1, width, 1023, 2049, 513],
            device="cuda",
            dtype=torch.int32,
        )
        selected = torch.randint(
            -1, width + 128, (rows, k), device="cuda", dtype=torch.int32
        )
        selected[:, :6] = torch.tensor([-1, 9, 17, 21, 127, width + 1], device="cuda")
        requests = torch.tensor(
            [2, 0, 3, 2, 1, 0, 3, 1, 2], device="cuda", dtype=torch.int32
        )
        table = torch.randint(1, 100, (4, 35), device="cuda", dtype=torch.int32)[:, :33]
        table[:, 0] = 0
        table[:, 2] = -1
        return logits, lengths, selected, requests, table

    @torch.inference_mode()
    def test_exact_publication_and_slot_graph(self):
        for k in (512, 1024, 2048):
            for dtype in (torch.float32, torch.bfloat16):
                for entries in (64, 128):
                    with self.subTest(k=k, dtype=dtype, entries=entries):
                        logits, lengths, selected, req, table = self.inputs(k, dtype)
                        expected = selected.clone()
                        actual = selected.clone()
                        slots = torch.empty_like(actual)

                        def run():
                            selector._finite_topk_slots_kernel[
                                (9, selector.triton.cdiv(k, 256))
                            ](
                                logits,
                                lengths,
                                actual,
                                req,
                                table,
                                slots,
                                logits.shape[1],
                                logits.stride(0),
                                k,
                                table.shape[0],
                                table.shape[1],
                                table.stride(0),
                                entries,
                                entries,
                                256,
                            )

                        run()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        for phase in range(3):
                            selected.copy_(selected.roll(1, 1))
                            req.copy_(req.roll(1))
                            table[:, 1].add_(1)
                            logits[:, 9] = float("inf") if phase % 2 else 1.0
                            expected.copy_(selected)
                            actual.copy_(selected)
                            slots.fill_(-777)
                            selector._finite_topk_kernel[
                                (9, selector.triton.cdiv(k, 256))
                            ](
                                logits,
                                lengths,
                                expected,
                                logits.shape[1],
                                logits.stride(0),
                                k,
                                256,
                            )
                            reference_slots = triton_convert_req_index_to_global_index(
                                req,
                                table,
                                expected,
                                ENTRIES_PER_BLOCK=entries,
                                TOKENS_PER_BLOCK_FOR_BLOCK_TABLE=entries,
                                NUM_TOPK_TOKENS=k,
                                BLOCK_N=128,
                            )
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            torch.testing.assert_close(
                                slots, reference_slots, rtol=0, atol=0
                            )

    @torch.inference_mode()
    def test_select_tokens_publishes_local_and_forward_local_slots(self):
        logits, lengths, selected, req, table = self.inputs(512, torch.float32)
        plan = selector.make_slot_plan(req, table, entries=128, logical_entries=128)
        expected = selected.clone()
        selector._finite_topk_kernel[(9, 2)](
            logits, lengths, expected, 4096, logits.stride(0), 512, 256
        )
        # Selection is unchanged by this optimization. A deterministic stand-in
        # isolates its publication interface from cutoff ties in native TopK.
        with patch.object(
            selector.deepselect, "try_select_tokens", return_value=None
        ), patch.object(
            selector.rtp_llm_ops,
            "topk_v3",
            side_effect=lambda x, n, out, *a: out.copy_(selected),
        ), patch.object(
            selector, "_get_topk_workspace", return_value=None
        ):
            actual = selector.select_tokens(logits, lengths, 512, slot_plan=plan)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNotNone(plan.slots)
        self.assertNotEqual(actual.data_ptr(), plan.slots.data_ptr())
        for requests, entries, logical in (
            (req.long(), 128, 128),
            (req[::2], 128, 128),
            (req, 128, 64),
        ):
            self.assertIsNone(
                selector.make_slot_plan(
                    requests, table, entries=entries, logical_entries=logical
                )
            )


if __name__ == "__main__":
    unittest.main()
