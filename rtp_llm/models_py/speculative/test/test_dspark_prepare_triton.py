import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.speculative._dspark_prepare_triton import (
    try_map_context_rows,
    try_prepare_proposal,
)


class PrepareAdmissionTest(unittest.TestCase):
    def test_cpu_stays_on_eager_path(self):
        ids = torch.zeros(10, dtype=torch.int32)
        prefix = torch.ones(2, dtype=torch.int64)
        self.assertIsNone(try_prepare_proposal(ids, prefix, width=5, noise_token_id=7))
        self.assertIsNone(try_map_context_rows(prefix, prefix, prefix, 10))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrepareCudaTest(unittest.TestCase):
    def test_proposal_graph_reads_changed_buffers_and_preserves_cast_order(self):
        for width in (1, 3, 5, 8):
            for ids_dtype in (torch.int32, torch.int64):
                for prefix_dtype in (torch.int32, torch.int64):
                    with self.subTest(width=width, ids=ids_dtype, prefix=prefix_dtype):
                        batch = 129
                        ids = torch.arange(
                            batch * width, device="cuda", dtype=ids_dtype
                        )
                        prefix = torch.arange(
                            batch + 3, device="cuda", dtype=prefix_dtype
                        )
                        for _ in range(3):
                            try_prepare_proposal(
                                ids, prefix, width=width, noise_token_id=7
                            )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            out = try_prepare_proposal(
                                ids, prefix, width=width, noise_token_id=7
                            )
                        for phase in range(3):
                            ids.add_(31)
                            prefix.add_(17)
                            prefix[::7] = 0
                            if prefix_dtype == torch.int64:
                                prefix[0] = 2**32
                                prefix[1] = 2**31 + 3
                            original_ids, original_prefix = ids.clone(), prefix.clone()
                            for value in out:
                                if value.data_ptr() != prefix.data_ptr():
                                    value.fill_(
                                        False if value.dtype == torch.bool else -777
                                    )
                            graph.replay()
                            expected_ids = torch.full(
                                (batch, width), 7, device="cuda", dtype=torch.int32
                            )
                            expected_ids[:, 0] = ids.to(torch.int32).view(batch, width)[
                                :, 0
                            ]
                            expected_prefix = prefix[:batch].to(torch.int32)
                            expected_positions = expected_prefix.long()[
                                :, None
                            ] + torch.arange(width, device="cuda")
                            for got, expected in zip(
                                out,
                                (
                                    expected_ids,
                                    expected_positions,
                                    expected_prefix,
                                    expected_prefix > 0,
                                ),
                            ):
                                self.assertEqual(got.dtype, expected.dtype)
                                self.assertTrue(
                                    torch.equal(got, expected), (width, phase)
                                )
                            self.assertTrue(torch.equal(ids, original_ids))
                            self.assertTrue(torch.equal(prefix, original_prefix))

    def test_map_graph_duplicates_holes_empty_requests_and_large_positions(self):
        batch, rows = 129, 1025
        lengths = torch.arange(batch, device="cuda", dtype=torch.int64).remainder_(7)
        starts = lengths.cumsum(0) - lengths
        ends = torch.arange(batch, device="cuda", dtype=torch.int64) + 2**35
        for _ in range(3):
            try_map_context_rows(starts, lengths, ends, rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = try_map_context_rows(starts, lengths, ends, rows)
        for phase in range(4):
            lengths.fill_(phase)
            lengths[::3] = 0
            starts.copy_(lengths.cumsum(0) - lengths + 3)
            ends.add_(19)
            for value in out:
                value.fill_(-777)
            graph.replay()
            row = torch.arange(rows, device="cuda")
            request = torch.searchsorted(starts, row, right=True) - 1
            safe = request.clamp(0, batch - 1)
            valid = (
                (request >= 0)
                & (row >= starts[safe])
                & (row < starts[safe] + lengths[safe])
            )
            position = ends[safe] - lengths[safe] + row - starts[safe]
            expected = (
                torch.where(valid, request, -1).int(),
                torch.where(valid, position, -1).int(),
            )
            for got, want in zip(out, expected):
                self.assertTrue(torch.equal(got, want), phase)

    def test_structured_mapping_example(self):
        starts = torch.tensor([2, 2, 5, 8], device="cuda", dtype=torch.int64)
        lengths = torch.tensor([0, 2, 1, 0], device="cuda", dtype=torch.int64)
        ends = torch.tensor([9, 12, 20, 99], device="cuda", dtype=torch.int64)
        request, position = try_map_context_rows(starts, lengths, ends, 10)
        self.assertEqual(request.tolist(), [-1, -1, 1, 1, -1, 2, -1, -1, -1, -1])
        self.assertEqual(position.tolist(), [-1, -1, 10, 11, -1, 19, -1, -1, -1, -1])

    def test_empty_and_unsupported_inputs(self):
        ids = torch.zeros(10, dtype=torch.int32, device="cuda")
        prefix = torch.ones(2, dtype=torch.int64, device="cuda")
        self.assertIsNone(
            try_prepare_proposal(ids[::2], prefix, width=5, noise_token_id=7)
        )
        self.assertIsNone(try_prepare_proposal(ids, prefix, width=3, noise_token_id=7))
        self.assertIsNone(
            try_prepare_proposal(ids, prefix[:1], width=5, noise_token_id=7)
        )
        self.assertIsNone(
            try_map_context_rows(prefix.int(), prefix.int(), prefix.int(), 10)
        )
        self.assertIsNone(try_map_context_rows(prefix[::2], prefix, prefix, 10))
        self.assertIsNone(try_map_context_rows(prefix[:0], prefix[:0], prefix[:0], 1))
        out = try_map_context_rows(prefix[:0], prefix[:0], prefix[:0], 0)
        self.assertEqual([t.numel() for t in out], [0, 0])
        out = try_prepare_proposal(ids[:0], prefix[:0], width=5, noise_token_id=7)
        self.assertEqual([tuple(t.shape) for t in out], [(0, 5), (0, 5), (0,), (0,)])
        with patch.dict(
            os.environ,
            DSV4_FUSED_DSPARK_QUERY_INPUTS="0",
            DSV4_FUSED_DSPARK_COMMIT_ROWS="0",
        ):
            self.assertIsNone(
                try_prepare_proposal(ids, prefix, width=5, noise_token_id=7)
            )
            self.assertIsNone(try_map_context_rows(prefix, prefix, prefix, 10))


if __name__ == "__main__":
    unittest.main()
