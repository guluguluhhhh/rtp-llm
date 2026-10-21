"""No-Bazel ABI isolation: unchanged mixin AST, real GPU helpers and outputs.

Only module-level config imports are isolated. No executable class or function
body is edited. The original unit-test draft hooks remain test doubles, and are
not used for the reported preparation-region performance measurements.
"""

import ast
import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(os.environ["RTP_MTP_SOURCE_ROOT"])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/opt/conda310/lib/python3.10/site-packages/rtp_llm/libs")
import librtp_compute_ops

name = "rtp_llm.models_py.speculative.dspark_proposer_mixin"
path = ROOT / "rtp_llm/models_py/speculative/dspark_proposer_mixin.py"
kvpath = ROOT / "rtp_llm/models_py/modules/dsv4/kv_cache_utils.py"
spec = importlib.util.spec_from_file_location("actual_cache_utils", kvpath)
kv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv)
m = ModuleType(name)
m.__file__ = str(path)
m.__package__ = name.rpartition(".")[0]
m.__dict__.update(
    torch=torch,
    primary_attention_inputs=kv.primary_attention_inputs,
    PyModelOutputs=librtp_compute_ops.PyModelOutputs,
)
tree = ast.parse(path.read_text())
selected = ast.Module(
    body=[
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        or isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
    ],
    type_ignores=[],
)
ast.fix_missing_locations(selected)
sys.modules[name] = m
exec(compile(selected, str(path), "exec"), m.__dict__)
from rtp_llm.models_py.speculative import _dspark_prepare_triton as prepare
from rtp_llm.models_py.speculative.test.test_dspark_proposer import (
    _propose_inputs,
    _ProposeProposer,
)


class IntegratedDispatch(unittest.TestCase):
    def test_original_entrypoints_dispatch_and_graph_replay(self):
        device = torch.device("cuda:0")
        b, w = 129, 5
        ids = torch.arange(b * w, device=device, dtype=torch.int64)
        prefix = torch.arange(b, device=device, dtype=torch.int64)
        inputs = _propose_inputs(
            ids, torch.full((b,), w, device=device, dtype=torch.int32), prefix
        )
        # A stale host field must not replace the CUDA-resident metadata mirror.
        inputs.attention_inputs.prefix_lengths_device = prefix
        inputs.attention_inputs.prefix_lengths = torch.zeros(b, dtype=torch.int64)
        owner = _ProposeProposer(width=w)
        captures = {}
        for enabled in ["0", "1"]:
            with patch.dict(os.environ, DSV4_FUSED_DSPARK_QUERY_INPUTS=enabled):
                for _ in range(3):
                    owner.run_propose_step(inputs, None, device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = owner.run_propose_step(inputs, None, device)
                captures[enabled] = (graph, owner.query_call, output)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                graph.replay()
                torch.cuda.synchronize()
            names = [
                e.name
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            self.assertEqual(
                any("_dspark_query_inputs_kernel" in x for x in names),
                enabled == "1",
                names,
            )
        for phase in range(3):
            prefix.add_(31)
            prefix[::7] = 0
            prefix[1] = 2**31 + 9
            ids.add_(17)
            for graph, _, _ in captures.values():
                graph.replay()
            torch.cuda.synchronize()
            for a, b in zip(captures["0"][1], captures["1"][1]):
                self.assertTrue(torch.equal(a, b), phase)
        starts = torch.tensor([2, 2, 5, 8], dtype=torch.int64, device=device)
        lengths = torch.tensor([0, 2, 1, 0], dtype=torch.int64, device=device)
        ends = torch.tensor([9, 12, 20, 99], dtype=torch.int64, device=device)
        with patch.dict(os.environ, DSV4_FUSED_DSPARK_COMMIT_ROWS="0"):
            baseline = m.map_context_rows(starts, lengths, ends, 10)
        with patch.dict(os.environ, DSV4_FUSED_DSPARK_COMMIT_ROWS="1"):
            with patch.object(
                prepare, "try_map_context_rows", wraps=prepare.try_map_context_rows
            ) as called:
                got = m.map_context_rows(starts, lengths, ends, 10)
                self.assertEqual(called.call_count, 1)
        for a, b in zip(baseline, got):
            self.assertTrue(torch.equal(a, b))
        # Unsupported int32 mapping must execute the original fallback unchanged.
        expected = m.map_context_rows(
            starts.cpu().int(), lengths.cpu().int(), ends.cpu().int(), 10
        )
        got = m.map_context_rows(starts.int(), lengths.int(), ends.int(), 10)
        for a, b in zip(expected, got):
            self.assertTrue(torch.equal(a.to(device), b))


suite = unittest.TestSuite()
loader = unittest.TestLoader()
for test in [
    "rtp_llm.models_py.speculative.test.test_dspark_prepare_triton",
    "rtp_llm.models_py.speculative.test.test_dspark_proposer",
]:
    suite.addTests(loader.loadTestsFromName(test))
suite.addTests(loader.loadTestsFromTestCase(IntegratedDispatch))
result = unittest.TextTestRunner(verbosity=2).run(suite)
meta = dict(
    tests=result.testsRun,
    failures=len(result.failures),
    errors=len(result.errors),
    skips=len(result.skipped),
    source_sha256={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            path,
            kvpath,
            Path(prepare.__file__),
            Path(__file__),
            ROOT / "rtp_llm/models_py/speculative/test/test_dspark_prepare_triton.py",
            ROOT / "rtp_llm/models_py/speculative/test/test_dspark_proposer.py",
        ]
    },
    selected_ast_sha256=hashlib.sha256(
        ast.dump(selected, include_attributes=False).encode()
    ).hexdigest(),
    scope="unchanged mixin class/function AST + actual cache normalizer + actual PyModelOutputs; unit-test draft math is explicitly a test double; no full MTP model claim",
)
(Path(__file__).parent / "contract_tests.json").write_text(json.dumps(meta, indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
