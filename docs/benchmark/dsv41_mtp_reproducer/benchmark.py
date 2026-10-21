"""Actual DSpark integer preparation regions; no full-MTP timing claim.

Baseline proposal statements and map_context_rows are compiled unchanged from
an archived source AST. Both variants include the same actual metadata consumer.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--batches", nargs="+", type=int, default=[128, 192, 256, 384, 512])
p.add_argument("--widths", nargs="+", type=int, default=[1, 5, 8])
p.add_argument("--modes", nargs="+", default=["baseline"])
p.add_argument("--samples", type=int, default=30)
p.add_argument("--reverse", action="store_true")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
assert torch.cuda.get_device_capability() == (10, 3)
os.environ["DSV4_FUSED_DSPARK_METADATA"] = "1"
ROOT = Path(os.environ["RTP_MTP_SOURCE_ROOT"])
source = HERE / "baseline_dspark_proposer_mixin.py"
tree = ast.parse(source.read_text())
map_fn = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "map_context_rows"
)
cls = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "DSparkProposerMixin"
)
method = next(
    n
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == "run_propose_step"
)
start = next(
    i
    for i, n in enumerate(method.body)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "prefix_lengths" for t in n.targets)
)
end = (
    next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "active_requests" for t in n.targets
        )
    )
    + 1
)
args_node = ast.parse(
    "def proposal_baseline(self,inputs,prefix_lengths_source,batch_size,width,device): pass"
).body[0]
args_node.body = method.body[start:end] + [
    ast.parse("return query_ids,query_positions,prefix_lengths,active_requests").body[0]
]
mod = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        map_fn,
        args_node,
    ],
    type_ignores=[],
)
ast.fix_missing_locations(mod)
ns = {"torch": torch}
exec(compile(mod, str(source), "exec"), ns)
proposal_baseline = ns["proposal_baseline"]
map_baseline = ns["map_context_rows"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


swa = load(
    "actual_swa_ops", ROOT / "rtp_llm/models_py/modules/dsv41/fp8/_swa_ops_triton.py"
)
metadata = load(
    "actual_dspark_metadata",
    ROOT / "rtp_llm/models_py/modules/dsv41/_dspark_metadata_triton.py",
)
if "fused" in a.modes:
    fused = load(
        "candidate_dspark_prepare",
        ROOT / "rtp_llm/models_py/speculative/_dspark_prepare_triton.py",
    )

# Load official numeric source for comparison, even for these integer outputs.
testing = Path("/work/DeepGEMM-runtime-559d79f")
assert (
    subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={testing}",
            "-C",
            str(testing),
            "remote",
            "get-url",
            "origin",
        ],
        text=True,
    ).strip()
    == "https://github.com/deepseek-ai/DeepGEMM.git"
)
assert (
    subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={testing}",
            "-C",
            str(testing),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    == "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
)
assert not subprocess.check_output(
    [
        "git",
        "-c",
        f"safe.directory={testing}",
        "-C",
        str(testing),
        "status",
        "--porcelain",
        "--",
        "deep_gemm/testing",
        "tests/test_bf16.py",
    ],
    text=True,
).strip()
numeric = load("official_numeric", testing / "deep_gemm/testing/numeric.py")
print(
    "TESTING https://github.com/deepseek-ai/DeepGEMM.git 559d79fb6994a58b8a15b4b93bf13ccc16edf247",
    flush=True,
)


def kernels(path):
    return sorted(
        [
            e
            for e in json.loads(path.read_text())["traceEvents"]
            if e.get("cat") == "kernel" and e.get("ph") == "X"
        ],
        key=lambda e: e["ts"],
    )


def profile(fn, path, repeats=1, flush=None):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for i in range(repeats):
            with torch.profiler.record_function(f"sample_{i}"):
                if flush is not None:
                    flush.zero_()
                fn()
                torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    return kernels(path)


cases = []
checks = []
input_hashes = []
torch.manual_seed(20261021)
for b in a.batches:
    for width in a.widths:
        for dtype in [torch.int32] if width != 5 else [torch.int32, torch.int64]:
            ids = torch.randint(0, 129280, (b * width,), device="cuda", dtype=dtype)
            prefix = torch.randint(0, 4096, (b,), device="cuda", dtype=torch.int32)
            prefix[::7] = 0
            table = torch.arange(
                1, b * 40 + 1, device="cuda", dtype=torch.int32
            ).reshape(b, 40)
            table[:, 0] = 0
            inputs = SimpleNamespace(input_ids=ids)
            owner = SimpleNamespace(_dspark_noise_token_id=7)

            def proposal(
                mode,
                ids=ids,
                prefix=prefix,
                table=table,
                b=b,
                width=width,
                inputs=inputs,
                owner=owner,
            ):
                if mode == "baseline":
                    out = proposal_baseline(owner, inputs, prefix, b, width, ids.device)
                else:
                    out = fused.try_prepare_proposal(
                        ids, prefix, width=width, noise_token_id=7
                    )
                    assert out is not None
                qi, qp, pl, active = out
                md = metadata.try_build_dspark_metadata(
                    qp,
                    pl,
                    active,
                    table,
                    gamma=width,
                    window_size=128,
                    entries_per_block=128,
                    tokens_per_block=128,
                )
                assert md is not None
                return (*out, *md)

            cases.append(
                dict(
                    name=f'proposal_b{b}_g{width}_{str(dtype).split(".")[-1]}',
                    fn=proposal,
                    region="proposal inputs + existing metadata consumer",
                    B=b,
                    width=width,
                    ids=ids,
                    prefix=prefix,
                    table=table,
                )
            )
        # Actual map_context_rows boundary takes int64 starts/lengths/ends from caller.
        length = torch.randint(0, width + 2, (b,), device="cuda", dtype=torch.int64)
        starts = length.cumsum(0) - length
        ends = torch.randint(100, 5000, (b,), device="cuda", dtype=torch.int64) + length
        rows = b * (width + 1)
        map_table = torch.arange(
            1, b * 40 + 1, device="cuda", dtype=torch.int32
        ).reshape(b, 40)
        map_table[:, 0] = 0

        def mapping(
            mode, starts=starts, length=length, ends=ends, rows=rows, table=map_table
        ):
            if mode == "baseline":
                out = map_baseline(starts, length, ends, rows)
            else:
                out = fused.try_map_context_rows(starts, length, ends, rows)
                assert out is not None
            end32 = ends.to(torch.int32)
            slots = swa.compute_swa_slot_mapping_from_positions(
                table, *out, end32, rows, 128, 128, 128
            )
            return (*out, end32, slots)

        cases.append(
            dict(
                name=f"commit_map_b{b}_g{width}",
                fn=mapping,
                region="commit row mapping + ends cast + existing SWA slot consumer",
                B=b,
                width=width,
                starts=starts,
                lengths=length,
                ends=ends,
                table=map_table,
            )
        )

for case in cases:
    graphs = {}
    outputs = {}
    for mode in a.modes:
        fn = lambda mode=mode: case["fn"](mode)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        graphs[mode] = graph
        outputs[mode] = out
    # Same-address metadata changes, including duplicate starts and padded rows.
    for phase in range(3):
        if "prefix" in case:
            case["prefix"].add_(131)
            case["prefix"][phase::9] = 0
            case["ids"].add_(17).remainder_(129280)
            case["table"].add_(1)
            case["table"][:, phase] = 0
        else:
            ll = case["lengths"]
            ll.copy_(
                torch.arange(ll.numel(), device="cuda").remainder_(case["width"] + 2)
            )
            if phase == 1:
                ll.zero_()  # duplicates all zero; final duplicate owns searchsorted result
            if phase == 2:
                ll.fill_(case["width"] + 1)
                ll[::11] = 0
            case["starts"].copy_(ll.cumsum(0) - ll)
            case["ends"].add_(29)
        ref = case["fn"]("baseline")
        torch.cuda.synchronize()
        input_hashes.append(
            dict(
                case=case["name"],
                phase=phase,
                tensors={
                    key: dict(
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        sha256=hashlib.sha256(
                            value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                        ).hexdigest(),
                    )
                    for key, value in case.items()
                    if isinstance(value, torch.Tensor)
                },
            )
        )
        for mode, graph in graphs.items():
            # Prove outputs are written by each captured graph, not stale buffers.
            for j, out in enumerate(outputs[mode]):
                if any(
                    isinstance(t, torch.Tensor) and t.data_ptr() == out.data_ptr()
                    for t in case.values()
                ):
                    continue  # caller-owned prefix alias
                out.fill_(False if out.dtype == torch.bool else -777)
            graph.replay()
            torch.cuda.synchronize()
            for j, (got, want) in enumerate(zip(outputs[mode], ref)):
                assert got.shape == want.shape and got.dtype == want.dtype
                assert torch.equal(got, want), (case["name"], mode, phase, j)
                diff = (
                    float(numeric.calc_diff(got.double(), want.double()))
                    if got.numel()
                    else 0
                )
                assert diff < 1e-12
                checks.append(
                    dict(
                        case=case["name"],
                        mode=mode,
                        phase=phase,
                        output=j,
                        calc_diff=diff,
                        bitwise_equal=True,
                    )
                )
    case["graphs"] = graphs
    case["outputs"] = outputs
    print("CORRECT", case["name"], flush=True)
(a.out / "input_hashes.json").write_text(json.dumps(input_hashes, indent=2))
(a.out / "correctness.json").write_text(json.dumps(checks, indent=2))
flush = torch.empty(int(8e9 // 4), device="cuda", dtype=torch.int32)
fk = profile(flush.zero_, a.out / "flush_signature.json")
flush_names = [e["name"] for e in fk]
assert len(fk) == 4
results = []
for case in cases[::-1] if a.reverse else cases:
    for mode in a.modes[::-1] if a.reverse else a.modes:
        g = case["graphs"][mode]
        path = a.out / f"{case['name']}_{mode}_signature.json"
        sig = profile(g.replay, path)
        names = [e["name"] for e in sig]
        assert names
        path = a.out / f"{case['name']}_{mode}_trace.json"
        ev = profile(g.replay, path, repeats=a.samples, flush=flush)
        n = len(sig)
        nf = len(fk)
        assert len(ev) == a.samples * (n + nf)
        samples = []
        for i in range(a.samples):
            block = ev[i * (n + nf) : (i + 1) * (n + nf)]
            ff = block[:nf]
            kk = block[nf:]
            assert [e["name"] for e in ff] == flush_names
            assert [e["name"] for e in kk] == names
            assert max(e["ts"] + e["dur"] for e in ff) <= min(e["ts"] for e in kk)
            t0 = min(e["ts"] for e in kk)
            end = max(e["ts"] + e["dur"] for e in kk)
            samples.append(
                dict(
                    sample=i,
                    envelope_us=end - t0,
                    kernels=[
                        dict(
                            name=e["name"],
                            start_offset_us=e["ts"] - t0,
                            duration_us=e["dur"],
                            resource=e["args"],
                        )
                        for e in kk
                    ],
                )
            )
        rec = dict(
            case=case["name"],
            B=case["B"],
            width=case["width"],
            rows=case["B"]
            * (
                case["width"] + 1
                if case["name"].startswith("commit")
                else case["width"]
            ),
            region=case["region"],
            mode=mode,
            kernel_count=n,
            metric=(
                "single_kernel_body_us" if n == 1 else "exact_graph_kernel_envelope_us"
            ),
            sample_count=a.samples,
            cache_policy="8GB flush -> one Graph replay -> fence",
            median_us=statistics.median(s["envelope_us"] for s in samples),
            first_kernel=names[0],
            last_kernel=names[-1],
            event_span_diagnostic=False,
            samples=samples,
        )
        results.append(rec)
        (a.out / f"{case['name']}_{mode}_metrics.json").write_text(
            json.dumps(rec, indent=2)
        )
        print(
            "TIMING",
            json.dumps({k: v for k, v in rec.items() if k != "samples"}),
            flush=True,
        )
meta = dict(
    torch=torch.__version__,
    runtime_capability=torch.cuda.get_device_capability(),
    sms=torch.cuda.get_device_properties(0).multi_processor_count,
    source_sha256={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [source, Path(__file__), Path(metadata.__file__), Path(swa.__file__)]
        + ([Path(fused.__file__)] if "fused" in a.modes else [])
    },
    baseline_ast_sha256=hashlib.sha256(
        ast.dump(mod, include_attributes=False).encode()
    ).hexdigest(),
    scope="only actual integer DSpark preparation regions, not draft Transformer, sampling, or whole MTP",
    warmup=3,
    replay_warmup=20,
    clock_policy="shared unlocked GPU",
    reverse=a.reverse,
)
(a.out / "environment.json").write_text(json.dumps(meta, indent=2))
(a.out / "summary.json").write_text(json.dumps(results, indent=2))
