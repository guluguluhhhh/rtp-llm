"""SM103 merged QB and finite/slots qualification; exact cold-L2 Graph activities.

Requires matching RTP native libraries and the branch's pinned DeepGEMM wheel.
Set DG_TESTING_ROOT to the clean official checkout pinned below. No JIT backend
is installed by this runner; warmup, allocations and captures precede timing.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

import torch
import deep_gemm as dg
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor as pack

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_qb_kv as producer
from rtp_llm.models_py.modules.dsv41.fp8._v41_qkv_rope_cache_triton import (
    try_fused_qkv_rope_cache,
)

TESTING_COMMIT = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
TESTING_URL = "https://github.com/deepseek-ai/DeepGEMM.git"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument(
    "--testing-root", type=Path, default=os.environ.get("DG_TESTING_ROOT")
)
parser.add_argument("--batches", nargs="+", type=int, default=[128, 192, 256, 512])
parser.add_argument(
    "--cases",
    nargs="+",
    choices=["pure", "qb-kv", "topk-slots"],
    default=["pure", "qb-kv", "topk-slots"],
)
parser.add_argument("--samples", type=int, default=30)
parser.add_argument("--reverse", action="store_true")
parser.add_argument(
    "--bandwidth",
    action="store_true",
    help="Also profile a separate 1 GB streaming read/write kernel",
)
args = parser.parse_args()
if args.testing_root is None:
    parser.error("--testing-root or DG_TESTING_ROOT is required")
args.out.mkdir(parents=True, exist_ok=False)


def git(*parts):
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={args.testing_root}",
            "-C",
            str(args.testing_root),
            *parts,
        ],
        text=True,
    ).strip()


assert git("remote", "get-url", "origin") == TESTING_URL
assert git("rev-parse", "HEAD") == TESTING_COMMIT
assert not git("status", "--porcelain", "--", "deep_gemm/testing", "tests/test_bf16.py")


def official(name, relative):
    spec = importlib.util.spec_from_file_location(name, args.testing_root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


numeric = official("qb_official_numeric", "deep_gemm/testing/numeric.py")
official_bench = official("qb_official_bench", "deep_gemm/testing/bench.py")
assert torch.cuda.get_device_capability() == (10, 3)
assert torch.cuda.get_device_properties(0).multi_processor_count == 148
assert producer._ops().dsv41_qb_kv_available()
os.environ["DSV41_CUSTOM_QB"] = "1"
os.environ["DSV41_FUSED_QB_KV"] = "1"
dg.set_pdl(False)
torch.manual_seed(20261020)
torch.backends.cuda.matmul.allow_tf32 = False
metadata = {
    "testing_repository": TESTING_URL,
    "testing_commit": TESTING_COMMIT,
    "testing_root": str(args.testing_root),
    "deepgemm": dg.__file__,
    "deepgemm_version": getattr(dg, "__version__", None),
    "capability": torch.cuda.get_device_capability(),
    "SMs": 148,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "argv": sys.argv,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "mode": "CUDA Graph replay",
    "cache": "8GB flush -> 1 replay -> fence",
    "pdl": False,
    "shape": {
        "N": 36864,
        "K": 1280,
        "raw_KV_row_stride": 1792,
        "cache_entry_bytes": 528,
    },
    "clock_policy": "unlocked/shared GPU; paired reverse-order rounds required",
    "scope": "isolated pure/QB+KV/finite+slots regions; excludes attention and full-layer stream competition",
}
source_root = Path(__file__).resolve().parents[3]
source_files = [
    Path(producer.__file__),
    *sorted(
        (source_root / "rtp_llm/models_py/bindings/cuda/kernels/dsv41_qb_kv").glob(
            "*.cuh"
        )
    ),
]
source_files.append(
    source_root / "rtp_llm/models_py/bindings/cuda/kernels/dsv41_qb_kv.cu"
)
metadata["source_sha256"] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
}
binaries = list(Path(dg.__file__).parent.glob("*.so"))
backend_file = getattr(producer._ops(), "__file__", None)
if backend_file:
    binaries.append(Path(backend_file))
metadata["binary_sha256"] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries
}
(args.out / "environment.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)


def extract_kernels(path):
    ev = json.loads(path.read_text())["traceEvents"]
    return sorted(
        [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"],
        key=lambda e: e["ts"],
    )


def profile_graph(graph, stream, path, flush=None, samples=1):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for sample in range(samples):
            with torch.profiler.record_function(f"baseline_sample_{sample}"):
                with torch.cuda.stream(stream):
                    if flush is not None:
                        flush.zero_()
                    if graph is not None:
                        graph.replay()
                stream.synchronize()
    prof.export_chrome_trace(str(path))
    return extract_kernels(path)


def benchmark(fn, label, flush):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    # Invocation outputs are allocated once from the graph-private pool;
    # replay reuses their addresses, with no allocator/JIT GPU nodes.
    with torch.cuda.graph(graph, stream=stream):
        result = fn()
    with torch.cuda.stream(stream):
        for _ in range(20):
            graph.replay()
    stream.synchronize()
    signature = profile_graph(graph, stream, args.out / f"{label}_signature.json")
    expected = [e["name"] for e in signature]
    assert expected, "No CUDA activity: exact benchmark blocked"
    flush_sig = profile_graph(
        None, stream, args.out / f"{label}_flush_signature.json", flush
    )
    flush_names = [e["name"] for e in flush_sig]
    assert flush_names, "L2 flush has no activity"
    events = profile_graph(
        graph, stream, args.out / f"{label}_trace.json", flush, args.samples
    )
    group_len = len(flush_names) + len(expected)
    assert len(events) == args.samples * group_len, (
        len(events),
        args.samples,
        group_len,
    )
    samples = []
    for i in range(args.samples):
        group = events[i * group_len : (i + 1) * group_len]
        assert [
            e["name"] for e in group
        ] == flush_names + expected, "Replay signature changed / unrelated CUDA work"
        kernels = group[len(flush_names) :]
        first = min(e["ts"] for e in kernels)
        last = max(e["ts"] + e["dur"] for e in kernels)
        assert max(e["ts"] + e["dur"] for e in group[: len(flush_names)]) <= first
        assert (
            len({e["args"].get("stream") for e in group}) == 1
        ), "Unexpected side stream: audit required"
        samples.append(
            {
                "sample": i,
                "envelope_us": last - first,
                "kernels": [
                    {
                        "name": e["name"],
                        "start_offset_us": e["ts"] - first,
                        "duration_us": e["dur"],
                        "stream": e["args"].get("stream"),
                    }
                    for e in kernels
                ],
            }
        )
    metric = (
        "single_kernel_body_us"
        if len(expected) == 1
        else "exact_graph_kernel_envelope_us"
    )
    summary = {
        "label": label,
        "metric": metric,
        "mode": "CUDA Graph replay",
        "cache_policy": "cold-L2 8 GB",
        "kernel_count": len(expected),
        "first_kernel": expected[0],
        "last_kernel": expected[-1],
        "sample_count": len(samples),
        "statistic": "median",
        "median_us": statistics.median(s["envelope_us"] for s in samples),
        "min_us": min(s["envelope_us"] for s in samples),
        "max_us": max(s["envelope_us"] for s in samples),
        "samples": samples,
        "cuda_event_dispatch_diagnostic": False,
    }
    # A one-node graph's exact envelope equals that kernel's activity duration.
    summary["exact_graph_kernel_envelope_us"] = statistics.median(
        s["envelope_us"] for s in samples
    )
    (args.out / f"{label}_metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        "TIMING",
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ["samples", "first_kernel", "last_kernel"]
            }
        ),
        flush=True,
    )
    return summary, result


def quant(x):
    q, scales = per_token_cast_to_fp8(x.view(-1, x.shape[-1]), True, gran_k=32)
    return q.view(x.shape), pack(scales)


def dequant(pair):
    words = pair[1].to(torch.int64)
    exponents = torch.stack([(words >> (8 * i)) & 255 for i in range(4)], -1).flatten(
        -2
    )
    return pair[0].float() * torch.exp2(exponents.float() - 127).repeat_interleave(
        32, -1
    )


def q_compare(actual, expected):
    diff = float(numeric.calc_diff(actual, expected))
    assert diff < 1e-5, diff
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {"calc_diff": diff, "bitwise_equal_dg": True}


records = []


def record(m, region, name, fn, check):
    check(fn())
    timing, replayed = benchmark(fn, f"m{m}_{region}_{name}", flush)
    correctness = check(replayed)
    records.append(
        {
            "M": m,
            "region": region,
            "implementation": name,
            "correctness": correctness,
            "timing": {k: v for k, v in timing.items() if k != "samples"},
        }
    )
    (args.out / "summary.json").write_text(json.dumps(records, indent=2))


def ordered(pairs):
    return list(reversed(pairs)) if args.reverse else pairs


@torch.inference_mode()
def run_qb(m, merged):
    assert m in producer._CONFIGS, f"M{m} has no qualified custom configuration"
    x, xs = quant(torch.randn(m, 1, 1280, device="cuda", dtype=torch.bfloat16) * 0.1)
    qref = torch.empty(m, 36864, device="cuda", dtype=torch.bfloat16)
    dg.fp8_fp4_gemm_nt(
        (x.view(m, 1280), xs),
        (merged.weight, merged.weight_scales),
        qref,
        recipe=(1, 1, 32),
    )
    diff = float(
        numeric.calc_diff(
            qref,
            (
                dequant((x.view(m, 1280), xs))
                @ dequant((merged.weight, merged.weight_scales)).T
            ).bfloat16(),
        )
    )
    assert diff < 1e-5, diff

    def baseline_q():
        dg.fp8_fp4_gemm_nt(
            (x.view(m, 1280), xs),
            (merged.weight, merged.weight_scales),
            qbase,
            recipe=(1, 1, 32),
        )
        return qbase[:, :32768].view(m, 1, 64, 512), qbase[:, 32768:].view(m, 1, 4096)

    qbase = torch.empty_like(qref)

    def custom_q():
        q, iq = producer.try_project(merged, x, xs)
        return q.view(m, 1, 64, 512), iq

    def check_q(result):
        q_compare(result[0].reshape(m, 32768), qref[:, :32768])
        q_compare(result[1].reshape(m, 4096), qref[:, 32768:])
        return {"bitwise_equal_dg": True, "fp32_calc_diff": diff}

    if "pure" in args.cases:
        for name, fn in ordered([("dg", baseline_q), ("custom", custom_q)]):
            record(m, "pure", name, fn, check_q)
    if "qb-kv" not in args.cases:
        return
    raw = torch.randn(m, 1, 1792, device="cuda", dtype=torch.bfloat16)[..., -512:]
    gamma = (1 + torch.randn(512, device="cuda") * 0.1).bfloat16()
    pos = torch.arange(m, device="cuda", dtype=torch.int32) + 17
    angle = torch.randn(2048, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angle), angle)
    gathered = freqs[pos.long()].contiguous()
    slots = torch.arange(m, device="cuda", dtype=torch.int32) + 128
    slots[0] = -1
    pool = torch.empty_strided(
        (6, 128, 528), (128 * 528 + 128, 528, 1), device="cuda", dtype=torch.uint8
    )
    pool.fill_(173)

    def baseline():
        q, iq = baseline_q()
        result = try_fused_qkv_rope_cache(
            q,
            raw,
            gamma,
            pos,
            freqs,
            pool,
            slots,
            freqs_cis=gathered,
            apply_q_rope=False,
            pad_query_heads=64,
            eps=1e-6,
        )
        assert result is not None
        return q, iq, result.kv

    def fused():
        return producer.try_project_kv(
            merged, x, xs, raw, gamma, pos, freqs, pool, slots, eps=1e-6
        )

    baseline_result = baseline()
    kvref = baseline_result[2].clone()
    poolref = pool.clone()

    def check(result):
        check_q(result)
        torch.testing.assert_close(result[2], kvref, rtol=0, atol=0)
        torch.testing.assert_close(pool, poolref, rtol=0, atol=0)
        return {"Q_IndexQ_KV_cache_exact": True, "fp32_calc_diff": diff}

    pool.fill_(173)
    check(fused())
    for name, fn in ordered([("two_kernel", baseline), ("one_kernel", fused)]):
        record(m, "qb_kv", name, fn, check)


@torch.inference_mode()
def run_slots(m):
    from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as selector
    from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
        triton_convert_req_index_to_global_index as translate,
    )

    k, width, entries = 512, 4096, 128
    logits = torch.randn(m, width, device="cuda")
    logits[:, 9] = float("inf")
    lengths = torch.full((m,), width, device="cuda", dtype=torch.int32)
    lengths[::4] = 127
    local = torch.randint(0, width, (m, k), device="cuda", dtype=torch.int32)
    local[:, 0] = -1
    req = torch.arange(m, device="cuda", dtype=torch.int32).flip(0).contiguous()
    table = torch.randint(
        1, 10000, (m, width // entries), device="cuda", dtype=torch.int32
    )
    table[:, 0] = 0
    global_slots = torch.empty_like(local)

    def baseline():
        selector._finite_topk_kernel[(m, 2)](
            logits, lengths, local, width, width, k, 256
        )
        return translate(
            req,
            table,
            local,
            ENTRIES_PER_BLOCK=entries,
            TOKENS_PER_BLOCK_FOR_BLOCK_TABLE=entries,
            NUM_TOPK_TOKENS=k,
            BLOCK_N=128,
        )

    def fused():
        selector._finite_topk_slots_kernel[(m, 2)](
            logits,
            lengths,
            local,
            req,
            table,
            global_slots,
            width,
            width,
            k,
            m,
            table.shape[1],
            table.stride(0),
            entries,
            entries,
            256,
        )
        return global_slots

    expected = baseline().clone()
    local_expected = local.clone()

    def check(result):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(local, local_expected, rtol=0, atol=0)
        return {"local_and_global_exact": True}

    for name, fn in ordered([("two_kernel", baseline), ("one_kernel", fused)]):
        record(m, "topk_slots", name, fn, check)


with torch.inference_mode():
    weight, scales = per_block_cast_to_fp8(
        torch.randn(36864, 1280, device="cuda", dtype=torch.bfloat16) * 0.1,
        True,
        gran_k=32,
    )
    merged = SimpleNamespace(
        main_width=32768,
        index_width=4096,
        weight=weight,
        weight_scales=pack(scales.repeat_interleave(32, 0)),
    )
    flush = torch.empty(int(8e9 // 4), device="cuda", dtype=torch.int32)
    for m in args.batches:
        if any(c in args.cases for c in ("pure", "qb-kv")):
            run_qb(m, merged)
        if "topk-slots" in args.cases:
            run_slots(m)
    if args.bandwidth:
        source = torch.randn(int(1e9 // 4), device="cuda", dtype=torch.float32)
        destination = torch.empty_like(source)
        # copy_ uses the DMA memcpy engine and has no CUDA-kernel activity.
        # Add zero explicitly measures a coalesced read+write GPU kernel.
        timing, copied = benchmark(
            lambda: torch.add(source, 0, out=destination),
            "streaming_read_write_1GB",
            flush,
        )
        torch.testing.assert_close(copied, source, rtol=0, atol=0)
        bandwidth = {
            "payload_bytes": source.numel() * source.element_size(),
            "traffic_bytes": source.numel() * source.element_size() * 2,
            "read_plus_write_TB_s": 2e9 / (timing["median_us"] * 1e-6) / 1e12,
            "timing": {k: v for k, v in timing.items() if k != "samples"},
            "correctness": "exact",
        }
        (args.out / "bandwidth.json").write_text(json.dumps(bandwidth, indent=2))
        print("BANDWIDTH", json.dumps(bandwidth), flush=True)
