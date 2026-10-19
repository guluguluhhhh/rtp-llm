"""Opt-in single-GPU FULL attention experiment; not a model/TPS benchmark.

See docs/benchmark/dsv41_full_layer_optimize.md for environment and protocol.
CUDA imports happen only after GPU admission and the cooperative GPU lock.
"""

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import statistics
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
PREFIX = "rtp_llm.models_py.modules.dsv41.fp8"
# multistream, DeepGEMM launch policy, Q-norm producer trigger
VARIANTS = {
    "serial": (False, False, False),
    "multistream": (True, False, False),
    "dg_pdl": (True, True, False),
    "qb_pdl": (True, True, True),
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def gpu_snapshot(gpu):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = [line.split(",") for line in output.strip().splitlines()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one physical GPU, got {output!r}")
    index, uuid, memory, utilization = (field.strip() for field in rows[0])
    return dict(
        index=int(index),
        uuid=uuid,
        memory_mib=int(memory),
        utilization=int(utilization),
    )


def require_idle(snapshot, args):
    if (
        snapshot["index"] != args.gpu
        or snapshot["memory_mib"] > args.idle_memory_mib
        or snapshot["utilization"] != 0
    ):
        raise RuntimeError(f"GPU admission failed: {snapshot}")


def prepare_source(args):
    native = args.native_lib_dir
    if native is None:
        candidates = (REPO / "rtp_llm/libs", REPO / "bazel-bin/rtp_llm/libs")
        native = next(
            (p for p in candidates if (p / "libth_transformer_config.so").is_file()),
            None,
        )
    if native is None:
        raise RuntimeError("Supply --native-lib-dir from a compatible RTP build")
    native = native.resolve()
    for name in ("libth_transformer_config.so", "librtp_compute_ops.so"):
        if not (native / name).is_file():
            raise FileNotFoundError(native / name)
    # Keep the checkout untouched; this overlay selects current Python sources
    # and the explicitly recorded native build, without recursive .so searches.
    package = args.output / "runtime/rtp_llm"
    package.mkdir(parents=True)
    for path in (REPO / "rtp_llm").iterdir():
        if path.name not in ("libs", "__pycache__"):
            (package / path.name).symlink_to(path, target_is_directory=path.is_dir())
    (package / "libs").symlink_to(native, target_is_directory=True)
    sys.path.insert(0, str(package.parent))
    return native


@contextlib.contextmanager
def policy(name):
    import deep_gemm

    previous = deep_gemm.get_pdl()
    _, library_pdl, trigger = VARIANTS[name]
    # This process is a standalone experiment, not a running inference server.
    # Production full_overlap only reads the engine-global library setting.
    deep_gemm.set_pdl(library_pdl)
    try:
        with patch.dict(os.environ, {"DSV41_FULL_DECODE_QB_PDL": str(int(trigger))}):
            yield
    finally:
        deep_gemm.set_pdl(previous)


def checked(result):
    if int(result[0]) != 0:
        raise RuntimeError(result)
    return result[1:]


def graph_edges(handle):
    class EdgeData(ctypes.Structure):
        _fields_ = [
            ("from_port", ctypes.c_ubyte),
            ("to_port", ctypes.c_ubyte),
            ("type", ctypes.c_ubyte),
            ("reserved", ctypes.c_ubyte * 5),
        ]

    # Own the v2 output buffers. The cuda-bindings edge-data views in the
    # experiment runtime returned corrupted first entries for the same graph.
    library = ctypes.CDLL("libcuda.so.1")
    operation = library.cuGraphGetEdges_v2
    operation.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(EdgeData),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    operation.restype = ctypes.c_int
    count = ctypes.c_size_t()
    status = operation(handle, None, None, None, ctypes.byref(count))
    if status:
        raise RuntimeError(f"cuGraphGetEdges_v2 count failed: {status}")
    sources = (ctypes.c_void_p * count.value)()
    targets = (ctypes.c_void_p * count.value)()
    data = (EdgeData * count.value)()
    status = operation(handle, sources, targets, data, ctypes.byref(count))
    if status:
        raise RuntimeError(f"cuGraphGetEdges_v2 failed: {status}")
    return [
        dict(
            source=a,
            target=b,
            type=e.type,
            from_port=e.from_port,
            to_port=e.to_port,
        )
        for a, b, e in zip(sources, targets, data)
    ]


def audit_graph(graph, destination, library_pdl):
    from cuda.bindings import driver as cuda

    handle = cuda.CUgraph(graph.raw_cuda_graph())
    _, count = checked(cuda.cuGraphGetNodes(handle))
    nodes, _ = checked(cuda.cuGraphGetNodes(handle, count))
    descriptions = {}
    for node in nodes:
        (kind,) = checked(cuda.cuGraphNodeGetType(node))
        item = {"type": str(kind)}
        if kind == cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            (params,) = checked(cuda.cuGraphKernelNodeGetParams(node))
            (name,) = checked(cuda.cuFuncGetName(params.func))
            item.update(
                name=name.decode() if isinstance(name, bytes) else name,
                grid=[params.gridDimX, params.gridDimY, params.gridDimZ],
                block=[params.blockDimX, params.blockDimY, params.blockDimZ],
                smem=params.sharedMemBytes,
            )
        descriptions[int(node)] = item
    edges = graph_edges(int(handle))
    qb_edges = [
        e
        for e in edges
        if "_query_norm_quant_kernel" in descriptions[e["source"]].get("name", "")
        and "gemm" in descriptions[e["target"]].get("name", "")
    ]
    save(destination, dict(nodes=descriptions, edges=edges, qb_edges=qb_edges))
    if len(qb_edges) != 1:
        raise RuntimeError(f"Expected one Q norm -> QB edge, got {qb_edges}")
    edge = qb_edges[0]
    expected = int(cuda.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC)
    if library_pdl:
        if (
            edge["type"] != expected
            or edge["from_port"] != cuda.CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC
        ):
            raise RuntimeError(f"Missing programmatic QB edge: {edge}")
    elif edge["type"] != 0 or edge["from_port"] != 0:
        raise RuntimeError(f"Expected ordinary completion QB edge: {edge}")


def capture(fn, name, destination):
    import torch

    from rtp_llm.models_py.modules.dsv41.fp8.decode import full_overlap, global_overlap

    reserved = {
        stream.cuda_stream
        for mapping in (
            full_overlap._INDEXER_STREAMS,
            global_overlap._GLOBAL_COMPUTE_STREAMS,
        )
        for stream in mapping.values()
    }
    # PyTorch returns pooled handles round-robin, including still-live streams.
    # A caller alias would silently serialize one of the measured branches.
    for _ in range(len(reserved) + 1):
        stream = torch.cuda.Stream()
        if stream.cuda_stream not in reserved:
            break
    else:
        raise RuntimeError("Cannot allocate a distinct capture stream")
    stream.wait_stream(torch.cuda.current_stream())
    with policy(name), torch.cuda.stream(stream):
        for _ in range(5):
            fn(name)
    torch.cuda.current_stream().wait_stream(stream)
    stream_ids = dict(
        caller=stream.cuda_stream,
        global_streams=[
            s.cuda_stream for s in global_overlap._GLOBAL_COMPUTE_STREAMS.values()
        ],
        indexer_streams=[s.cuda_stream for s in full_overlap._INDEXER_STREAMS.values()],
    )
    save(destination.with_suffix(".streams.json"), stream_ids)
    if VARIANTS[name][0]:
        handles = [
            stream_ids["caller"],
            *stream_ids["global_streams"],
            *stream_ids["indexer_streams"],
        ]
        if len(set(handles)) != len(handles):
            raise RuntimeError(f"FULL capture streams alias: {stream_ids}")
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    begin, end = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
    with policy(name), torch.cuda.graph(graph, stream=stream):
        begin.record()
        outputs = fn(name)
        end.record()
    audit_graph(graph, destination, VARIANTS[name][1])
    graph.instantiate()
    return graph, begin, end, outputs


def compare(actual, expected):
    import torch

    if set(actual) != set(expected):
        raise AssertionError("Mismatching output keys")
    for key, value in actual.items():
        ref = expected[key]
        if value.dtype == torch.float8_e4m3fn:
            value, ref = value.view(torch.uint8), ref.view(torch.uint8)
        elif not torch.isfinite(value).all():
            raise AssertionError(f"Non-finite {key}")
        torch.testing.assert_close(value, ref, rtol=0, atol=0, msg=key)


def measure(graphs, flush, rng, count):
    import torch

    for _ in range(30):
        for graph, _, _, _ in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in graphs}
    for _ in range(count):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            flush.add_(1)
            graph, begin, end, _ = graphs[name]
            graph.replay()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000)
    return samples


def profile(graphs, destination):
    import torch

    reports = {}
    for name, (graph, _, _, _) in graphs.items():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
        path = destination.parent / f"{destination.name}_{name}.json"
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
        reports[name] = dict(
            count_per_replay=len(kernels) / 3,
            kernel_sum_us=sum(e["dur"] for e in kernels) / 3,
            names=dict(Counter(e["name"] for e in kernels)),
        )
    names = [r["names"] for r in reports.values()]
    if not names or not names[0] or any(n != names[0] for n in names[1:]):
        raise AssertionError(f"Unexpected kernel names/counts: {reports}")
    return reports


def statistics_us(samples):
    ordered = sorted(samples)
    return dict(
        median=statistics.median(ordered),
        p90=ordered[(9 * len(ordered) + 9) // 10 - 1],
        min=ordered[0],
        max=ordered[-1],
    )


def run_benchmark(args):
    import torch

    from rtp_llm.models_py.modules.dsv41._v41_query_quant import query_norm_quant
    from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_global_overlap import (
        full_attention_case,
        run_full_attention,
        update_full_positions,
    )

    rng = random.Random(args.seed)
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    results = []
    cases = (
        [("chain", rows, 1, 2, 32768) for rows in args.rows]
        if args.scope == "chain"
        else [
            ("full", args.batch, args.span, ratio, context)
            for context in args.contexts
            for ratio in args.ratios
        ]
    )
    for scope, batch, span, ratio, context in cases:
        case = full_attention_case(
            ratio, batch if scope == "full" else 1, span, context
        )
        label = f"{scope}{ratio}_b{batch}_s{span}_ctx{context}"
        initial = {k: v.clone() for k, v in case.owner._test_pools.items()}
        x = (
            torch.randn(batch, 1792, device="cuda", dtype=torch.bfloat16)[:, :1280]
            if scope == "chain"
            else None
        )
        weight = (
            torch.randn(1280, device="cuda", dtype=torch.bfloat16)
            if scope == "chain"
            else None
        )

        def forward(name):
            if scope == "full":
                result = run_full_attention(case, VARIANTS[name][0])
                if VARIANTS[name][0] and not getattr(
                    case.owner, "_full_decode_overlap_warmed", None
                ):
                    raise RuntimeError(
                        "FULL multistream fell back; do not benchmark this case"
                    )
                return result
            y, (q, sf) = query_norm_quant(x, weight, 1e-6, pdl=VARIANTS[name][2])
            main, index = case.owner._mega_attention.merged_qb.forward_quantized(q, sf)
            return dict(norm=y, quant=q, scales=sf, main=main, index=index)

        for repeat in range(args.captures):
            prefix = f"{label}_capture{repeat}"
            print("START", prefix, flush=True)
            order = list(args.variants)
            rng.shuffle(order)
            graphs = {
                name: capture(
                    forward, name, args.output / f"{prefix}_{name}_edges.json"
                )
                for name in order
            }
            replays = 10 if scope == "chain" else 3
            for iteration in range(replays):
                if scope == "chain":
                    x.copy_(
                        torch.randn_like(x) * (0 if iteration == 0 else iteration + 1)
                    )
                    weight.copy_(torch.randn_like(weight))
                else:
                    case.x.copy_(torch.randn_like(case.x))
                    case.starts.sub_(1)
                    update_full_positions(case)
                    for table in case.owner._block_tables_by_type.values():
                        table.copy_(table.roll(1, 1))
                # Every candidate sees the same pool/state snapshot. The
                # serial, library-PDL-off path is the numerical reference.
                for key, value in initial.items():
                    case.owner._test_pools[key].copy_(value)
                with policy("serial"):
                    expected = forward("serial")
                expected = {k: v.clone() for k, v in expected.items()}
                pools = {k: v.clone() for k, v in case.owner._test_pools.items()}
                for graph, _, _, actual in graphs.values():
                    for key, value in initial.items():
                        case.owner._test_pools[key].copy_(value)
                    graph.replay()
                    compare(actual, expected)
                    compare(case.owner._test_pools, pools)
                del expected, pools
            samples = measure(graphs, flush, rng, args.samples)
            row = dict(
                label=label,
                capture=repeat,
                samples_us=samples,
                statistics_us={name: statistics_us(s) for name, s in samples.items()},
                bitwise_changed_input_graph_replays=replays,
                wo_a_quant_ready=case.wo_a_quant_ready,
                cache_flush_bytes=flush.numel(),
            )
            # Save timing before profiling so a profiler failure cannot erase it.
            results.append(row)
            save(args.output / "results.json", results)
            if args.profile:
                row["profile"] = profile(graphs, args.output / prefix)
                save(args.output / "results.json", results)
            print(prefix, row["statistics_us"], flush=True)
            del graphs, graph, actual
        del case, initial, x, weight
        torch.cuda.empty_cache()


def run(args, native):
    import deep_gemm
    import torch
    import triton

    from rtp_llm import ops
    from rtp_llm.models_py.modules.dsv41.fp8.decode import full_overlap

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("This experiment requires exactly one visible SM10x GPU")
    ops.ensure_compute_ops_loaded()
    if not Path(full_overlap.__file__).resolve().is_relative_to(REPO):
        raise RuntimeError("Imported FULL code does not belong to this checkout")
    versions = {}
    for package in ("deep-gemm", "flash-mla", "rtp-kernel", "cuda-bindings"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "distribution metadata unavailable"
    hashes = {}
    for name in ("libth_transformer_config.so", "librtp_compute_ops.so"):
        digest = hashlib.sha256()
        with (native / name).open("rb") as source:
            for block in iter(lambda: source.read(8 << 20), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    save(
        args.output / "versions.json",
        dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            triton=triton.__version__,
            libraries=versions,
            native_library_sha256=hashes,
            native_lib_dir=str(native),
            deep_gemm=deep_gemm.__file__,
            initial_deepgemm_pdl=deep_gemm.get_pdl(),
            capability=torch.cuda.get_device_capability(),
            source=full_overlap.__file__,
            variant_policy=VARIANTS,
        ),
    )
    with torch.inference_mode():
        if not args.skip_unit_tests:
            modules = (
                "test_v41_global_overlap",
                "test_v41_decode_device_metadata",
                "test_attention_v41_sparse_selection",
                "test_v41_decode_indexer",
            )
            suite = unittest.TestSuite(
                unittest.defaultTestLoader.loadTestsFromModule(
                    importlib.import_module(PREFIX + ".test." + name)
                )
                for name in modules
            )
            with policy("qb_pdl"):
                result = unittest.TextTestRunner(verbosity=2).run(suite)
            save(
                args.output / "tests.json",
                dict(
                    tests=result.testsRun,
                    failures=len(result.failures),
                    errors=len(result.errors),
                    skipped=result.skipped,
                ),
            )
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError("Unit-test failures/skips: do not accept this run")
        if args.scope != "tests":
            run_benchmark(args)
    save(args.output / "completed.json", {"passed": True})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument(
        "--gpu", type=int, required=True, help="Physical nvidia-smi index"
    )
    parser.add_argument("--native-lib-dir", type=Path)
    parser.add_argument(
        "--idle-memory-mib",
        type=int,
        default=32,
        help="Raise only after confirming ownership of resident allocations",
    )
    parser.add_argument(
        "--lock-dir", type=Path, default=Path("/tmp/rtp_llm/smoke/test/gpu_status")
    )
    parser.add_argument("--scope", choices=("tests", "chain", "full"), default="full")
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 131072])
    parser.add_argument("--ratios", nargs="+", type=int, choices=(1, 2), default=[2, 1])
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[1, 7, 64, 97, 384, 769, 1024]
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--span", type=int, default=6)
    parser.add_argument("--captures", type=int, default=3)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile", action="store_true", help="Separate, untimed profiler replays"
    )
    parser.add_argument("--skip-unit-tests", action="store_true")
    args = parser.parse_args()
    values = (
        [args.batch, args.span, args.captures, args.samples] + args.rows + args.contexts
    )
    if any(value <= 0 for value in values) or args.gpu < 0 or args.idle_memory_mib < 0:
        parser.error(
            "Shapes/counts must be positive; GPU index/memory limit nonnegative"
        )
    if any(
        context % 128 or context <= args.span + args.captures * 3 + 2
        for context in args.contexts
    ):
        parser.error(
            "Contexts must be multiples of 128 and leave room for dynamic replay positions"
        )
    if len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate variants")
    if args.scope == "tests" and args.skip_unit_tests:
        parser.error("--scope tests cannot skip unit tests")
    return args


def main():
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "arguments.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    save(
        args.output / "source.json",
        dict(
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            status=subprocess.check_output(
                ["git", "status", "--short"], cwd=REPO, text=True
            ),
        ),
    )
    (args.output / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO)
    )
    before = gpu_snapshot(args.gpu)
    save(args.output / "gpu_before.json", before)
    require_idle(before, args)
    args.lock_dir.mkdir(parents=True, exist_ok=True)
    with (args.lock_dir / str(args.gpu)).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = gpu_snapshot(args.gpu)
        save(args.output / "gpu_locked.json", locked)
        require_idle(locked, args)
        os.environ["CUDA_VISIBLE_DEVICES"] = locked["uuid"]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["DSV4_FP8_QUANT_KERNEL"] = "auto"
        os.environ["DSV41_FUSED_QUERY_QUANT"] = "1"
        os.environ["DSV41_FUSE_INDEXER_METADATA"] = "1"
        os.environ["DSV41_FUSED_DECODE_GLOBAL"] = "1"
        os.environ["DSV41_FUSED_DECODE_INDEXER"] = "1"
        os.environ["DSV41_FUSED_QKV_ROPE_CACHE"] = "1"
        os.environ["DSV41_OVERLAP_GLOBAL_QKV"] = "0"
        os.environ["MOEDBG"] = "0"
        save(
            args.output / "settings.json",
            {
                k: v
                for k, v in os.environ.items()
                if k.startswith("DSV") or k == "CUDA_HOME"
            },
        )
        try:
            native = prepare_source(args)
            run(args, native)
        except BaseException as error:
            save(args.output / "failed.json", {"error": repr(error)})
            raise
        finally:
            # The process may still own its CUDA context here. OS process exit
            # releases it; do not interpret this snapshot as post-exit usage.
            save(args.output / "gpu_before_process_exit.json", gpu_snapshot(args.gpu))


if __name__ == "__main__":
    main()
