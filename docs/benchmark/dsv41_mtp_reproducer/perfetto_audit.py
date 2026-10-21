"""Use official Perfetto Trace Processor to audit GPU slices and samples."""

import argparse
import hashlib
import json
from pathlib import Path

from perfetto.trace_processor import TraceProcessor

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
output = {}
for path in sorted(a.directory.glob("*_trace.json")):
    raw = json.loads(path.read_text())
    raw_k = [
        x for x in raw["traceEvents"] if x.get("cat") == "kernel" and x.get("ph") == "X"
    ]
    with TraceProcessor(trace=str(path)) as tp:
        kernel_sql = "SELECT ts,dur,name,track_id,arg_set_id FROM slice WHERE category='kernel' ORDER BY ts"
        rows = [x.__dict__ for x in tp.query(kernel_sql)]
        assert len(rows) == len(raw_k), (path, len(rows), len(raw_k))
        # JSON us to Perfetto integer ns rounding; converter may shift timestamps
        # to a common origin but durations and relative ordering must agree.
        for r, e in zip(rows, sorted(raw_k, key=lambda x: x["ts"])):
            assert abs(r["dur"] / 1000 - e["dur"]) < 0.002, (path, r, e)
            assert r["name"] == e["name"]
        group_sql = "SELECT name,COUNT(*) AS calls,SUM(dur)/1000.0 AS total_kernel_body_us FROM slice WHERE category='kernel' GROUP BY name ORDER BY total_kernel_body_us DESC"
        grouped = [x.__dict__ for x in tp.query(group_sql)]
        stats = [
            x.__dict__
            for x in tp.query(
                "SELECT name,value FROM stats WHERE severity='error' AND value>0"
            )
        ]
        assert all(
            x["name"] == "slice_spill_overlapping_complete_event" for x in stats
        ), (path, stats)
        # Kineto CPU/GPU complete events can overlap without nesting. Perfetto
        # recovers these slices; independently prove GPU tracks are unchanged.
        if raw_k:
            shift = rows[0]["ts"] / 1000 - min(e["ts"] for e in raw_k)
            for r, e in zip(rows, sorted(raw_k, key=lambda e: e["ts"])):
                assert abs(r["ts"] / 1000 - e["ts"] - shift) < 0.004, (path, r, e)
    # Exact envelope independently recomputed from Perfetto's parsed slices.
    metrics_path = path.with_name(path.name.replace("_trace.json", "_metrics.json"))
    rec = json.loads(metrics_path.read_text())
    n = rec["kernel_count"]
    samples = []
    assert len(rows) == rec["sample_count"] * (n + 4)
    for i, s in enumerate(rec["samples"]):
        ks = rows[i * (n + 4) + 4 : (i + 1) * (n + 4)]
        envelope = (
            max(k["ts"] + k["dur"] for k in ks) - min(k["ts"] for k in ks)
        ) / 1000
        assert abs(envelope - s["envelope_us"]) < 0.004, (
            path,
            i,
            envelope,
            s["envelope_us"],
        )
        samples.append(envelope)
    output[path.name] = dict(
        kernel_sql=kernel_sql,
        group_sql=group_sql,
        kernel_count_total=len(rows),
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        samples_us=samples,
        grouped_kernel_bodies=grouped,
        parse_warnings=stats,
    )
(a.directory / "perfetto_audit.json").write_text(json.dumps(output, indent=2))
print("PERFETTO_AUDIT_PASS", len(output), "traces")
