"""Structured per-node execution traces, for validating simulation cost models.

For one run, writes to data/traces/:
  run<id>.node.json.gz     the node: params, hardware, timing, isolation context, and every GPU kernel/copy
                           (name, category, start/duration in µs from first GPU activity, grid, block, stream)
  run<id>.perfetto.json.gz the same kernels/copies in Chrome trace format (open in ui.perfetto.dev)
Kernels are process scope (this node's process tree only), so they stay valid under concurrency.
"""
from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

from gauge import nsight
from gauge.interference import rows as interference_rows
from gauge import paths

OUT = paths.DATA / "traces"
SCHEMA_VERSION = 1


def export_run(conn, run_id: int) -> dict:
    db = nsight.OUT_DIR / f"run{run_id}.sqlite"
    if not db.exists():
        raise FileNotFoundError(f"no Nsight export for run {run_id}")
    r = conn.execute("SELECT r.*, s.json AS sj FROM run r JOIN summary s ON s.run_id=r.id WHERE r.id=?", (run_id,)).fetchone()
    spec, summary = json.loads(r["spec_json"]), json.loads(r["sj"])
    host = json.loads(r["host_json"] or "{}")
    init = json.loads(r["initial_json"] or "{}")
    t = sqlite3.connect(db)
    ks = t.execute("SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.streamId,"
                   " k.registersPerThread, k.staticSharedMemory + k.dynamicSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k"
                   " JOIN StringIds s ON s.id = k.shortName ORDER BY k.start").fetchall()
    has_copy = t.execute("SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_MEMCPY'").fetchone()
    cs = t.execute("SELECT start, end, bytes, copyKind, streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall() \
        if has_copy else []
    t0 = min([k[0] for k in ks[:1]] + [c[0] for c in cs[:1]]) if (ks or cs) else 0
    kinds = {1: "h2d", 2: "d2h", 8: "d2d", 10: "p2p"}
    kernels = [{"name": k[2], "category": nsight.category(k[2]), "start_us": (k[0] - t0) / 1e3, "dur_us": (k[1] - k[0]) / 1e3,
                "grid": [k[3], k[4], k[5]], "block": [k[6], k[7], k[8]], "stream": k[9], "regs_per_thread": k[10],
                "shared_mem_bytes": k[11]} for k in ks]
    copies = [{"kind": kinds.get(c[3], str(c[3])), "start_us": (c[0] - t0) / 1e3, "dur_us": (c[1] - c[0]) / 1e3,
               "bytes": c[2], "stream": c[4]} for c in cs]
    corun = next((x for x in interference_rows(conn) if x["run_id"] == run_id), None)
    node = {
        "schema_version": SCHEMA_VERSION,
        "node": {"run_id": run_id, "workload": r["workload"], "params": spec["params"], "image_digest": r["image_digest"]},
        "hardware": {k: host.get(k) for k in ("gpu_name", "gpu_uuid", "driver", "vram_total", "cpus", "ram_total", "kernel")},
        "isolation": {"concurrent_with_runs": init.get("overlapping_run_ids", []), "corun_group": r["corun_group"],
                      "other_gpu_processes_at_start": len(init.get("gpu_processes", [])),
                      "kernel_scope": "this node's process tree only (CUPTI)",
                      "interference_vs_solo": corun},
        "timing": {"container_s": summary.get("runtime_s"), "phase_marks_s": summary.get("marks"),
                   "gpu_window_s": (max(k[1] for k in ks) - ks[0][0]) / 1e9 if ks else None},
        "resources": {k: summary.get(k) for k in ("cpu_time_s", "ram_peak_bytes", "vram_peak_bytes", "dev_power_mean_w",
                                                   "dev_energy_j")},
        "outcome": json.loads(r["outcome_json"] or "{}"),
        "kernels": kernels,
        "copies": copies,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / f"run{run_id}.node.json.gz", "wt") as f:
        json.dump(node, f)
    events = [{"name": k["name"], "cat": k["category"], "ph": "X", "ts": k["start_us"], "dur": k["dur_us"], "pid": run_id,
               "tid": f"stream {k['stream']}", "args": {"grid": k["grid"], "block": k["block"]}} for k in kernels]
    events += [{"name": f"memcpy {c['kind']}", "cat": "copy", "ph": "X", "ts": c["start_us"], "dur": c["dur_us"],
                "pid": run_id, "tid": f"stream {c['stream']}", "args": {"bytes": c["bytes"]}} for c in copies]
    events.append({"name": "process_name", "ph": "M", "pid": run_id, "args": {"name": f"run {run_id} {r['workload']}"}})
    with gzip.open(OUT / f"run{run_id}.perfetto.json.gz", "wt") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)
    return {"run_id": run_id, "kernels": len(kernels), "copies": len(copies)}
