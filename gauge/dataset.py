"""Export the ledger as a flat, documented dataset (data/dataset/): CSV tables + DATA_CARD.md.

Every column is one of three scopes:
  container  cgroup v2 counters for the job's own container (exact, isolated)
  process    Nsight CUDA trace of the job's own process tree: kernels and copies (exact, isolated)
  device     whole-GPU NVML / Nsight hardware counters (shared; attributable only when the run had the GPU alone)
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from gauge import nsight
from gauge import paths

OUT = paths.DATA / "dataset"
GIB = 2**30


def _write(name: str, rows: list[dict], gz=False) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    opener = (lambda p: gzip.open(p, "wt", newline="")) if gz else (lambda p: open(p, "w", newline=""))
    with opener(OUT / (name + (".csv.gz" if gz else ".csv"))) as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _experiment(spec: dict, corun: str | None) -> str:
    wl, p, prof = spec["workload"], spec["params"], spec.get("profile", "none")
    if wl == "micro.idle":
        return "X0 idle floor"
    if wl == "micro.ceilings":
        return "X1 ceilings"
    if wl == "llamacpp.batched":
        return "X4 batched"
    if corun:
        return "X4/X5 co-run"
    if wl == "llamacpp.phase" and p.get("tokens") in (256,) and prof == "trace":
        return "X4 solo reference"
    if wl == "llamacpp.phase" and prof == "metrics":
        return "X2 llm anatomy" if p.get("ngl", 99) == 99 else "X2 partial offload"
    if prof == "metrics":
        return "X3 task anatomy"
    if prof == "trace":
        return "X5 solo reference"
    if wl == "llm.generate" and "evict_page_cache" in p:
        return "X6 cold start"
    return "earlier (profiling v1)"


def export(conn) -> dict:
    counts = {}
    profiles = {r["run_id"]: json.loads(r["json"]) for r in conn.execute("SELECT * FROM gpu_profile")} \
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='gpu_profile'").fetchone() else {}
    runs, procs, devs, kern, cats, outcomes = [], [], [], [], [], []
    for r in conn.execute("SELECT r.*, s.json AS sj FROM run r LEFT JOIN summary s ON s.run_id=r.id"
                          " WHERE r.status='done' ORDER BY r.id"):
        spec, s = json.loads(r["spec_json"]), json.loads(r["sj"] or "{}")
        init = json.loads(r["initial_json"] or "{}")
        own = set(s.get("gpu_pids") or [])
        tenants = [g for g in init.get("gpu_processes", []) if g["pid"] not in own]
        exp = _experiment(spec, r["corun_group"])
        runs.append({
            "run_id": r["id"], "experiment": exp, "workload": r["workload"], "family": r["family"],
            "params": json.dumps(spec["params"], sort_keys=True), "profile": spec.get("profile", "none"),
            "corun_group": r["corun_group"] or "", "outcome": r["failure_class"],
            "image_digest": r["image_digest"],
            "other_gpu_processes_at_start": len(tenants),
            "other_gpu_vram_at_start_mib": round(sum((g.get("vram") or 0) for g in tenants) / 2**20),
            "device_metrics_attributable": int(not r["corun_group"] and len(tenants) <= 1),
            # container scope
            "runtime_s": s.get("runtime_s"), "cpu_time_s": s.get("cpu_time_s"),
            "cpu_mean_cores": s.get("cpu_mean_cores"), "ram_peak_gib": (s.get("ram_peak_bytes") or 0) / GIB,
            "ram_anon_peak_gib": (s.get("ram_anon_peak_sampled_bytes") or 0) / GIB,
            "io_read_mib": (s.get("io_read_bytes") or 0) / 2**20,
            # process scope (NVML per-pid)
            "vram_peak_gib": (s.get("vram_peak_bytes") or 0) / GIB,
            # device scope (NVML)
            "dev_util_mean_pct": s.get("dev_util_mean"), "dev_power_mean_w": s.get("dev_power_mean_w"),
            "dev_power_peak_w": s.get("dev_power_peak_w"), "dev_energy_j": s.get("dev_energy_j"),
            "dev_enc_util_mean_pct": s.get("dev_enc_util_mean"), "dev_dec_util_mean_pct": s.get("dev_dec_util_mean"),
            "dev_temp_peak_c": s.get("dev_temp_peak_c"), "dev_throttle_any": s.get("dev_throttle_any"),
        })
        out = json.loads(r["outcome_json"] or "{}")
        for o in out.get("responses", [out] if out else []):
            outcomes.append({"run_id": r["id"], "workload": r["workload"], **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                                                                            for k, v in o.items() if k not in ("message", "samples_ns", "samples_ts")}})
        prof = profiles.get(r["id"])
        if not prof:
            continue
        p = prof.get("process", {})
        if p:
            procs.append({"run_id": r["id"], "workload": r["workload"],
                          **{k: v for k, v in p.items() if not isinstance(v, (dict, list)) and not k.endswith("_ns")},
                          **{f"copy_{k}_{m}": v[m] for k, v in (prof.get("copies") or {}).items() for m in ("bytes", "gb_per_s")}})
            for c, v in p.get("by_category", {}).items():
                cats.append({"run_id": r["id"], "workload": r["workload"], "category": c, **v})
            for k in p.get("top_kernels", []):
                kern.append({"run_id": r["id"], "workload": r["workload"], **k})
        d = prof.get("device_metrics")
        if d:
            row = {"run_id": r["id"], "workload": r["workload"]}
            for name, v in d.items():
                short = name.split(" [")[0].lower().replace(" ", "_").replace("/", "_")
                row[f"{short}_gpu_window"] = v.get("mean_gpu_window")
                row[f"{short}_whole_run"] = v.get("mean_whole")
            devs.append(row)
    counts["runs"] = _write("runs", runs)
    counts["outcomes"] = _write("workload_outcomes", outcomes)
    counts["gpu_process"] = _write("gpu_process", procs)
    counts["gpu_kernel_categories"] = _write("gpu_kernel_categories", cats)
    counts["gpu_top_kernels"] = _write("gpu_top_kernels", kern)
    counts["gpu_device_counters"] = _write("gpu_device_counters", devs)
    samples = [dict(r) for r in conn.execute("SELECT * FROM sample ORDER BY run_id, t")]
    counts["samples"] = _write("samples_100ms", samples, gz=True)
    sb = [dict(r) for r in conn.execute("SELECT * FROM startbench WHERE error IS NULL")] \
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='startbench'").fetchone() else []
    counts["container_starts"] = _write("container_starts", sb)
    from gauge.interference import rows as interference_rows

    counts["corun_interference"] = _write("corun_interference",
                                          [{**r, "params": json.dumps(r["params"], sort_keys=True)} for r in interference_rows(conn)])
    # 10 ms timelines of kernel categories for every profiled run (process scope)
    tl = []
    for rid in profiles:
        db = nsight.OUT_DIR / f"run{rid}.sqlite"
        if db.exists():
            tl += [{"run_id": rid, **row} for row in nsight.timeline(db, 10.0)]
    counts["kernel_timeline_10ms"] = _write("kernel_timeline_10ms", tl, gz=True)
    card = paths.DOCS / "DATA_CARD.md"
    if card.exists():
        (OUT / "DATA_CARD.md").write_text(card.read_text())
    return counts
