"""A, B and A+B on one continuous device timeline.

One observer container holds the GPU hardware counters (10 kHz) for the whole sequence:
    idle -> A -> idle -> B -> idle -> A+B -> idle
A and B run as ordinary gauge jobs with per-process CUDA traces (profile="trace"), so for the A+B window we get both
the combined device load and each process's own kernels. Everything is aligned on the host wall clock via Nsight's
session start epoch.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
import time
import uuid

from gauge import nsight, store
from gauge.probe.gpu import Gpu
from gauge.runner import Job, gpu_lock
from gauge.spec import JobSpec

OBS_IMAGE = "gauge/torch:1"
METRICS = ("SMs Active", "SM Issue", "Tensor Active", "Compute Warps in Flight [Throughput", "DRAM Read Bandwidth",
           "DRAM Write Bandwidth", "PCIe RX Throughput", "PCIe TX Throughput", "GR Active")


def _observer_start(name: str) -> None:
    nsight.OUT_DIR.mkdir(parents=True, exist_ok=True)
    stop = nsight.OUT_DIR / f"{name}.stop"
    stop.unlink(missing_ok=True)
    subprocess.run(["docker", "run", "-d", "--name", name, "--label", "gauge=1", "--gpus", "all", "--cap-add", "SYS_ADMIN",
                    "-v", f"{nsight.NSYS_DIR}:/nsys:ro", "-v", f"{nsight.OUT_DIR}:/nsys-out", "--entrypoint", "/bin/sh",
                    OBS_IMAGE, "-c",
                    f"/nsys/target-linux-x64/nsys profile -o /nsys-out/{name} --force-overwrite=true --trace=none "
                    f"--sample=none --cpuctxsw=none --gpu-metrics-devices=all --gpu-metrics-frequency={nsight.METRICS_HZ} "
                    f"sh -c 'while [ ! -f /nsys-out/{name}.stop ]; do sleep 0.2; done'"],
                   check=True, capture_output=True)


def _observer_stop(name: str) -> None:
    (nsight.OUT_DIR / f"{name}.stop").touch()
    subprocess.run(["docker", "wait", name], capture_output=True, timeout=600)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _run_jobs(conn, specs: list[JobSpec], gpu: Gpu, group: str | None) -> list[int]:
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    jobs = [Job(store.connect(path), s, gpu, corun_group=group) for s in specs]
    ids = [j.commit() for j in jobs]
    for j in jobs:
        j.launch(overlapping=[i for i in ids if i != j.run_id])
    threads = [threading.Thread(target=j.finish) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return ids


def run_sequence(conn, a: JobSpec, b: JobSpec, *, idle_s: float = 5.0, label: str = "") -> dict:
    a = a.model_copy(update={"profile": "trace"})
    b = b.model_copy(update={"profile": "trace"})
    seq = f"seq{uuid.uuid4().hex[:8]}"
    gpu = Gpu()
    marks: list[dict] = []
    with gpu_lock(f"gauge sequence {seq}", True):
        _observer_start(seq)
        time.sleep(3)  # let the observer's sampling start
        try:
            for step, specs, group in (("A", [a], None), ("B", [b], None), ("A+B", [a, b], f"{seq}-ab")):
                t0 = time.time()
                time.sleep(idle_s)
                marks.append({"step": "idle", "start": t0, "end": time.time()})
                ids = _run_jobs(conn, specs, gpu, group)
                marks.append({"step": step, "run_ids": ids})
            t0 = time.time()
            time.sleep(idle_s)
            marks.append({"step": "idle", "start": t0, "end": time.time()})
        finally:
            _observer_stop(seq)
    conn.execute("CREATE TABLE IF NOT EXISTS sequence (id TEXT PRIMARY KEY, label TEXT, json TEXT NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO sequence (id, label, json) VALUES (?,?,?)",
                 (seq, label, json.dumps({"marks": marks, "a": a.model_dump(), "b": b.model_dump()})))
    nsight.export_name(seq)
    return {"id": seq, "marks": marks}


def _metrics_db(seq: str) -> sqlite3.Connection:
    """The observer's export, indexed once: ~20M counter rows per sequence are unusable without it."""
    c = sqlite3.connect(nsight.OUT_DIR / f"{seq}.sqlite")
    c.execute("CREATE INDEX IF NOT EXISTS gm_metric_time ON GPU_METRICS(metricId, timestamp)")
    return c


def _kernel_window(conn, run_id: int) -> tuple[float, float] | None:
    """[first kernel, last kernel] of a run on the host clock (epoch seconds), from its own trace."""
    db = nsight.OUT_DIR / f"run{run_id}.sqlite"
    if not db.exists():
        nsight.export(run_id)
    if not db.exists():
        return None
    c = sqlite3.connect(db)
    epoch = c.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    lo, hi = c.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    return (epoch + lo) / 1e9, (epoch + hi) / 1e9


def analyze(conn, seq: str, trim: float = 0.1) -> dict:
    """Mean device counters per step. A step's window is its GPU-active span from the job traces (both jobs active,
    for A+B), trimmed by `trim` at each end; idle windows are the sleeps between steps."""
    rec = json.loads(conn.execute("SELECT json FROM sequence WHERE id=?", (seq,)).fetchone()[0])
    db = nsight.OUT_DIR / f"{seq}.sqlite"
    c = _metrics_db(seq)
    epoch = c.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    names = dict(c.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
    want = {mid: n for mid, n in names.items() if any(n.startswith(m) for m in METRICS)}
    windows = []
    for m in rec["marks"]:
        if m["step"] == "idle":
            s, e = m["start"], m["end"]
        else:
            ws = [_kernel_window(conn, rid) for rid in m["run_ids"]]
            if not all(ws):
                continue
            s, e = max(w[0] for w in ws), min(w[1] for w in ws)
        d = e - s
        windows.append({"step": m["step"], "start": s + trim * d, "end": e - trim * d, "run_ids": m.get("run_ids", [])})
    out = []
    for w in windows:
        lo, hi = int(w["start"] * 1e9 - epoch), int(w["end"] * 1e9 - epoch)
        row = {"step": w["step"], "run_ids": w["run_ids"], "seconds": w["end"] - w["start"],
               "start_abs": w["start"], "end_abs": w["end"]}
        for mid, n in want.items():
            v = c.execute("SELECT avg(value), count(*) FROM GPU_METRICS WHERE metricId=? AND timestamp BETWEEN ? AND ?",
                          (mid, lo, hi)).fetchone()
            row[n.split(" [")[0]] = v[0]
            row["samples"] = v[1]
        out.append(row)
    return {"id": seq, "windows": out}


def timeline(conn, seq: str, bin_ms: float = 50.0) -> list[dict]:
    """Device counters in bins over the whole sequence, seconds from the observer's start."""
    c = _metrics_db(seq)
    names = dict(c.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
    keep = {mid: n.split(" [")[0] for mid, n in names.items()
            if n.startswith(("SMs Active", "Tensor Active", "DRAM Read Bandwidth"))}
    b = int(bin_ms * 1e6)
    acc: dict = {}
    for mid, short in keep.items():
        for t, v in c.execute("SELECT timestamp/?, avg(value) FROM GPU_METRICS WHERE metricId=? AND timestamp>=0 GROUP BY 1",
                              (b, mid)):
            acc.setdefault(t, {"t_s": t * bin_ms / 1000})[short] = v
    return [acc[k] for k in sorted(acc)]


def _intervals(run_id: int, lo: float, hi: float) -> list[tuple[float, float]]:
    """This run's kernel intervals on the host clock, clipped to [lo, hi] (epoch seconds), merged."""
    c = sqlite3.connect(nsight.OUT_DIR / f"run{run_id}.sqlite")
    c.execute("CREATE INDEX IF NOT EXISTS k_start ON CUPTI_ACTIVITY_KIND_KERNEL(start)")
    epoch = c.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    a, b = int(lo * 1e9 - epoch), int(hi * 1e9 - epoch)
    out: list[list[float]] = []
    for s, e in c.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end > ? AND start < ? ORDER BY start", (a, b)):
        s, e = max(s, a), min(e, b)
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [((s + epoch) / 1e9, (e + epoch) / 1e9) for s, e in out]


def _overlap(x: list, y: list) -> float:
    i = j = 0
    tot = 0.0
    while i < len(x) and j < len(y):
        lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if hi > lo:
            tot += hi - lo
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return tot


def relationship(conn, seq: str) -> dict:
    """Observed A+B device counters vs an additive model and a time-slicing model; plus how often the two processes'
    kernels actually ran at the same moment."""
    res = analyze(conn, seq)
    w = {x["step"]: x for x in res["windows"] if x["step"] != "idle"}
    idle = [x for x in res["windows"] if x["step"] == "idle"]
    ab = w["A+B"]
    rec = json.loads(conn.execute("SELECT json FROM sequence WHERE id=?", (seq,)).fetchone()[0])
    m = next(x for x in rec["marks"] if x["step"] == "A+B")
    ws = [_kernel_window(conn, rid) for rid in m["run_ids"]]
    s, e = max(x[0] for x in ws), min(x[1] for x in ws)
    d = e - s
    lo, hi = s + 0.1 * d, e - 0.1 * d
    ia, ib = (_intervals(rid, lo, hi) for rid in m["run_ids"])
    span = hi - lo
    busy_a, busy_b = sum(e - s for s, e in ia) / span, sum(e - s for s, e in ib) / span
    both = _overlap(ia, ib) / span
    # solo busy fractions of A and B in their own windows, to turn solo counters into "per busy second" rates
    def solo_busy(step):
        x = w[step]
        iv = _intervals(x["run_ids"][0], x_start(x), x_end(x))
        return sum(e - s for s, e in iv) / (x_end(x) - x_start(x))
    rows = []
    keys = [k for k in ab if k not in ("step", "run_ids", "seconds", "samples", "start_abs", "end_abs")]
    sa, sb = solo_busy("A"), solo_busy("B")
    # Kernel spans include time a kernel was switched out, so on a time-sliced GPU the two shares can sum past 1.
    # Normalize them to shares of the window when they do.
    tot = busy_a + busy_b
    fa, fb = (busy_a / tot, busy_b / tot) if tot > 1 else (busy_a, busy_b)
    for k in keys:
        A, B, O = w["A"].get(k) or 0, w["B"].get(k) or 0, ab.get(k) or 0
        floor = sum((x.get(k) or 0) for x in idle) / len(idle)
        rows.append({"metric": k, "A": A, "B": B, "A+B observed": O, "additive": min(100.0, A + B - floor),
                     "time_sliced": floor + (A - floor) * min(1.0, fa / sa) + (B - floor) * min(1.0, fb / sb)})
    return {"id": seq, "busy_A_in_AB": busy_a, "busy_B_in_AB": busy_b, "share_A": fa, "share_B": fb,
            "kernel_spans_overlap": both,  # includes preemption: not proof of simultaneous execution
            "solo_busy_A": sa, "solo_busy_B": sb, "metrics": rows}


def x_start(x):
    return x["start_abs"]


def x_end(x):
    return x["end_abs"]
