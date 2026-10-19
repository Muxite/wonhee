"""Observer effect: how much does measuring change what is measured, in these experiments?

O1 Nsight in the container: steady-state rate vs wall time, and where the wall-time cost goes (startup, export).
O2 gauge's host sampler (NVML + cgroup reads every 100 ms): its CPU per sample and per run, missed samples.
O3 the measurement floor: what an observed-but-idle GPU draws.
O4 the harness: container launch vs native, per launch variant.
O5 attribution: how much co-run evidence rests on device-scope readings shared by every job.
O6 sampler interval contrast (C9, profiling/sweeps/x/c9_observer.yaml), when its runs exist.

    uv run python profiling/analysis/observer.py [--db ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "profiling" / "audit")]
from audit import condition  # noqa: E402
from gauge import paths  # noqa: E402

UNOBSERVED_IDLE_W = 16.8  # nvidia-smi in P8 with no CUDA context and no sampler (DATA_CARD.md, 2026-10-16)


def stat(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "median": st.median(xs), "min": min(xs), "max": max(xs)} if xs else None


def fmt(s, d=2, unit=""):
    return f"{s['median']:.{d}f}{unit} [{s['min']:.{d}f}–{s['max']:.{d}f}] n={s['n']}" if s else "–"


def done(conn):
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT r.*, s.json AS sj FROM run r JOIN summary s ON s.run_id=r.id WHERE r.status='done'"
                        " AND r.failure_class='success' ORDER BY r.id").fetchall()


def rate(r) -> tuple[float | None, str | None]:
    """The workload's own steady-state rate, where it reports one (independent of container wall time)."""
    o = json.loads(r["outcome_json"] or "{}")
    resp = o.get("responses") or ([o] if o else [])
    if r["workload"] == "llamacpp.phase":
        v = [x["avg_ts"] for x in resp if "avg_ts" in x]
        return (st.mean(v), "tok/s") if v else (None, None)
    if r["workload"] == "train.small" and o.get("imgs_per_s"):
        return o["imgs_per_s"], "images/s"
    if r["workload"] == "stress.gpu" and o.get("rate"):
        return o["rate"], f"{o.get('unit')}/s"
    if r["workload"] == "wonhee.gpu":
        sp = (json.loads(r["sj"]).get("phase_s") or {}).get("spans") or []
        return (1 / st.median(sp[1:]), "calls/s") if len(sp) > 1 else (None, None)
    return None, None


# --- O1 ----------------------------------------------------------------------------------------------------------

def launch_rates(conn) -> dict[int, float]:
    out = {}
    for run_id, js in conn.execute("SELECT run_id, json FROM gpu_profile"):
        lr = (json.loads(js).get("process") or {}).get("launches_per_s")
        if lr:
            out[run_id] = lr
    return out


def o1_nsight(rows, launches: dict | None = None) -> dict:
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["corun_group"]:
            continue
        wl, p, mode, sharing = condition(r)
        # same image only: the old X-series llama.cpp image is a different build from round 2's
        by[(wl, p, sharing, (r["image_digest"] or "").split("@")[0])][mode].append(r)
    out = []
    for (wl, p, _, _img), modes in by.items():
        if "none" not in modes or len(modes) < 2:
            continue
        off = modes["none"]
        for mode in ("trace", "metrics"):
            on = modes.get(mode)
            if not on:
                continue
            s = lambda rs, f: stat([f(x) for x in rs])  # noqa: E731
            wall = lambda x: json.loads(x["sj"]).get("runtime_s")  # noqa: E731
            tail = lambda x: (x["container_finished"] - x["workload_finished"]) if x["container_finished"] and x["workload_finished"] else None  # noqa: E731
            ro, rn = s(off, lambda x: rate(x)[0]), s(on, lambda x: rate(x)[0])
            wo, wn = s(off, wall), s(on, wall)
            out.append({"workload": wl, "params": json.loads(p), "mode": mode, "unit": rate(off[0])[1],
                        "rate_off": ro, "rate_on": rn,
                        "steady_change": (rn["median"] / ro["median"] - 1) if ro and rn else None,
                        "wall_off_s": wo, "wall_on_s": wn,
                        "wall_added_s": (wn["median"] - wo["median"]) if wo and wn else None,
                        "wall_ratio": (wn["median"] / wo["median"]) if wo and wn else None,
                        "after_exit_off_s": s(off, tail), "after_exit_on_s": s(on, tail),
                        # how many kernels per second the job launches (its own trace): the per-launch cost of tracing
                        "launches_per_s": st.median([launches[x["id"]] for x in on if x["id"] in (launches or {})])
                        if any(x["id"] in (launches or {}) for x in on) else None})
    pts = [(p["launches_per_s"], p["steady_change"]) for p in out if p["launches_per_s"] and p["steady_change"] is not None]
    rho = None
    if len(pts) >= 4:  # Spearman rank correlation, launches/s vs steady-state change
        rk = lambda v: {x: i for i, x in enumerate(sorted(v))}  # noqa: E731
        a, b = rk([x for x, _ in pts]), rk([y for _, y in pts])
        n = len(pts)
        rho = 1 - 6 * sum((a[x] - b[y]) ** 2 for x, y in pts) / (n * (n * n - 1))
    return {"pairs": out, "launch_rate_vs_steady_change_spearman": rho, "n_points": len(pts)}


# --- O2 ----------------------------------------------------------------------------------------------------------

def o2_sampler(rows) -> dict:
    per_sample, cores, share, missed, total = [], [], [], 0, 0
    for r in rows:
        s = json.loads(r["sj"])
        c, n, wall = s.get("collector_cpu_s"), s.get("sample_count"), s.get("container_wall_s")
        if not (c and n and wall):
            continue
        per_sample.append(c / n * 1000)
        cores.append(c / wall)
        if s.get("cpu_time_s"):
            share.append(c / s["cpu_time_s"])
        missed += s.get("missed_samples") or 0
        total += n
    return {"cpu_ms_per_sample": stat(per_sample), "collector_cores": stat(cores),
            "collector_vs_workload_cpu": stat(share), "missed_samples": missed, "samples": total,
            "note": "the sampler runs on the host, outside the job's cgroup: its CPU is never charged to the job"}


# --- O3 ----------------------------------------------------------------------------------------------------------

def o3_floor(rows, conn) -> dict:
    out = {}
    for r in rows:
        if r["workload"] != "micro.idle":
            continue
        mode = json.loads(r["spec_json"]).get("profile", "none")
        exp = (json.loads(r["spec_json"]).get("provenance") or {}).get("experiment") or "X0 idle floor"
        interval = json.loads(r["sj"]).get("sampling_interval_ms")
        w = [x[0] for x in conn.execute("SELECT dev_power_w FROM sample WHERE run_id=? AND dev_power_w IS NOT NULL", (r["id"],))]
        if w:
            out.setdefault(f"{exp} · profile {mode} · sampler {interval:.0f} ms", []).append(st.median(w))
    return {"conditions": {k: stat(v) for k, v in out.items()}, "unobserved_idle_w": UNOBSERVED_IDLE_W,
            "note": "unobserved = nvidia-smi in P8 with no CUDA context and no gauge sampler (DATA_CARD.md)"}


# --- O4 ----------------------------------------------------------------------------------------------------------

def o4_harness(conn) -> dict:
    v = collections.defaultdict(list)
    for s in conn.execute("SELECT * FROM startbench WHERE error IS NULL"):
        if s["t_req"] is not None and s["first"] is not None:
            v[s["variant"]].append((s["first"] - s["t_req"]) * 1000)
    base = st.median(v["native"]) if v.get("native") else None
    return {"variants": {k: {**stat(x), "added_vs_native_ms": (st.median(x) - base) if base else None} for k, x in v.items()},
            "native_ms": base}


# --- O5 ----------------------------------------------------------------------------------------------------------

def o5_attribution(rows) -> dict:
    co = [r for r in rows if r["corun_group"]]
    profiled = [r for r in co if json.loads(r["spec_json"]).get("profile", "none") != "none"]
    own_rate = [r for r in co if rate(r)[0] is not None]
    return {"corun_members": len(co), "with_process_scope_trace": len(profiled), "with_own_rate": len(own_rate),
            "device_scope_only": len(co) - len({r["id"] for r in profiled} | {r["id"] for r in own_rate}),
            "note": "device readings in a co-run (power, busy, clocks) belong to every job in it; per-job evidence comes "
                    "from the job's own counters (tokens/s, calls, imgs/s) or its process-scope Nsight trace"}


# --- O6 (C9) -------------------------------------------------------------------------------------------------------

def o6_interval(rows, conn) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        if (json.loads(r["spec_json"]).get("provenance") or {}).get("experiment") != "C9 observer":
            continue
        s = json.loads(r["sj"])
        iv = s.get("sampling_interval_ms")
        if r["workload"] == "micro.idle":
            w = [x[0] for x in conn.execute("SELECT dev_power_w FROM sample WHERE run_id=? AND dev_power_w IS NOT NULL", (r["id"],))]
            val, unit = (st.median(w), "W at idle") if w else (None, None)
        elif r["workload"] == "ffmpeg.transcode":
            val, unit = s.get("runtime_s"), "s wall"
        else:
            val, unit = rate(r)
        by[(r["workload"], unit, iv)].append(val)
    out = collections.defaultdict(dict)
    for (wl, unit, iv), vals in by.items():
        out[f"{wl} ({unit})"][f"{iv:.0f} ms"] = stat(vals)
    verdict = {}
    for k, d in out.items():
        base = d.get("1000 ms")
        for iv, s in d.items():
            if base and s and iv != "1000 ms":
                verdict[f"{k} @ {iv}"] = s["median"] / base["median"] - 1
    return {"by_workload": out, "change_vs_1s_sampling": verdict}


def markdown(res: dict) -> str:
    L = ["# Observer effect in the gauge experiments\n"]
    L.append("## O1 Nsight inside the container\n")
    L.append("| workload | mode | steady rate off | on | steady change | wall off s | wall on s | added s | ratio | after exit off / on s | launches/s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in res["o1"]["pairs"]:
        L.append(f"| {p['workload']} | {p['mode']} | {fmt(p['rate_off'], 1)} | {fmt(p['rate_on'], 1)} | "
                 f"{p['steady_change'] * 100:+.1f}% | " if p["steady_change"] is not None else f"| {p['workload']} | {p['mode']} | – | – | – | ")
        L[-1] += (f"{fmt(p['wall_off_s'])} | {fmt(p['wall_on_s'])} | {p['wall_added_s']:+.1f} | {p['wall_ratio']:.2f} | "
                  f"{fmt(p['after_exit_off_s'])} / {fmt(p['after_exit_on_s'])} | "
                  f"{p['launches_per_s']:,.0f} |" if p.get("launches_per_s") else f"{fmt(p['after_exit_off_s'])} / {fmt(p['after_exit_on_s'])} | – |")
    rho = res["o1"]["launch_rate_vs_steady_change_spearman"]
    if rho is not None:
        L.append(f"\nSpearman rank correlation between kernel launches per second and the steady-state change: "
                 f"{rho:+.2f} over {res['o1']['n_points']} conditions (negative: more launches, bigger slowdown).")
    o2 = res["o2"]
    L.append("\n## O2 gauge's host sampler (100 ms)\n")
    L.append(f"- CPU per sample: {fmt(o2['cpu_ms_per_sample'], 2, ' ms')}")
    L.append(f"- Cores used by the sampler over a run: {fmt(o2['collector_cores'], 3)}")
    L.append(f"- Sampler CPU relative to the job's own CPU: {fmt(o2['collector_vs_workload_cpu'], 3)}")
    L.append(f"- Missed samples: {o2['missed_samples']} of {o2['samples']}")
    L.append(f"- {o2['note']}")
    L.append("\n## O3 measurement floor (idle GPU, observed)\n")
    L.append(f"Unobserved idle: {res['o3']['unobserved_idle_w']} W ({res['o3']['note']}).\n")
    for k, s in res["o3"]["conditions"].items():
        L.append(f"- {k}: {fmt(s, 1, ' W')}")
    L.append("\n## O4 harness: time from launch request to first output\n")
    L.append("| variant | ms | added vs native ms |\n|---|---|---|")
    for k, s in sorted(res["o4"]["variants"].items(), key=lambda kv: kv[1]["median"]):
        L.append(f"| {k} | {fmt(s, 1)} | {s['added_vs_native_ms']:+.1f} |" if s["added_vs_native_ms"] is not None else f"| {k} | {fmt(s, 1)} | – |")
    o5 = res["o5"]
    L.append("\n## O5 attribution in co-runs\n")
    L.append(f"{o5['corun_members']} co-run members: {o5['with_process_scope_trace']} with a process-scope trace, "
             f"{o5['with_own_rate']} with their own rate, {o5['device_scope_only']} with device-scope evidence only. {o5['note']}.")
    if res["o6"]["by_workload"]:
        L.append("\n## O6 sampler interval (C9)\n")
        for k, d in res["o6"]["by_workload"].items():
            L.append(f"- {k}: " + "; ".join(f"{iv}: {fmt(s, 2)}" for iv, s in sorted(d.items(), key=lambda kv: float(kv[0].split()[0]))))
        L.append("\nChange vs 1 s sampling: " + ", ".join(f"{k} {v * 100:+.2f}%" for k, v in res["o6"]["change_vs_1s_sampling"].items()))
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(paths.DATA / "gauge.sqlite"))
    ap.add_argument("--json")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    rows = done(conn)
    res = {"o1": o1_nsight(rows, launch_rates(conn)), "o2": o2_sampler(rows), "o3": o3_floor(rows, conn), "o4": o4_harness(conn),
           "o5": o5_attribution(rows), "o6": o6_interval(rows, conn)}
    print(markdown(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
