"""Co-run interference, per process: each co-run job against the median of its identical solo runs
(same params, same profile mode, so profiler overhead cancels)."""
from __future__ import annotations

import json
import statistics as st

from gauge.workloads.base import get


def _key(conn_row):
    spec = json.loads(conn_row["spec_json"])
    w = get(conn_row["workload"])
    return (conn_row["workload"], json.dumps(w.validate(spec["params"]).model_dump(), sort_keys=True),
            spec.get("profile", "none"), spec.get("sharing", "timeslice"))


def _tps(outcome: dict) -> float | None:
    rs = [r for r in outcome.get("responses", [outcome]) if "avg_ts" in r]
    return st.mean(r["avg_ts"] for r in rs) if rs else None


def _window(r) -> tuple[float, float] | None:
    """Absolute (start, end) of a run's compute phase when it marks one, else its whole workload."""
    if r["container_started"] is None or r["workload_finished"] is None:
        return None
    t0 = r["container_started"]
    ms = json.loads(r["sj"]).get("marks") or []
    starts = [m["t"] for m in ms if m["mark"] == "compute_started"]
    ends = [m["t"] for m in ms if m["mark"] == "compute_finished"]
    if starts and ends:
        return t0 + starts[0], t0 + ends[-1]
    return t0, r["workload_finished"]


def overlap_frac(me, others) -> float | None:
    """Share of my window during which at least one partner was running (its whole workload window)."""
    w = _window(me)
    if w is None or w[1] <= w[0]:
        return None
    spans = sorted((max(o["container_started"], w[0]), min(o["workload_finished"], w[1])) for o in others
                   if o["container_started"] is not None and o["workload_finished"] is not None)
    covered, cur = 0.0, w[0]
    for a, b in spans:
        a = max(a, cur)
        if b > a:
            covered += b - a
            cur = b
    return covered / (w[1] - w[0])


def rows(conn) -> list[dict]:
    prof = {r[0]: json.loads(r[1]) for r in conn.execute("SELECT run_id, json FROM gpu_profile")}
    runs = conn.execute("SELECT r.id, r.workload, r.spec_json, r.corun_group, r.outcome_json, r.container_started,"
                        " r.workload_finished, s.json AS sj FROM run r"
                        " JOIN summary s ON s.run_id=r.id WHERE r.status='done' AND r.failure_class='success'"
                        " AND json_extract(r.spec_json,'$.profile') IN ('trace','metrics')").fetchall()
    solo: dict = {}
    for r in runs:
        if not r["corun_group"]:
            solo.setdefault(_key(r), []).append(r)
    groups: dict = {}
    for r in runs:
        if r["corun_group"]:
            groups.setdefault(r["corun_group"], []).append(r)
    sizes = dict(conn.execute("SELECT corun_group, COUNT(*) FROM run WHERE corun_group IS NOT NULL GROUP BY corun_group"))
    out = []
    for g, members in groups.items():
        if len(members) != sizes.get(g):  # a member failed or was abandoned: the group's conditions are unknown
            continue
        for r in members:
            ref = solo.get(_key(r))
            if not ref:
                continue
            s = json.loads(r["sj"])
            p = prof.get(r["id"], {}).get("process", {})
            med = lambda f: st.median([v for v in (f(x) for x in ref) if v is not None]) if any(f(x) is not None for x in ref) else None  # noqa: E731
            solo_rt = med(lambda x: json.loads(x["sj"]).get("runtime_s"))
            solo_busy = med(lambda x: prof.get(x["id"], {}).get("process", {}).get("kernel_busy_s"))
            solo_ksum = med(lambda x: prof.get(x["id"], {}).get("process", {}).get("kernel_time_sum_s"))
            solo_tps = med(lambda x: _tps(json.loads(x["outcome_json"] or "{}")))
            tps = _tps(json.loads(r["outcome_json"] or "{}"))
            # the compute phase alone (workloads that mark it): model load would dilute the slowdown
            comp = (s.get("phase_s") or {}).get("compute")
            solo_comp = med(lambda x: (json.loads(x["sj"]).get("phase_s") or {}).get("compute"))
            partners = [m for m in members if m["id"] != r["id"]]
            others = [m["workload"] for m in partners]
            out.append({
                "run_id": r["id"], "corun_group": g, "n_jobs": len(members), "workload": r["workload"],
                "params": json.loads(r["spec_json"])["params"], "alongside": ", ".join(sorted(others)),
                "sharing": json.loads(r["spec_json"]).get("sharing", "timeslice"),
                "n_solo_refs": len(ref),
                # < ~0.8 means the partner was gone for part of this job's compute: its slowdown understates contention
                "overlap_frac": overlap_frac(r, partners),
                "runtime_s": s.get("runtime_s"), "solo_runtime_s": solo_rt,
                "slowdown": s["runtime_s"] / solo_rt if solo_rt else None,
                # process scope: this job's own kernels
                "kernel_time_s": p.get("kernel_time_sum_s"), "solo_kernel_time_s": solo_ksum,
                "kernel_stretch": p["kernel_time_sum_s"] / solo_ksum if solo_ksum and p.get("kernel_time_sum_s") else None,
                "kernel_busy_s": p.get("kernel_busy_s"), "solo_kernel_busy_s": solo_busy,
                "tokens_per_s": tps, "solo_tokens_per_s": solo_tps,
                "throughput_ratio": tps / solo_tps if tps and solo_tps else None,
                "compute_s": comp, "solo_compute_s": solo_comp,
                "compute_slowdown": comp / solo_comp if comp and solo_comp else None,
            })
    return sorted(out, key=lambda x: (x["n_jobs"], x["alongside"], x["workload"]))
