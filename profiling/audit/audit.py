"""P0 evidence audit of the gauge ledger: is the recorded data good enough to answer questions with, before any new
runs are added? Checks each item of the experiment plan's "things we need" that the ledger can show, and prints
facts only (counts, ratios, run ids). The interpretation lives in docs/profiling/AUDIT.md.

    uv run python profiling/audit/audit.py [--db data/gauge/gauge.sqlite] [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gauge import paths  # noqa: E402
from gauge.interference import overlap_frac, rows as interference_rows  # noqa: E402
from gauge.workloads.base import get  # noqa: E402


def _params(r) -> str:
    spec = json.loads(r["spec_json"])
    try:
        p = get(r["workload"]).validate(spec["params"]).model_dump()
    except Exception:  # noqa: BLE001 - a workload since removed or renamed: its raw params still identify it
        p = spec["params"]
    return json.dumps(p, sort_keys=True)


def condition(r) -> tuple:
    spec = json.loads(r["spec_json"])
    return r["workload"], _params(r), spec.get("profile", "none"), spec.get("sharing", "timeslice")


def audit(conn) -> dict:
    conn.row_factory = sqlite3.Row
    runs = conn.execute("SELECT r.*, s.json AS sj FROM run r LEFT JOIN summary s ON s.run_id=r.id ORDER BY r.id").fetchall()
    done = [r for r in runs if r["status"] == "done"]
    out: dict = {"runs": len(runs), "by_status": dict(collections.Counter(
        f"{r['status']}/{r['failure_class']}" for r in runs))}

    # 1. manifest: what each run records about itself
    host = [json.loads(r["host_json"] or "{}") for r in runs]
    out["manifest"] = {
        "image_digest": sum(bool(r["image_digest"]) for r in runs),
        "host_fingerprint": sum(bool(h) for h in host),
        "drivers": sorted({h.get("driver") for h in host if h}),
        "initial_state": sum(bool(r["initial_json"]) for r in runs),
        "argv_recorded": 0,  # only workload + params: the command is rebuilt from today's code
        "input_hashes_recorded": 0,  # media files and GGUF blobs are named, not hashed, per run
        "phase_marks": sum(r["marks_json"] not in (None, "[]") for r in runs),
        "experiment_label_recorded": sum(bool(json.loads(r["provenance_json"] or "{}").get("experiment")) for r in runs),
    }

    # 2. raw exports
    prof = [r for r in done if json.loads(r["spec_json"]).get("profile", "none") != "none"]
    nsys = paths.DATA / "nsys"
    out["raw"] = {
        "profiled_runs": len(prof),
        "profiled_with_nsys_rep": sum((nsys / f"run{r['id']}.nsys-rep").exists() for r in prof),
        "runs_with_100ms_samples": conn.execute("SELECT COUNT(DISTINCT run_id) FROM sample").fetchone()[0],
        "failed_or_censored_kept": sum(r["failure_class"] not in (None, "success") for r in runs),
    }

    # 3. repeats and order, solo conditions only
    solo = collections.defaultdict(list)
    for r in done:
        if not r["corun_group"] and r["failure_class"] == "success":
            solo[condition(r)].append(r)
    reps = collections.Counter(min(len(v), 3) for v in solo.values())
    ids = [r["id"] for r in done if not r["corun_group"]]
    pos = {rid: i for i, rid in enumerate(ids)}
    multi = [v for v in solo.values() if len(v) >= 2]
    contiguous = sum(max(pos[r["id"]] for r in v) - min(pos[r["id"]] for r in v) == len(v) - 1 for v in multi)
    spreads = []
    for v in multi:
        rt = [json.loads(r["sj"] or "{}").get("runtime_s") for r in v]
        rt = [x for x in rt if x]
        if len(rt) >= 2 and st.median(rt) > 0:
            spreads.append((max(rt) - min(rt)) / st.median(rt))
    out["repeats"] = {
        "solo_conditions": len(solo), "with_1": reps[1], "with_2": reps[2], "with_3plus": reps[3],
        "repeats_run_back_to_back": f"{contiguous}/{len(multi)}",
        "runtime_range_over_median": {"median": round(st.median(spreads), 3), "p90": round(sorted(spreads)[int(.9 * len(spreads))], 3),
                                      "max": round(max(spreads), 3)} if spreads else None,
    }

    # 4. conditions at start: other GPU tenants, temperature
    tenants, temps = 0, []
    for r in done:
        init = json.loads(r["initial_json"] or "{}")
        own = set(json.loads(r["sj"] or "{}").get("gpu_pids") or [])
        if [g for g in init.get("gpu_processes", []) if g["pid"] not in own]:
            tenants += 1
        if init.get("gpu_temp_c") is not None:
            temps.append(init["gpu_temp_c"])
    out["conditions"] = {"runs_with_other_gpu_process_at_start": tenants,
                         "start_temp_c": [min(temps), round(st.median(temps)), max(temps)] if temps else None,
                         "runs_throttled": sum(bool(json.loads(r["sj"] or "{}").get("dev_throttle_any")) for r in done)}

    # 5. profiler effect: the same solo condition recorded under more than one profile mode
    by_mode = collections.defaultdict(dict)
    for (wl, p, mode, sharing), v in solo.items():
        rt = [json.loads(r["sj"] or "{}").get("runtime_s") for r in v]
        rt = [x for x in rt if x]
        if rt:
            by_mode[(wl, p, sharing)][mode] = (st.median(rt), len(rt))
    pairs = []
    for (wl, p, _), m in by_mode.items():
        if "none" in m and len(m) > 1:
            for mode in ("trace", "metrics"):
                if mode in m:
                    pairs.append({"workload": wl, "params": json.loads(p), "mode": mode,
                                  "runtime_ratio_vs_none": round(m[mode][0] / m["none"][0], 3),
                                  "n_none": m["none"][1], "n_mode": m[mode][1]})
    out["profiler_effect"] = pairs

    # 6. co-runs: matching solo references, both sides, overlap
    inter = interference_rows(conn)
    members = {r["id"]: r for r in done if r["corun_group"]}
    groups = collections.defaultdict(list)
    for r in members.values():
        groups[r["corun_group"]].append(r)
    with_ref = {x["run_id"] for x in inter}
    both = sum(all(m["id"] in with_ref for m in g) for g in groups.values())
    ov = []
    for x in inter:
        me = members[x["run_id"]]
        o = overlap_frac(me, [m for m in groups[me["corun_group"]] if m["id"] != me["id"]])
        if o is not None:
            ov.append(o)
    out["coruns"] = {"groups": len(groups), "member_runs": len(members), "members_with_matching_solo": len(with_ref),
                     "groups_with_both_sides_referenced": both,
                     "overlap_frac": {"min": round(min(ov), 2), "median": round(st.median(ov), 2),
                                      "below_0.8": sum(o < 0.8 for o in ov)} if ov else None,
                     "sharing_modes": sorted({json.loads(r["spec_json"]).get("sharing", "timeslice") for r in members.values()})}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(paths.DATA / "gauge.sqlite"))
    ap.add_argument("--json")
    a = ap.parse_args()
    res = audit(sqlite3.connect(a.db))
    text = json.dumps(res, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
