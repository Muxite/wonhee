"""Raw-level tables for the findings dashboard, beyond the per-contrast summaries: every run, 1 s time series,
phase marks, kernel mixes, energy per unit of work, every co-run member, batching levels, container starts.
Written next to dashboard_export.py's tables (data/gauge/dashboard/); large ones as CSV.

    uv run python profiling/analysis/dashboard_extra.py [--db ...] [--out data/gauge/dashboard]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sqlite3
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "profiling" / "analysis")]
import capacity as CA  # noqa: E402
from gauge import paths  # noqa: E402
from gauge.dataset import _experiment  # noqa: E402
from gauge.interference import overlap_frac, rows as interference_rows  # noqa: E402
from gauge.workloads.wonhee import clip_duration  # noqa: E402

GIB = 2**30


def r(v, d=3):
    return round(v, d) if isinstance(v, float) else v


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def short_params(wl: str, p: dict) -> str:
    keep = {k: v for k, v in p.items() if k not in ("reps",)}
    return ", ".join(f"{k}={v}" for k, v in keep.items())


def load(conn):
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT r.*, s.json AS sj FROM run r LEFT JOIN summary s ON s.run_id=r.id"
                        " WHERE r.status='done' ORDER BY r.id").fetchall()


def experiment(row) -> str:
    spec = json.loads(row["spec_json"])
    return (spec.get("provenance") or {}).get("experiment") or _experiment(spec, row["corun_group"])


def work_units(row) -> tuple[float | None, str | None]:
    """Useful work a run did, in its natural unit, for energy per unit (whole run, load included)."""
    p = json.loads(row["spec_json"])["params"]
    o = json.loads(row["outcome_json"] or "{}")
    resp = o.get("responses") or ([o] if o else [])
    wl = row["workload"]
    if wl == "wonhee.gpu" and p.get("task") in ("whisper", "separate", "mask", "embed") and not p.get("hold"):
        per = clip_duration(p.get("clip_s", 20)) if p["task"] in ("whisper", "separate") else p.get("images", 4)
        return per * p.get("repeat", 2), "audio s" if p["task"] in ("whisper", "separate") else "images"
    if wl == "wonhee.gpu" and p.get("task") == "worker":
        return clip_duration(p.get("clip_s", 20)) * p.get("repeat", 2), "audio s"
    if wl == "wonhee.speech_subs":
        return clip_duration(p.get("clip_s", 20)), "audio s"
    if wl == "ffmpeg.transcode":
        return p.get("seconds", 20) * 30, "frames"
    if wl == "llamacpp.phase":
        n = sum(x.get("n_gen", 0) or x.get("n_prompt", 0) for x in resp if "avg_ts" in x)
        reps = p.get("reps", 3)
        return (p.get("tokens", 256) * (reps + 1)) if n else None, "tokens"
    if wl == "llamacpp.batched":
        n = sum(x.get("pl", 0) * (x.get("tg", 0) + x.get("pp", 0)) for x in resp if "pl" in x)
        return n or None, "tokens"
    if wl == "train.small":
        return p.get("batch", 64) * p.get("steps", 30), "images"
    if wl == "stress.gpu":
        v = [x.get("units") for x in resp if "units" in x]
        u = [x.get("unit") for x in resp if "unit" in x]
        return (v[0], u[0]) if v else (None, None)
    return None, None


# --- tables ------------------------------------------------------------------------------------------------------

def runs_table(rows) -> list[dict]:
    groups = collections.Counter(x["corun_group"] for x in rows if x["corun_group"])
    out = []
    for x in rows:
        s = json.loads(x["sj"] or "{}")
        spec = json.loads(x["spec_json"])
        ph = s.get("phase_s") or {}
        units, unit = work_units(x)
        e = s.get("dev_energy_j")
        out.append({
            "run_id": x["id"], "experiment": experiment(x), "workload": x["workload"],
            "params": short_params(x["workload"], spec["params"]), "profile": spec.get("profile", "none"),
            "sharing": spec.get("sharing", "timeslice"), "corun_group": x["corun_group"] or "",
            "jobs_in_group": groups.get(x["corun_group"], 1), "outcome": x["failure_class"], "started_at": iso(x["container_started"]),
            "runtime_s": r(s.get("runtime_s")), "cpu_s": r(s.get("cpu_time_s")), "cpu_mean_cores": r(s.get("cpu_mean_cores")),
            "ram_peak_gib": r((s.get("ram_peak_bytes") or 0) / GIB), "vram_peak_gib": r((s.get("vram_peak_bytes") or 0) / GIB),
            "gpu_util_mean": r(s.get("dev_util_mean")), "memctl_util_mean": r(s.get("dev_mem_util_mean")),
            "power_mean_w": r(s.get("dev_power_mean_w")), "power_peak_w": r(s.get("dev_power_peak_w")),
            "energy_j": r(e), "nvenc_mean": r(s.get("dev_enc_util_mean")), "nvdec_mean": r(s.get("dev_dec_util_mean")),
            "temp_peak_c": r(s.get("dev_temp_peak_c")), "throttled": s.get("dev_throttle_any"),
            "load_s": r(ph.get("load")), "compute_s": r(ph.get("compute")), "calls": len(ph.get("spans") or []),
            "work_units": r(units), "work_unit": unit,
            "energy_per_unit_j": r(e / units, 5) if e and units else None,
            "image_digest": (x["image_digest"] or "")[:19],
        })
    return out


def timeseries(conn, rows, since_id: int) -> list[dict]:
    """1 s bins of the 100 ms samples (device-scope GPU, container CPU/RAM), for runs from `since_id` on."""
    out = []
    for x in rows:
        if x["id"] < since_id:
            continue
        smp = conn.execute("SELECT * FROM sample WHERE run_id=? ORDER BY t", (x["id"],)).fetchall()
        bins = collections.defaultdict(list)
        for s in smp:
            bins[int(s["t"])].append(s)
        prev = None
        for t in sorted(bins):
            b = bins[t]
            m = lambda k: r(st.mean([s[k] for s in b if s[k] is not None]), 2) if any(s[k] is not None for s in b) else None  # noqa: E731
            cpu = None
            if prev is not None and b[-1]["cpu_usage_usec"] is not None and prev["cpu_usage_usec"] is not None and b[-1]["t"] > prev["t"]:
                cpu = r((b[-1]["cpu_usage_usec"] - prev["cpu_usage_usec"]) / 1e6 / (b[-1]["t"] - prev["t"]), 2)
            pcie = None
            has = "dev_pcie_rx_bytes" in b[-1].keys()  # ledgers from before the PCIe counters lack the columns
            if has and prev is not None and b[-1]["dev_pcie_rx_bytes"] is not None and prev["dev_pcie_rx_bytes"] is not None \
                    and b[-1]["t"] > prev["t"]:
                dt = b[-1]["t"] - prev["t"]
                pcie = r(max(b[-1]["dev_pcie_rx_bytes"] - prev["dev_pcie_rx_bytes"], b[-1]["dev_pcie_tx_bytes"] - prev["dev_pcie_tx_bytes"]) / dt / 1e9, 3)
            out.append({"run_id": x["id"], "t": t, "power_w": m("dev_power_w"), "gpu_util": m("dev_util"),
                        "memctl_util": m("dev_mem_util"), "vram_used_gib": r((m("dev_mem_used") or 0) / GIB, 2),
                        "vram_proc_gib": r((m("vram_proc_bytes") or 0) / GIB, 2), "sm_mhz": m("dev_sm_mhz"),
                        "mem_mhz": m("dev_mem_mhz"), "temp_c": m("dev_temp_c"), "nvenc": m("dev_enc_util"),
                        "nvdec": m("dev_dec_util"), "cpu_cores": cpu, "ram_gib": r((m("mem_current") or 0) / GIB, 2),
                        "power_capped": int(any((s["dev_throttle"] or 0) & 4 for s in b)), "pcie_gbs": pcie})
            prev = b[-1]
    return out


def marks_table(rows, since_id: int) -> list[dict]:
    out = []
    for x in rows:
        if x["id"] < since_id:
            continue
        for m in json.loads(x["sj"] or "{}").get("marks", []):
            out.append({"run_id": x["id"], "mark": m["mark"], "t": r(m["t"])})
    return out


def kernels(conn, rows) -> tuple[list[dict], list[dict], list[dict]]:
    byid = {x["id"]: x for x in rows}
    cats, tops, procs = [], [], []
    for pr in conn.execute("SELECT run_id, json FROM gpu_profile ORDER BY run_id"):
        x = byid.get(pr[0])
        if not x:
            continue
        p = json.loads(pr[1]).get("process") or {}
        exp, wl, params = experiment(x), x["workload"], short_params(x["workload"], json.loads(x["spec_json"])["params"])
        procs.append({"run_id": x["id"], "experiment": exp, "workload": wl, "params": params,
                      **{k: r(v) for k, v in p.items() if not isinstance(v, (dict, list)) and not k.endswith("_ns")}})
        for c, v in (p.get("by_category") or {}).items():
            cats.append({"run_id": x["id"], "experiment": exp, "workload": wl, "params": params, "category": c,
                         "share": r(v.get("share")), "time_s": r(v.get("time_s")), "count": v.get("count")})
        for k in p.get("top_kernels") or []:
            tops.append({"run_id": x["id"], "workload": wl, "kernel": (k.get("name") or "")[:90], "category": k.get("category"),
                         "share": r(k.get("share")), "count": k.get("count"), "median_us": r(k.get("median_us"))})
    return cats, tops, procs


def coruns(conn, rows) -> list[dict]:
    byid = {x["id"]: x for x in rows}
    groups = collections.defaultdict(list)
    for x in rows:
        if x["corun_group"]:
            groups[x["corun_group"]].append(x)
    out = []
    for i in interference_rows(conn):
        me = byid[i["run_id"]]
        out.append({"run_id": i["run_id"], "experiment": experiment(me), "workload": i["workload"], "alongside": i["alongside"],
                    "jobs": i["n_jobs"], "sharing": i.get("sharing"), "profile": json.loads(me["spec_json"]).get("profile", "none"),
                    "slowdown": r(i["slowdown"]), "compute_slowdown": r(i.get("compute_slowdown")),
                    "kernel_stretch": r(i["kernel_stretch"]), "throughput_ratio": r(i["throughput_ratio"]),
                    "overlap": r(overlap_frac(me, [m for m in groups[me["corun_group"]] if m["id"] != me["id"]])),
                    "solo_refs": i["n_solo_refs"]})
    return out


def batching(rows) -> list[dict]:
    out = []
    for x in rows:
        if x["workload"] != "llamacpp.batched" or x["failure_class"] != "success":
            continue
        p = json.loads(x["spec_json"])["params"]
        o = json.loads(x["outcome_json"] or "{}")
        for lv in o.get("responses") or ([o] if o else []):
            if "pl" not in lv:
                continue
            out.append({"run_id": x["id"], "experiment": experiment(x), "model": p.get("model"), "sequences": lv["pl"],
                        "prompt_tokens": lv.get("pp"), "gen_tokens": lv.get("tg"), "kv_tokens": lv.get("n_kv"),
                        "decode_tok_s": r(lv.get("speed_tg"), 1), "prefill_tok_s": r(lv.get("speed_pp"), 1),
                        "decode_s": r(lv.get("t_tg")), "per_sequence_tok_s": r(lv.get("speed_tg", 0) / lv["pl"], 1),
                        "profile": json.loads(x["spec_json"]).get("profile", "none")})
    return out


def starts(conn) -> list[dict]:
    out = []
    for s in conn.execute("SELECT * FROM startbench WHERE error IS NULL"):
        def d(a, b):
            return r((s[b] - s[a]) * 1000, 2) if s[a] is not None and s[b] is not None else None
        out.append({"variant": s["variant"], "task": s["task"], "burst": s["burst"], "rep": s["rep"],
                    "to_created_ms": d("t_req", "created"), "to_started_ms": d("t_req", "started"),
                    "to_first_output_ms": d("t_req", "first"), "total_ms": d("t_req", "t_ret")})
    return out


def stress_series(conn, rows) -> list[dict]:
    """The stressors' cumulative work over time in co-runs (C3), as rate per ~0.5 s, with the LLM's window."""
    out = []
    groups = collections.defaultdict(list)
    for x in rows:
        if x["corun_group"] and experiment(x) == "C3 stressors":
            groups[x["corun_group"]].append(x)
    for g, members in groups.items():
        llm = next((m for m in members if m["workload"] == "llamacpp.phase"), None)
        for m in members:
            if m["workload"] != "stress.gpu":
                continue
            o = json.loads(m["outcome_json"] or "{}")
            ser = o.get("series") or []
            t0 = m["container_started"]
            for (ta, ua), (tb, ub) in zip(ser, ser[1:]):
                out.append({"run_id": m["id"], "group": g, "mode": json.loads(m["spec_json"])["params"]["mode"],
                            "t": r(tb - t0, 2), "rate": r((ub - ua) / (tb - ta), 2), "unit": f"{o.get('unit')}/s",
                            "llm_from": r(llm["container_started"] - t0, 2) if llm else None,
                            "llm_to": r(llm["workload_finished"] - t0, 2) if llm else None})
    for x in rows:  # solo series as the reference
        if x["workload"] == "stress.gpu" and not x["corun_group"] and experiment(x) == "C3 stressors":
            o = json.loads(x["outcome_json"] or "{}")
            ser = o.get("series") or []
            for (ta, ua), (tb, ub) in zip(ser, ser[1:]):
                out.append({"run_id": x["id"], "group": "solo", "mode": json.loads(x["spec_json"])["params"]["mode"],
                            "t": r(tb - x["container_started"], 2), "rate": r((ub - ua) / (tb - ta), 2),
                            "unit": f"{o.get('unit')}/s", "llm_from": None, "llm_to": None})
    return out


def capacity_raw(conn) -> list[dict]:
    out = []
    for g in CA.rungs(conn):
        out.append({"ladder": g["ladder"], "level": g["level"], "copies": g["n"], "run_ids": " ".join(map(str, g["run_ids"])),
                    "rate": r(g.get("rate")), "unit": g["unit"], "failed": g.get("failed"),
                    "all_computing_s": r(g.get("all_computing_s")),
                    **{k: r((g.get("res") or {}).get(k)) for k in (*CA.RESOURCES, "power_capped_share", "cpu_cores", "power_w", "vram_gib")}})
    return out


def write(out: Path, name: str, rows: list[dict], fmt: str):
    p = out / f"{name}.{fmt}"
    if fmt == "csv":
        cols = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["empty"]
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    else:
        p.write_text(json.dumps(rows, default=str))
    print(f"{name:16s} {len(rows):6d} rows  {p.stat().st_size / 1e3:8.1f} kB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(paths.DATA / "gauge.sqlite"))
    ap.add_argument("--out", default=str(paths.DATA / "dashboard"))
    ap.add_argument("--series-since", type=int, default=1, help="first run id with 1 s series and marks")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    rows = load(conn)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cats, tops, procs = kernels(conn, rows)
    write(out, "runs", runs_table(rows), "csv")
    write(out, "series", timeseries(conn, rows, a.series_since), "csv")
    write(out, "marks", marks_table(rows, a.series_since), "csv")
    write(out, "kernel_mix", cats, "csv")
    write(out, "top_kernels", tops, "csv")
    write(out, "gpu_process", procs, "csv")
    write(out, "coruns", coruns(conn, rows), "csv")
    write(out, "batching", batching(rows), "csv")
    write(out, "starts", starts(conn), "csv")
    write(out, "stress_series", stress_series(conn, rows), "csv")
    write(out, "capacity_raw", capacity_raw(conn), "csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
