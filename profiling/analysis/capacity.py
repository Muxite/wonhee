"""C8 capacity ladders: for each kind of work, throughput per rung of load and how close each resource came to its
ceiling, then the pre-registered plateau and limiter rules (profiling/sweeps/x/c8_*.yaml).

A rung is one load level: N concurrent copies (a co-run group), an in-process batch size, or one level of
llama-batched-bench. Resources are read over the rung's window, where every copy was computing:
device-scope NVML samples (100 ms), container CPU of every copy, PCIe from the cumulative counters.

    uv run python profiling/analysis/capacity.py [--db ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sqlite3
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gauge import paths  # noqa: E402
from gauge.workloads.wonhee import clip_duration  # noqa: E402

PCIE_CEILING = 28.8e9  # B/s, pinned host->device copy measured in X1 (micro.ceilings)
CPU_CORES = 16
NEAR = 0.9  # a resource "topped out" at >= 90% of its ceiling
FLAT = 1.10  # plateau: < 10% more throughput per doubling of load
RESOURCES = ("sm_busy", "memctl_busy", "vram", "power", "sm_clock", "nvenc", "nvdec", "cpu", "pcie")


def runs(conn):
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT r.*, s.json AS sj FROM run r JOIN summary s ON s.run_id=r.id WHERE r.status='done'"
                        " AND json_extract(r.spec_json, '$.provenance.experiment') = 'C8 capacity' ORDER BY r.id").fetchall()


def samples(conn, run_id):
    return [dict(r) for r in conn.execute("SELECT * FROM sample WHERE run_id=? ORDER BY t", (run_id,))]


def marks_abs(r, name):
    s = json.loads(r["sj"])
    return [r["container_started"] + m["t"] for m in s.get("marks", []) if m["mark"] == name]


def host(r):
    return json.loads(r["host_json"] or "{}")


def resources(conn, members, t0, t1) -> dict | None:
    """Fractions of each ceiling over [t0, t1] (unix time). Device fields from the first member's samples (they
    are device-wide), CPU summed over every member's container."""
    if t1 - t0 < 0.5:
        return None
    h = host(members[0])
    rows = [x for x in samples(conn, members[0]["id"]) if t0 <= members[0]["container_started"] + x["t"] <= t1]
    if len(rows) < 3:
        return None
    mean = lambda k: st.mean([x[k] for x in rows if x.get(k) is not None]) if any(x.get(k) is not None for x in rows) else None  # noqa: E731
    out = {"sm_busy": (mean("dev_util") or 0) / 100, "memctl_busy": (mean("dev_mem_util") or 0) / 100,
           "vram": max(x["dev_mem_used"] or 0 for x in rows) / (h.get("vram_total") or 1),
           "power": (mean("dev_power_w") or 0) / (h.get("power_limit_w") or 300),
           "power_capped_share": st.mean([1.0 if (x.get("dev_throttle") or 0) & 4 else 0.0 for x in rows]),
           "sm_clock": (mean("dev_sm_mhz") or 0) / (h.get("max_sm_mhz") or 3210),
           "nvenc": (mean("dev_enc_util") or 0) / 100, "nvdec": (mean("dev_dec_util") or 0) / 100}
    tx = [x for x in rows if x.get("dev_pcie_tx_bytes") is not None]
    if len(tx) >= 2:
        dt = tx[-1]["t"] - tx[0]["t"]
        out["pcie"] = max(tx[-1]["dev_pcie_tx_bytes"] - tx[0]["dev_pcie_tx_bytes"],
                          tx[-1]["dev_pcie_rx_bytes"] - tx[0]["dev_pcie_rx_bytes"]) / dt / PCIE_CEILING if dt > 0 else None
    cores = 0.0
    for m in members:
        cr = [x for x in samples(conn, m["id"]) if t0 <= m["container_started"] + x["t"] <= t1 and x.get("cpu_usage_usec") is not None]
        if len(cr) >= 2 and cr[-1]["t"] > cr[0]["t"]:
            cores += (cr[-1]["cpu_usage_usec"] - cr[0]["cpu_usage_usec"]) / 1e6 / (cr[-1]["t"] - cr[0]["t"])
    out["cpu"] = cores / CPU_CORES
    out["cpu_cores"] = cores
    out["power_w"] = mean("dev_power_w")
    out["vram_gib"] = max(x["dev_mem_used"] or 0 for x in rows) / 2**30
    return out


# --- rungs -------------------------------------------------------------------------------------------------------

def _units(p) -> tuple[float, str]:
    if p["task"] in ("whisper", "separate"):
        return clip_duration(p.get("clip_s", 20)), "audio s/s"
    return float(p.get("images", 4)), "images/s"


def rungs(conn) -> list[dict]:
    """One dict per (ladder, level, repeat): throughput, unit, resource fractions, failure."""
    out = []
    groups = collections.defaultdict(list)
    for r in runs(conn):
        if json.loads(r["spec_json"]).get("profile", "none") != "none":
            continue
        groups[r["corun_group"] or f"solo{r['id']}"].append(r)
    for members in groups.values():
        r0 = members[0]
        p = json.loads(r0["spec_json"])["params"]
        wl, n = r0["workload"], len(members)
        failed = [m["failure_class"] for m in members if m["failure_class"] != "success"]
        base = {"n": n, "run_ids": [m["id"] for m in members], "failed": failed[0] if failed else None}
        if wl == "wonhee.gpu":
            units, unit = _units(p)
            if p["task"] == "whisper" and p.get("whisper_batch", 0) > 1:
                ladder, level = "whisper: batch in one process", p["whisper_batch"]
            elif p["task"] == "embed" and n == 1 and p.get("batch") != 64:
                ladder, level = "SigLIP2: batch in one process", p["batch"]
            else:
                ladder, level = f"{p['task']}: concurrent processes", n
                if p["task"] == "embed" and n == 1:  # batch 64, one process: both ladders' shared point
                    out.append({**base, "ladder": "SigLIP2: batch in one process", "level": 64, "unit": unit,
                                **_wonhee_rate(conn, members, units)})
            if p["task"] == "whisper" and p.get("whisper_batch", 0) <= 1 and n == 1:  # shared point: batch 1
                out.append({**base, "ladder": "whisper: batch in one process", "level": 1, "unit": unit,
                            **_wonhee_rate(conn, members, units)})
            out.append({**base, "ladder": ladder, "level": level, "unit": unit, **_wonhee_rate(conn, members, units)})
        elif wl == "ffmpeg.transcode":
            t0, t1 = max(m["container_started"] for m in members), min(m["workload_finished"] for m in members)
            rate = sum(p["seconds"] * 30 / (m["workload_finished"] - m["container_started"]) for m in members) if not failed else None
            out.append({**base, "ladder": f"NVENC transcode, {p.get('decode', 'cpu')} decode: concurrent copies",
                        "level": n, "unit": "frames/s", "rate": rate, "res": resources(conn, members, t0, t1)})
        elif wl == "llamacpp.batched":
            o = json.loads(r0["outcome_json"] or "{}")
            levels = [x for x in (o.get("responses") or ([o] if o else [])) if "speed_tg" in x]
            name = f"{p['model']} decode: batch in one process"
            if failed and not levels:
                out.append({**base, "ladder": name, "level": max(int(x) for x in str(p["parallel"]).split(",")),
                            "unit": "tok/s", "rate": None, "res": None})
            for x in levels:
                at = x.get("gauge_at")
                res = resources(conn, members, at - x["t_tg"], at) if at else None  # the decode part of this level
                out.append({**base, "ladder": name, "level": x["pl"], "unit": "tok/s", "rate": x["speed_tg"], "res": res})
        elif wl == "llamacpp.phase":
            o = json.loads(r0["outcome_json"] or "{}")
            t0 = (marks_abs(r0, "start_tool") or [r0["container_started"]])[0] + 3  # skip the model load
            out.append({**base, "ladder": f"{p['model']} prefill: prompt tokens", "level": p["tokens"], "unit": "tok/s",
                        "rate": o.get("avg_ts"), "res": resources(conn, members, t0, r0["workload_finished"])})
        elif wl == "train.small":
            o = json.loads(r0["outcome_json"] or "{}")
            s, e = marks_abs(r0, "compute_started"), marks_abs(r0, "compute_finished")
            out.append({**base, "ladder": f"{p['model']} training (AMP): batch", "level": p["batch"], "unit": "images/s",
                        "rate": o.get("imgs_per_s"), "res": resources(conn, members, s[0], e[-1]) if s and e else None})
    return out


def _wonhee_rate(conn, members, units) -> dict:
    """Sum of each copy's warm rate (units / median of spans after the first), and the window where all copies
    were inside their compute phase."""
    rates, starts, ends = [], [], []
    for m in members:
        ph = json.loads(m["sj"]).get("phase_s") or {}
        spans = ph.get("spans") or []
        if m["failure_class"] != "success" or len(spans) < 2:
            return {"rate": None, "res": None}
        rates.append(units / st.median(spans[1:]))
        s, e = marks_abs(m, "compute_started"), marks_abs(m, "compute_finished")
        starts.append(s[1] if len(s) > 1 else s[0])  # from the second call on: past the first-call warm-up
        ends.append(e[-1])
    t0, t1 = max(starts), min(ends)
    return {"rate": sum(rates), "res": resources(conn, members, t0, t1), "all_computing_s": max(0.0, t1 - t0)}


# --- rules -------------------------------------------------------------------------------------------------------

def summarize(rs: list[dict]) -> dict:
    lad = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rs:
        lad[r["ladder"]][r["level"]].append(r)
    out = {}
    for name, levels in lad.items():
        steps = []
        for level in sorted(levels):
            reps = levels[level]
            ok = [r for r in reps if r["rate"] is not None and not r["failed"]]
            fr = {k: st.median([r["res"][k] for r in ok if r.get("res") and r["res"].get(k) is not None])
                  for k in (*RESOURCES, "power_capped_share", "cpu_cores", "power_w", "vram_gib")
                  if any(r.get("res") and r["res"].get(k) is not None for r in ok)}
            steps.append({"level": level, "n_ok": len(ok), "failed": [r["failed"] for r in reps if r["failed"]],
                          "rate": st.median([r["rate"] for r in ok]) if ok else None,
                          "rate_min": min((r["rate"] for r in ok), default=None),
                          "rate_max": max((r["rate"] for r in ok), default=None), "res": fr, "unit": reps[0]["unit"]})
        good = [s for s in steps if s["rate"]]
        wall = next((s["level"] for s in steps if not s["rate"] and s["failed"]), None)
        plateau = None
        for a, b in zip(good, good[1:]):
            per_doubling = (b["rate"] / a["rate"]) ** (1 / math.log2(b["level"] / a["level"]))
            if per_doubling < FLAT:
                plateau = a
                break
        at = plateau or (good[-1] if good else None)
        limiter = None
        if at:
            hot = sorted((k for k in RESOURCES if (at["res"].get(k) or 0) >= NEAR), key=lambda k: -at["res"][k])
            limiter = hot[0] if len(hot) == 1 else ("co-limited: " + ", ".join(hot) if hot else "software / serialization")
        peak = max(good, key=lambda s: s["rate"]) if good else None
        out[name] = {"steps": steps, "plateau_level": plateau["level"] if plateau else None,
                     "judged_at_level": at["level"] if at else None, "limiter": limiter,
                     "peak_level": peak["level"] if peak else None, "peak_rate": peak["rate"] if peak else None,
                     "speedup_peak_vs_first": peak["rate"] / good[0]["rate"] if good and peak else None,
                     "oom_or_failure_at": wall}
    return out


def markdown(summary: dict) -> str:
    lines = []
    for name, d in sorted(summary.items()):
        unit = d["steps"][0]["unit"] if d["steps"] else ""
        lines.append(f"### {name}\n")
        lines.append("| level | throughput | " + " | ".join(RESOURCES) + " | power capped | failed |")
        lines.append("|" + "---|" * (len(RESOURCES) + 4))
        for s in d["steps"]:
            rate = f"{s['rate']:.1f} [{s['rate_min']:.1f}–{s['rate_max']:.1f}] {unit}" if s["rate"] else "–"
            cells = [f"{s['res'][k]:.2f}" if s["res"].get(k) is not None else "–" for k in RESOURCES]
            cap = f"{s['res']['power_capped_share']:.2f}" if s["res"].get("power_capped_share") is not None else "–"
            lines.append(f"| {s['level']} | {rate} | " + " | ".join(cells) + f" | {cap} | {','.join(s['failed']) or ''} |")
        sp = d["speedup_peak_vs_first"]
        lines.append(f"\n**Peak** {d['peak_rate']:.1f} {unit} at {d['peak_level']} ({sp:.2f}x the first rung). "
                     f"**Plateau** at {d['plateau_level']}. **Tops out first:** {d['limiter']} (judged at {d['judged_at_level']})."
                     + (f" **Wall:** failed at {d['oom_or_failure_at']}." if d["oom_or_failure_at"] else "") + "\n"
                     if d["peak_rate"] else "\nno successful rung\n")
    return "\n".join(lines)


# --- packing (c8_mix.yaml) ------------------------------------------------------------------------------------

def _job_rate(r) -> float | None:
    """One job's own throughput, in its natural unit, while it ran."""
    if r["failure_class"] != "success":
        return None
    p = json.loads(r["spec_json"])["params"]
    if r["workload"] == "llamacpp.batched":
        o = json.loads(r["outcome_json"] or "{}")
        v = [x["speed_tg"] for x in (o.get("responses") or [o]) if "speed_tg" in x]
        return v[-1] if v else None
    if r["workload"] == "ffmpeg.transcode":
        return p["seconds"] * 30 / (r["workload_finished"] - r["container_started"])
    if r["workload"] == "wonhee.gpu":
        spans = (json.loads(r["sj"]).get("phase_s") or {}).get("spans") or []
        return _units(p)[0] / st.median(spans[1:]) if len(spans) > 1 else None
    return None


def _label(r) -> str:
    p = json.loads(r["spec_json"])["params"]
    return {"llamacpp.batched": "LLM decode", "ffmpeg.transcode": "NVENC"}.get(r["workload"]) or p.get("task", r["workload"])


def packing(conn) -> dict:
    conn.row_factory = sqlite3.Row
    rs = conn.execute("SELECT r.*, s.json AS sj FROM run r JOIN summary s ON s.run_id=r.id WHERE r.status='done'"
                      " AND json_extract(r.spec_json, '$.provenance.experiment') = 'C8 packing'").fetchall()
    solo = collections.defaultdict(list)
    groups = collections.defaultdict(list)
    for r in rs:
        (groups[r["corun_group"]] if r["corun_group"] else solo[_label(r)]).append(r)
    base = {k: st.median([x for x in map(_job_rate, v) if x]) for k, v in solo.items() if any(map(_job_rate, v))}
    mixes = collections.defaultdict(list)
    for members in groups.values():
        key = " + ".join(sorted(_label(m) for m in members))
        shares = {_label(m): (_job_rate(m) / base[_label(m)]) if _job_rate(m) and _label(m) in base else None
                  for m in members}
        if all(v is not None for v in shares.values()):
            mixes[key].append(shares)
    out = {}
    for key, reps in mixes.items():
        gains = [sum(s.values()) for s in reps]
        worst = min(min(s.values()) for s in reps)
        out[key] = {"gain": st.median(gains), "gain_min": min(gains), "gain_max": max(gains), "n": len(reps),
                    "member_share": {k: st.median([s[k] for s in reps]) for k in reps[0]},
                    "packs_well": st.median(gains) >= 1.5 and worst >= 0.5}
    return {"solo_rates": base, "mixes": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(paths.DATA / "gauge.sqlite"))
    ap.add_argument("--json")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    summary = summarize(rungs(conn))
    pack = packing(conn)
    print(markdown(summary))
    if pack["mixes"]:
        print("### Packing: tasks with different limiters on one GPU\n")
        print("| mix | packing gain (sum of solo shares) | each member's share of its solo rate | packs well |")
        print("|---|---|---|---|")
        for k, v in sorted(pack["mixes"].items(), key=lambda kv: -kv[1]["gain"]):
            shares = ", ".join(f"{m} {x:.2f}" for m, x in v["member_share"].items())
            print(f"| {k} | {v['gain']:.2f} [{v['gain_min']:.2f}–{v['gain_max']:.2f}] n={v['n']} | {shares} | "
                  f"{'yes' if v['packs_well'] else 'no'} |")
    if a.json:
        Path(a.json).write_text(json.dumps({"ladders": summary, "packing": pack}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
