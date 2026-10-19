"""Row tables for the findings dashboard, computed from the gauge ledger by the same code as the analysis scripts
(audit.py, round2.py, capacity.py). One JSON file per dataset in data/gauge/dashboard/. Re-run after new runs land,
then re-upload the changed files to the dashboard.

    uv run python profiling/analysis/dashboard_export.py [--db ...] [--out data/gauge/dashboard]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "profiling" / "analysis"), str(ROOT / "profiling" / "audit")]
import yaml  # noqa: E402

import audit as AU  # noqa: E402
import capacity as CA  # noqa: E402
import round2 as R2  # noqa: E402
from gauge import paths  # noqa: E402
from gauge.cli import expand  # noqa: E402

X = ROOT / "profiling" / "sweeps" / "x"
GIB = 2**30


def med(s):
    return s["median"] if s else None


def rng(s, k):
    return s[k] if s else None


def r3(v, d=3):
    return round(v, d) if isinstance(v, (int, float)) and v is not None else v


# --- progress ----------------------------------------------------------------------------------------------------

PLAN = [("C1 profiler", ["c1_profiler.yaml"]), ("C7 warm-cold", ["c7_warm_cold.yaml"]), ("C5 inputs", ["c5_inputs.yaml"]),
        ("C3 stressors", ["c3_stressors.yaml"]), ("C6 media", ["c6_media.yaml", "c6_parallel.yaml"]),
        ("C2 sharing", ["c2_sharing.yaml"]), ("C0 resident", ["c0_resident.yaml"]),
        ("C8 capacity", ["c8_ladders.yaml", "c8_copies.yaml"]), ("C8 packing", ["c8_mix.yaml"])]


def planned(files) -> int:
    n = 0
    for f in files:
        p = X / f
        if not p.exists():
            continue
        d = yaml.safe_load(p.read_text())
        if "jobs" in d:
            n += len(expand(d))
        else:
            n += sum(len(g if isinstance(g, list) else g["jobs"]) for g in d["pairs"]) * d.get("repeats", 1)
    return n


def progress(conn) -> list[dict]:
    rows = []
    for exp, files in PLAN:
        done, failed = conn.execute(
            "SELECT SUM(failure_class='success'), SUM(failure_class!='success') FROM run WHERE status='done'"
            " AND json_extract(spec_json,'$.provenance.experiment')=?", (exp,)).fetchone()
        plan = planned(files)
        done, failed = done or 0, failed or 0
        rows.append({"experiment": exp, "planned_runs": plan, "succeeded": done, "failed": failed,
                     "state": "complete" if done + failed >= plan else ("running" if done + failed else "queued")})
    return rows


# --- tables ------------------------------------------------------------------------------------------------------

def audit_rows(conn) -> list[dict]:
    a = AU.audit(conn)
    m, rw, rp, co, cd = a["manifest"], a["raw"], a["repeats"], a["coruns"], a["conditions"]
    return [
        {"need": "Run manifest", "status": "partly", "evidence": f"{m['image_digest']}/{a['runs']} runs keep a local image ID (not pullable once pruned); argv, input hashes and cache/JIT state were not recorded before 2026-10-19"},
        {"need": "Raw exports", "status": "yes", "evidence": f"{rw['profiled_with_nsys_rep']}/{rw['profiled_runs']} profiled runs keep their Nsight file; {rw['runs_with_100ms_samples']} runs keep 100 ms samples"},
        {"need": "Failed and OOM runs kept", "status": "yes", "evidence": f"{rw['failed_or_censored_kept']} failed or censored runs kept, never used as 'fits' labels"},
        {"need": "Alternating order", "status": "no", "evidence": f"{rp['repeats_run_back_to_back']} repeated conditions ran their repeats back to back"},
        {"need": "Repeats", "status": "thin", "evidence": f"{rp['with_1']} conditions ran once, {rp['with_2']} twice, {rp['with_3plus']} three or more times"},
        {"need": "Matching co-run baselines", "status": "partly", "evidence": f"{co['members_with_matching_solo']}/{co['member_runs']} co-run members have matching solo runs; {co['overlap_frac']['below_0.8'] if co['overlap_frac'] else 0} overlapped their partner < 80% of their run"},
        {"need": "Quiet GPU", "status": "known tenant", "evidence": f"{cd['runs_with_other_gpu_process_at_start']} runs had one 234 MiB display process resident; X0b found the same idle floor without it"},
        {"need": "Throttling", "status": "power cap", "evidence": f"{cd['runs_throttled']} runs hit a clock limit; every sample says software power cap, none thermal"},
    ]


def profiler_rows(r2) -> list[dict]:
    out = []
    for key, v in r2["C1 profiler"]["results"].items():
        wl, mode = key.split("/")
        on = v[mode]
        out.append({"case": f"{'LLM decode' if wl == 'llamacpp.phase' else 'Mask call'}, {mode}", "metric": v["metric"],
                    "off": r3(med(v["none"])), "off_min": r3(rng(v["none"], "min")), "off_max": r3(rng(v["none"], "max")),
                    "on": r3(med(on)), "on_min": r3(rng(on, "min")), "on_max": r3(rng(on, "max")),
                    "n": on["n"], "steady_ratio": r3(v["steady_ratio"]), "wall_ratio": r3(v["wall_ratio"]),
                    # a slowdown, whatever the metric's direction: < 0 = the profiler made the work slower
                    "steady_change": r3((v["steady_ratio"] - 1) if wl == "llamacpp.phase" else (1 / v["steady_ratio"] - 1))})
    return out


def warmcold_rows(r2) -> list[dict]:
    out = []
    for name, d in r2["C7 warm-cold"]["results"].items():
        task = name.split(" ")[0] if not name.startswith("node run") else "node run"
        jit = "shared" if "jit shared" in name else "none"
        page = "cold" if "page cold" in name else ("warm" if "page warm" in name else "–")
        first = d.get("first_call_s") or d.get("first_job_s")
        later = d.get("later_call_s") or d.get("later_job_s")
        out.append({"condition": name, "task": task, "page_cache": page, "jit_cache": jit,
                    "imports_s": r3(med(d.get("started_s"))), "framework_s": r3(med(d.get("framework_s"))),
                    "weights_s": r3(med(d.get("weights_s") or d.get("load_s"))),
                    "first_result_s": r3(med(first)), "first_min": r3(rng(first, "min")), "first_max": r3(rng(first, "max")),
                    "later_s": r3(med(later)), "wall_s": r3(med(d.get("wall_s"))),
                    "wall_min": r3(rng(d.get("wall_s"), "min")), "wall_max": r3(rng(d.get("wall_s"), "max")),
                    "n": (d.get("wall_s") or {}).get("n")})
    return out


def input_rows(r2) -> list[dict]:
    out = []
    for task, sizes in r2["C5 inputs"]["results"].items():
        for size, d in sizes.items():
            s = json.loads(size)
            label = (f"{s[1]} frame{'s' if s[1] > 1 else ''} at {s[0]} px" if isinstance(s, list)
                     else (f"{s} s audio" if task in ("whisper", "separate") else f"{s} images"))
            out.append({"node": task, "input": label, "load_s": r3(med(d["load_s"])), "first_s": r3(med(d["first_call_s"])),
                        "later_s": r3(med(d["warm_call_s"])), "later_min": r3(rng(d["warm_call_s"], "min")),
                        "later_max": r3(rng(d["warm_call_s"], "max")), "vram_gib": r3(med(d["vram_gib"]), 2),
                        "n": (d["warm_call_s"] or {}).get("n")})
    return out


def interference_rows(r2) -> list[dict]:
    c3 = r2["C3 stressors"]
    names = {"bw": "Bandwidth stressor", "compute": "Compute stressor", "launch": "Launch stressor",
             "training": "ResNet-50 training"}
    out = []
    for k in ("launch", "bw", "compute", "training"):
        v = c3["results"].get(k)
        if not v:
            continue
        out.append({"partner": names[k], "llm_keeps": r3(med(v["llm_ratio"])), "llm_min": r3(rng(v["llm_ratio"], "min")),
                    "llm_max": r3(rng(v["llm_ratio"], "max")), "partner_keeps": r3(med(v["partner_ratio"])),
                    "partner_min": r3(rng(v["partner_ratio"], "min")), "partner_max": r3(rng(v["partner_ratio"], "max")),
                    "n": (v["llm_ratio"] or {}).get("n")})
    return out


def shape_rows(conn) -> list[dict]:
    """Solo resource profile of the one-resource stressors and solo LLM decode (C3 solos): which ceiling each hits."""
    conn.row_factory = sqlite3.Row
    rs = conn.execute("SELECT r.*, s.json AS sj FROM run r JOIN summary s ON s.run_id=r.id WHERE r.status='done'"
                      " AND r.failure_class='success' AND r.corun_group IS NULL"
                      " AND json_extract(r.spec_json,'$.provenance.experiment')='C3 stressors'").fetchall()
    groups = {}
    for r in rs:
        p = json.loads(r["spec_json"])["params"]
        name = {"bw": "Streaming memory (copy)", "compute": "Dense matmul (fp16)", "launch": "Tiny kernels"}.get(
            p.get("mode"), {"llamacpp.phase": "LLM decode, 3B", "train.small": "ResNet-50 training"}.get(r["workload"]))
        s, e = CA.marks_abs(r, "compute_started"), CA.marks_abs(r, "compute_finished")
        t0, t1 = (s[0], e[-1]) if s and e else (r["container_started"] + 4, r["workload_finished"])
        res = CA.resources(conn, [r], t0, t1)
        o = json.loads(r["outcome_json"] or "{}")
        rate, unit = (o.get("rate"), f"{o.get('unit')}/s") if r["workload"] == "stress.gpu" else (
            (o.get("avg_ts"), "tok/s") if r["workload"] == "llamacpp.phase" else (o.get("imgs_per_s"), "images/s"))
        if res:
            groups.setdefault(name, []).append({**res, "rate": rate, "unit": unit})
    out = []
    for name, v in groups.items():
        m = lambda k: r3(st.median([x[k] for x in v if x.get(k) is not None]))  # noqa: E731
        out.append({"work": name, "rate": m("rate"), "unit": v[0]["unit"], "power_w": m("power_w"), "power": m("power"),
                    "power_capped": m("power_capped_share"), "memctl_busy": m("memctl_busy"), "sm_busy": m("sm_busy"),
                    "sm_clock": m("sm_clock"), "cpu_cores": m("cpu_cores"), "n": len(v)})
    return sorted(out, key=lambda x: -(x["power"] or 0))


def sharing_rows(r2) -> list[dict]:
    out = []
    for arm, d in r2["C2 sharing"]["results"].items():
        out.append({"arm": arm, "agg_tok_s": r3(med(d.get("agg_tok_s")), 1), "agg_min": r3(rng(d.get("agg_tok_s"), "min"), 1),
                    "agg_max": r3(rng(d.get("agg_tok_s"), "max"), 1), "tail_s": r3(med(d.get("tail_s"))),
                    "vram_gib": r3(med(d.get("vram_gib")), 2), "failed": d.get("failed", 0),
                    "n": (d.get("agg_tok_s") or {}).get("n")})
    return out


def media_rows(r2) -> tuple[list[dict], list[dict]]:
    c6 = r2["C6 media"]
    stages = []
    for k, d in c6["results"].items():
        mode, dec, codec = k.split("/")
        what = {"decode": f"decode on {'NVDEC' if dec == 'cuvid' else 'CPU'}", "source": "synthetic frames only",
                "encode": f"encode with {'NVENC' if codec == 'h264_nvenc' else 'x264'}"}.get(
            mode, f"transcode: {'NVDEC' if dec == 'cuvid' else 'CPU'} decode, {'NVENC' if codec == 'h264_nvenc' else 'x264'}")
        stages.append({"case": what, "stage": mode, "decode": dec, "encoder": codec, "wall_s": r3(med(d["wall_s"])),
                       "cpu_s": r3(med(d["cpu_s"]), 1), "nvdec_pct": r3(med(d["dec_util"]), 1),
                       "nvenc_pct": r3(med(d["enc_util"]), 1), "backend_ok": d["backend_ok"], "n": (d["wall_s"] or {}).get("n")})
    par = [{"decode": dec, "per_copy_wall_s": r3(med(v["per_copy_wall_s"])), "slowdown_vs_solo": r3(v["slowdown_vs_solo"]),
            "cpu_s_all4": r3(med(v["cpu_s_total"]), 1), "failed": v["failed"]} for dec, v in c6["parallel"].items()]
    return stages, par


def resident_rows(r2) -> list[dict]:
    names = {"whisper": "whisper turbo (ctranslate2)", "mask": "Grounding-DINO + SAM 2.1", "separate": "demucs htdemucs",
             "embed": "SigLIP2 so400m"}
    return [{"model": names.get(t, t), "vram_gib": r3(med(v), 2), "vram_min": r3(rng(v, "min"), 2), "vram_max": r3(rng(v, "max"), 2),
             "n": (v or {}).get("n")} for t, v in r2["C0 resident"]["results"].items() if v]


def capacity_rows(conn) -> tuple[list[dict], list[dict], list[dict]]:
    summ = CA.summarize(CA.rungs(conn))
    rungs, lad = [], []
    for name, d in summ.items():
        for s in d["steps"]:
            rungs.append({"ladder": name, "level": s["level"], "rate": r3(s["rate"], 2), "rate_min": r3(s["rate_min"], 2),
                          "rate_max": r3(s["rate_max"], 2), "unit": s["unit"], "n": s["n_ok"],
                          "failed": ",".join(s["failed"]) or None,
                          **{k: r3(s["res"].get(k)) for k in (*CA.RESOURCES, "power_capped_share")}})
        lad.append({"ladder": name, "unit": d["steps"][0]["unit"] if d["steps"] else None, "peak_rate": r3(d["peak_rate"], 2),
                    "peak_level": d["peak_level"], "speedup": r3(d["speedup_peak_vs_first"], 2),
                    "plateau_level": d["plateau_level"], "limiter": d["limiter"], "wall_at": d["oom_or_failure_at"]})
    pk = CA.packing(conn)
    pack = [{"mix": k, "gain": r3(v["gain"]), "gain_min": r3(v["gain_min"]), "gain_max": r3(v["gain_max"]), "n": v["n"],
             "shares": ", ".join(f"{m} {x:.2f}" for m, x in v["member_share"].items()), "packs_well": v["packs_well"]}
            for k, v in sorted(pk["mixes"].items(), key=lambda kv: -kv[1]["gain"])]
    return rungs, lad, pack


def findings(tables: dict) -> list[dict]:
    """Headline figures, each computed from the tables above (the page cites the dataset)."""
    out = []
    prof = {r["case"]: r for r in tables["profiler"]}
    t = prof.get("LLM decode, trace")
    if t:
        out.append({"id": "profiler", "figure": r3(-t["steady_change"] * 100, 1), "unit": "% slower",
                    "label": "LLM decode under the Nsight trace", "detail": f"and {t['wall_ratio']:.1f}x the wall time; rates now come from profiler-off runs only"})
    wc = {r["condition"]: r for r in tables["warmcold"]}
    a, b = wc.get("node run speech_subs (jit none)"), wc.get("node run speech_subs (jit shared)")
    if a and b:
        out.append({"id": "jit", "figure": r3(a["wall_s"] - b["wall_s"], 1), "unit": "s",
                    "label": "of a cold whisper call is CUDA JIT", "detail": f"node run of a 40 s clip: {a['wall_s']:.1f} s fresh vs {b['wall_s']:.1f} s with a shared JIT cache"})
    w = wc.get("worker (page warm, jit shared)")
    if a and w and w.get("wall_s"):
        out.append({"id": "worker", "figure": r3(6 * a["wall_s"] / w["wall_s"], 1), "unit": "x",
                    "label": "faster: 6 jobs on one warm worker", "detail": f"{w['wall_s']:.1f} s vs 6 cold node runs at {a['wall_s']:.1f} s each"})
    sh = {r["work"]: r for r in tables["shapes"]}
    c, m = sh.get("Dense matmul (fp16)"), sh.get("Streaming memory (copy)")
    if c and m:
        out.append({"id": "power", "figure": r3(c["power_w"], 0), "unit": "W",
                    "label": "dense matmul sits on the power limit", "detail": f"capped {c['power_capped']:.0%} of the time; streaming memory runs at {m['power_w']:.0f} W with the memory controller {m['memctl_busy']:.0%} busy"})
    it = {r["partner"]: r for r in tables["interference"]}
    tr = it.get("ResNet-50 training")
    if tr:
        out.append({"id": "interference", "figure": r3(tr["llm_keeps"], 2), "unit": "x",
                    "label": "LLM decode speed kept next to training", "detail": f"training keeps {tr['partner_keeps']:.2f}x of its own; next to a bandwidth stressor decode keeps {it.get('Bandwidth stressor', {}).get('llm_keeps', float('nan')):.2f}x"})
    sh2 = {r["arm"]: r for r in tables["sharing"]}
    b, c, mp, one = (sh2.get(k) for k in ("batched x4", "copies x4 (timeslice)", "copies x4 (mps)", "single (timeslice)"))
    if b and c and mp and one:
        out.append({"id": "batching", "figure": r3(b["agg_tok_s"] / c["agg_tok_s"], 2), "unit": "x",
                    "label": "more decode throughput: 4 requests batched vs 4 processes",
                    "detail": f"{b['agg_tok_s']:.0f} vs {c['agg_tok_s']:.0f} tok/s (one request alone: {one['agg_tok_s']:.0f}); the 4 processes take {c['vram_gib'] / b['vram_gib']:.1f}x the VRAM"})
        gap = (mp["agg_tok_s"] - c["agg_tok_s"]) / (b["agg_tok_s"] - c["agg_tok_s"])
        out.append({"id": "mps", "figure": r3(gap * 100, 0), "unit": "%",
                    "label": "of the copies-to-batching gap recovered by MPS",
                    "detail": f"MPS lifts 4 processes from {c['agg_tok_s']:.0f} to {mp['agg_tok_s']:.0f} tok/s: context switching is part of the loss, missing batching is most of it"})
    mpar = {r["decode"]: r for r in tables["media_parallel"]}
    if mpar.get("cpu") and mpar.get("cuvid"):
        out.append({"id": "nvdec", "figure": r3(mpar["cpu"]["cpu_s_all4"] / mpar["cuvid"]["cpu_s_all4"], 1), "unit": "x",
                    "label": "less CPU for 4 parallel transcodes with NVDEC",
                    "detail": f"{mpar['cpu']['cpu_s_all4']:.0f} vs {mpar['cuvid']['cpu_s_all4']:.0f} CPU-s at about the same speed per copy ({mpar['cpu']['per_copy_wall_s']:.1f} vs {mpar['cuvid']['per_copy_wall_s']:.1f} s)"})
    if tables["packing"]:
        best = tables["packing"][0]
        out.append({"id": "packing", "figure": best["gain"], "unit": "x", "label": f"useful work when packing {best['mix']}",
                    "detail": f"sum of each task's share of its solo speed ({best['shares']})"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(paths.DATA / "gauge.sqlite"))
    ap.add_argument("--out", default=str(paths.DATA / "dashboard"))
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    r2 = {name: fn(conn) for name, fn in R2.CONTRASTS.items()}
    stages, par = media_rows(r2)
    rungs, lad, pack = capacity_rows(conn)
    tables = {"progress": progress(conn), "audit": audit_rows(conn), "profiler": profiler_rows(r2),
              "warmcold": warmcold_rows(r2), "inputs": input_rows(r2), "interference": interference_rows(r2),
              "shapes": shape_rows(conn), "sharing": sharing_rows(r2), "media": stages, "media_parallel": par,
              "capacity": rungs, "ladders": lad, "packing": pack, "resident": resident_rows(r2)}
    tables["findings"] = findings(tables)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=1, default=str))
        print(f"{name:15s} {len(rows):4d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
