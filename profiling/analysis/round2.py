"""Round 2 analysis: applies each contrast's pre-registered rule (profiling/sweeps/x/c*.yaml) to the runs labelled
with its experiment. Every number is reported with n, median and min-max; verdicts compare against the tolerance
written in the file before the runs. Prints markdown; --json writes the same numbers for the presentation.

    uv run python profiling/analysis/round2.py [--db data/gauge/gauge.sqlite] [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics as st
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gauge import paths  # noqa: E402

SWEEPS = ROOT / "profiling" / "sweeps" / "x"
GIB = 2**30


# --- loading -----------------------------------------------------------------------------------------------------

class Run(dict):
    """A done run: spec, summary, outcome and its responses, flattened for the rules below."""

    @property
    def p(self) -> dict:
        return self["spec"]["params"]

    def responses(self) -> list[dict]:
        o = self["outcome"]
        return o.get("responses") or ([o] if o else [])

    def phase(self, k):
        return (self["summary"].get("phase_s") or {}).get(k)

    def mark(self, name):
        return next((m["t"] for m in self["summary"].get("marks", []) if m["mark"] == name), None)


def load(conn, experiment: str) -> list[Run]:
    conn.row_factory = sqlite3.Row
    out = []
    for r in conn.execute("SELECT r.*, s.json AS sj FROM run r JOIN summary s ON s.run_id=r.id WHERE r.status='done'"
                          " AND json_extract(r.spec_json, '$.provenance.experiment') = ? ORDER BY r.id",
                          (experiment,)):
        out.append(Run(id=r["id"], workload=r["workload"], spec=json.loads(r["spec_json"]), group=r["corun_group"],
                       ok=r["failure_class"] == "success", fclass=r["failure_class"],
                       summary=json.loads(r["sj"] or "{}"), outcome=json.loads(r["outcome_json"] or "{}"),
                       start=r["container_started"], end=r["workload_finished"]))
    return out


# --- statistics ----------------------------------------------------------------------------------------------------

def stat(xs) -> dict | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "median": st.median(xs), "min": min(xs), "max": max(xs)}


def fmt(s: dict | None, unit: str = "", scale: float = 1, digits: int = 2) -> str:
    if not s:
        return "–"
    f = lambda v: f"{v * scale:.{digits}f}"  # noqa: E731
    return f"{f(s['median'])}{unit} [{f(s['min'])}–{f(s['max'])}] n={s['n']}"


def ratio(a: dict | None, b: dict | None) -> float | None:
    return a["median"] / b["median"] if a and b and b["median"] else None


def meta(name: str) -> dict:
    d = yaml.safe_load((SWEEPS / name).read_text())
    return {k: d.get(k) for k in ("question", "measure", "analysis", "tolerance", "stop", "decision")}


def table(headers, rows) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def avg_ts(r: Run) -> float | None:
    v = [x["avg_ts"] for x in r.responses() if "avg_ts" in x]
    return st.mean(v) if v else None


def warm_spans(r: Run) -> float | None:
    s = r.phase("spans") or []
    return st.median(s[1:]) if len(s) > 1 else None


# --- contrasts ---------------------------------------------------------------------------------------------------

def c1(conn) -> dict:
    runs = [r for r in load(conn, "C1 profiler") if r["ok"]]
    out, rows = {}, []
    for wl, metric, name in (("llamacpp.phase", avg_ts, "decode tok/s"), ("wonhee.gpu", warm_spans, "warm mask call s")):
        by = {m: [r for r in runs if r["workload"] == wl and r["spec"].get("profile", "none") == m]
              for m in ("none", "trace", "metrics")}
        base, base_wall = stat(metric(r) for r in by["none"]), stat(r["summary"].get("runtime_s") for r in by["none"])
        for m in ("trace", "metrics"):
            s, wall = stat(metric(r) for r in by[m]), stat(r["summary"].get("runtime_s") for r in by[m])
            if not s:
                continue
            q = ratio(s, base)
            within = q is not None and abs(q - 1) <= 0.03
            out[f"{wl}/{m}"] = {"metric": name, "none": base, m: s, "steady_ratio": q, "wall_ratio": ratio(wall, base_wall),
                                "within_3pct": within}
            rows.append([wl, name, m, fmt(base), fmt(s), f"{q:.3f}" if q else "–",
                         f"{ratio(wall, base_wall):.2f}" if ratio(wall, base_wall) else "–", "yes" if within else "**no**"])
    md = table(["workload", "metric", "mode", "profiler off", "profiler on", "steady ratio", "wall ratio", "within ±3%"], rows)
    return {"meta": meta("c1_profiler.yaml"), "results": out, "md": md}


def _arm(r: Run, members: list[Run]) -> str:
    n = len(members)
    par = str(r.p.get("parallel"))
    mode = r["spec"].get("sharing", "timeslice")
    if n == 1:
        return "batched x4" if par == "4" else f"single ({mode})"
    return f"copies x{n} ({mode})"


def c2(conn) -> dict:
    runs = load(conn, "C2 sharing")
    groups = collections.defaultdict(list)
    for r in runs:
        groups[r["group"] or f"solo{r['id']}"].append(r)
    arms = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in groups.values():
        if not all(r["ok"] for r in g):
            arms[_arm(g[0], g)]["failed"].append(1)
            continue
        resp = [x for r in g for x in r.responses() if "speed_tg" in x]
        arm = _arm(g[0], g)
        arms[arm]["agg_tok_s"].append(sum(x["speed_tg"] for x in resp))
        arms[arm]["tail_s"].append(max(x["t_tg"] for x in resp))
        arms[arm]["vram_gib"].append(sum((r["summary"].get("vram_peak_bytes") or 0) for r in g) / GIB)
    order = ["single (timeslice)", "single (mps)", "batched x4", "copies x4 (timeslice)", "copies x4 (mps)"]
    res = {a: {k: stat(v) for k, v in arms[a].items() if k != "failed"} | {"failed": len(arms[a]["failed"])}
           for a in order if a in arms}
    rows = [[a, fmt(r.get("agg_tok_s"), digits=0), fmt(r.get("tail_s")), fmt(r.get("vram_gib")), r["failed"]]
            for a, r in res.items()]
    verdict = None
    try:
        cp, mp, bt = (res[a]["agg_tok_s"]["median"] for a in ("copies x4 (timeslice)", "copies x4 (mps)", "batched x4"))
        verdict = {"mps_recovers_fraction_of_gap": (mp - cp) / (bt - cp) if bt != cp else None}
        verdict["context_switching_main_loss"] = verdict["mps_recovers_fraction_of_gap"] is not None and \
            verdict["mps_recovers_fraction_of_gap"] > 0.5
    except (KeyError, TypeError):
        pass
    md = table(["arm (4 requests x 2048 tokens)", "aggregate decode tok/s", "slowest request s", "VRAM GiB (sum)", "failed"], rows)
    return {"meta": meta("c2_sharing.yaml"), "results": res, "verdict": verdict, "md": md}


def _rate_in(r: Run, t0: float, t1: float) -> float | None:
    s = [x for resp in r.responses() for x in resp.get("series", [])]
    inside = [x for x in s if t0 <= x[0] <= t1]
    if len(inside) < 2:
        return None
    return (inside[-1][1] - inside[0][1]) / (inside[-1][0] - inside[0][0])


def c3(conn) -> dict:
    runs = [r for r in load(conn, "C3 stressors") if r["ok"]]
    solo = [r for r in runs if not r["group"]]
    llm_solo = stat(avg_ts(r) for r in solo if r["workload"] == "llamacpp.phase")
    stress_solo = {m: stat(r.responses()[0].get("rate") for r in solo if r["workload"] == "stress.gpu" and r.p["mode"] == m)
                   for m in ("bw", "compute", "launch")}
    train_solo = stat(r["outcome"].get("imgs_per_s") for r in solo if r["workload"] == "train.small")
    groups = collections.defaultdict(list)
    for r in runs:
        if r["group"]:
            groups[r["group"]].append(r)
    pairs = collections.defaultdict(lambda: {"llm": [], "partner": [], "overlap": []})
    for g in groups.values():
        llm = next((r for r in g if r["workload"] == "llamacpp.phase"), None)
        other = next((r for r in g if r is not llm), None)
        if not llm or not other:
            continue
        key = other.p.get("mode", "training") if other["workload"] == "stress.gpu" else "training"
        pairs[key]["llm"].append(avg_ts(llm) / llm_solo["median"] if avg_ts(llm) and llm_solo else None)
        if other["workload"] == "stress.gpu" and stress_solo.get(key):
            rate = _rate_in(other, llm["start"], llm["end"])
            pairs[key]["partner"].append(rate / stress_solo[key]["median"] if rate else None)
        elif train_solo and other["outcome"].get("imgs_per_s"):  # whole-run rate: diluted by time it ran alone
            pairs[key]["partner"].append(other["outcome"]["imgs_per_s"] / train_solo["median"])
        pairs[key]["overlap"].append(min(other["end"], llm["end"]) - max(other["start"], llm["start"]))
    res = {k: {"llm_ratio": stat(v["llm"]), "partner_ratio": stat(v["partner"])} for k, v in pairs.items()}
    t = res.get("training", {}).get("llm_ratio")
    rows = []
    for k in ("bw", "compute", "launch", "training"):
        if k in res:
            lr = res[k]["llm_ratio"]
            close = t and lr and abs(lr["median"] - t["median"]) <= 0.1 and k != "training"
            res[k]["explains_training_loss"] = bool(close)
            rows.append([k, fmt(lr, digits=2), fmt(res[k]["partner_ratio"], digits=2),
                         "yes" if close else ("(reference)" if k == "training" else "no")])
    md = (f"LLM decode solo: {fmt(llm_solo, ' tok/s', digits=0)}\n\n"
          + table(["partner", "LLM tok/s ratio", "partner rate ratio (in LLM window)", "within 0.1 of training"], rows))
    return {"meta": meta("c3_stressors.yaml"), "solo": {"llm": llm_solo, "stress": stress_solo, "train": train_solo},
            "results": res, "md": md}


def c5(conn) -> dict:
    runs = [r for r in load(conn, "C5 inputs") if r["ok"] and r["spec"].get("profile", "none") == "none"]
    size = {"mask": lambda p: (p["px"], p["images"]), "separate": lambda p: p["clip_s"], "whisper": lambda p: p["clip_s"],
            "embed": lambda p: p["images"]}
    cells = collections.defaultdict(list)
    for r in runs:
        cells[(r.p["task"], json.dumps(size[r.p["task"]](r.p)))].append(r)
    rows, res = [], {}
    for (task, sz), rs in sorted(cells.items()):
        d = {"load_s": stat(r.phase("load") for r in rs), "warm_call_s": stat(warm_spans(r) for r in rs),
             "first_call_s": stat((r.phase("spans") or [None])[0] for r in rs),
             "vram_gib": stat((r["summary"].get("vram_peak_bytes") or 0) / GIB for r in rs)}
        res.setdefault(task, {})[sz] = d
        rows.append([task, sz, fmt(d["load_s"]), fmt(d["first_call_s"]), fmt(d["warm_call_s"]), fmt(d["vram_gib"])])
    verdicts = {}
    for task, d in res.items():
        ks = sorted(d, key=lambda k: json.loads(k) if not isinstance(json.loads(k), list) else json.loads(k)[0] * json.loads(k)[1])
        if len(ks) >= 2:
            lo, hi = d[ks[0]], d[ks[-1]]
            q = ratio(hi["warm_call_s"], lo["warm_call_s"])
            verdicts[task] = {"small": ks[0], "large": ks[-1], "warm_call_ratio": q, "vram_ratio": ratio(hi["vram_gib"], lo["vram_gib"]),
                              "size_is_required_descriptor": q is not None and q >= 1.5}
    md = table(["node", "input", "load s", "first call s", "later calls s", "VRAM GiB"], rows)
    return {"meta": meta("c5_inputs.yaml"), "results": res, "verdicts": verdicts, "md": md}


def c6(conn) -> dict:
    runs = load(conn, "C6 media")
    solo = [r for r in runs if r["ok"] and not r["group"]]
    cells = collections.defaultdict(list)
    for r in solo:
        p = r.p
        cells[(p.get("mode", "transcode"), p.get("decode", "cpu") if p.get("mode", "transcode") in ("transcode", "decode") else "-",
               p.get("codec") if p.get("mode", "transcode") in ("transcode", "encode") else "-")].append(r)
    rows, res = [], {}
    for k, rs in sorted(cells.items()):
        s = lambda f: stat(f(r) for r in rs)  # noqa: E731
        d = {"wall_s": s(lambda r: r["summary"].get("runtime_s")), "cpu_s": s(lambda r: r["summary"].get("cpu_time_s")),
             "dec_util": s(lambda r: r["summary"].get("dev_dec_util_mean")), "enc_util": s(lambda r: r["summary"].get("dev_enc_util_mean"))}
        mode, dec, codec = k
        want_dec, want_enc = dec == "cuvid", codec == "h264_nvenc"
        got_dec = (d["dec_util"] or {}).get("median", 0) > 0
        got_enc = (d["enc_util"] or {}).get("median", 0) > 0
        d["backend_ok"] = got_dec == want_dec and got_enc == want_enc
        res["/".join(k)] = d
        rows.append([mode, dec, codec, fmt(d["wall_s"]), fmt(d["cpu_s"], digits=1), fmt(d["dec_util"], digits=1),
                     fmt(d["enc_util"], digits=1), "ok" if d["backend_ok"] else "**check**"])
    groups = collections.defaultdict(list)
    for r in runs:
        if r["group"]:
            groups[r["group"]].append(r)
    par = collections.defaultdict(lambda: {"per_copy_wall": [], "cpu_s_total": [], "failed": 0})
    for g in groups.values():
        dec = g[0].p.get("decode", "cpu")
        if not all(r["ok"] for r in g):
            par[dec]["failed"] += 1
            continue
        par[dec]["per_copy_wall"].append(st.median(r["summary"]["runtime_s"] for r in g))
        par[dec]["cpu_s_total"].append(sum(r["summary"].get("cpu_time_s") or 0 for r in g))
    pres = {d: {"per_copy_wall_s": stat(v["per_copy_wall"]), "cpu_s_total": stat(v["cpu_s_total"]), "failed": v["failed"]}
            for d, v in par.items()}
    for d, v in pres.items():
        solo_wall = res.get(f"transcode/{d}/h264_nvenc", {}).get("wall_s")
        v["slowdown_vs_solo"] = ratio(v["per_copy_wall_s"], solo_wall)
    md = (table(["stage", "decode", "encoder", "wall s", "CPU s", "NVDEC %", "NVENC %", "backend"], rows) + "\n\n"
          + table(["4 parallel NVENC transcodes", "per-copy wall s", "slowdown vs solo", "CPU s (all 4)", "failed"],
                  [[d, fmt(v["per_copy_wall_s"]), f"{v['slowdown_vs_solo']:.2f}" if v["slowdown_vs_solo"] else "–",
                    fmt(v["cpu_s_total"], digits=1), v["failed"]] for d, v in sorted(pres.items())]))
    return {"meta": meta("c6_media.yaml"), "results": res, "parallel": pres, "md": md}


def c7(conn) -> dict:
    runs = [r for r in load(conn, "C7 warm-cold") if r["ok"]]
    cells = collections.defaultdict(list)
    for r in runs:
        name = "node run speech_subs (cold path)" if r["workload"] == "wonhee.speech_subs" else f"{r.p['task']} ({r.p.get('cache', 'as_is')})"
        cells[name].append(r)
    rows, res = [], {}
    for name, rs in sorted(cells.items()):
        if rs[0]["workload"] == "wonhee.speech_subs":
            d = {"wall_s": stat(r["summary"].get("runtime_s") for r in rs)}
            rows.append([name, "–", "–", "–", "–", fmt(d["wall_s"])])
        elif rs[0].p["task"] == "worker":
            d = {"started_s": stat(r.mark("started") for r in rs), "load_s": stat(r.phase("load") for r in rs),
                 "first_job_s": stat((r["outcome"].get("job_s") or [None])[0] for r in rs),
                 "later_job_s": stat(st.median(r["outcome"]["job_s"][1:]) for r in rs if len(r["outcome"].get("job_s") or []) > 1),
                 "wall_s": stat(r["summary"].get("runtime_s") for r in rs)}
            rows.append([name, fmt(d["started_s"]), fmt(d["load_s"]), fmt(d["first_job_s"]), fmt(d["later_job_s"]), fmt(d["wall_s"])])
        else:
            d = {"started_s": stat(r.mark("started") for r in rs), "load_s": stat(r.phase("load") for r in rs),
                 "first_call_s": stat((r.phase("spans") or [None])[0] for r in rs),
                 "later_call_s": stat(warm_spans(r) for r in rs), "wall_s": stat(r["summary"].get("runtime_s") for r in rs)}
            rows.append([name, fmt(d["started_s"]), fmt(d["load_s"]), fmt(d["first_call_s"]), fmt(d["later_call_s"]), fmt(d["wall_s"])])
        res[name] = d
    md = table(["condition (40 s clip / 4 images)", "interpreter+imports s", "model load s", "first result s", "later results s", "wall s"], rows)
    return {"meta": meta("c7_warm_cold.yaml"), "results": res, "md": md}


def c0(conn) -> dict:
    runs = [r for r in load(conn, "C0 resident") if r["ok"]]
    res = {t: stat((r["summary"].get("vram_peak_bytes") or 0) / GIB for r in runs if r.p["task"] == t)
           for t in ("whisper", "mask", "separate", "embed")}
    md = table(["model held resident", "VRAM GiB"], [[t, fmt(s)] for t, s in res.items()])
    return {"meta": meta("c0_resident.yaml"), "results": res, "md": md}


CONTRASTS = {"C1 profiler": c1, "C2 sharing": c2, "C3 stressors": c3, "C5 inputs": c5, "C6 media": c6,
             "C7 warm-cold": c7, "C0 resident": c0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(paths.DATA / "gauge.sqlite"))
    ap.add_argument("--json")
    ap.add_argument("--only", help="one contrast, e.g. 'C1 profiler'")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    out = {}
    for name, fn in CONTRASTS.items():
        if a.only and a.only != name:
            continue
        res = fn(conn)
        out[name] = res
        m = res["meta"]
        print(f"## {name}\n\n**Question:** {m['question']}  \n**Rule:** {m['analysis']}  \n"
              f"**Tolerance:** {m.get('tolerance')}\n\n{res['md']}\n")
        for k in ("verdict", "verdicts"):
            if res.get(k):
                print(f"**Verdict:** `{json.dumps(res[k], default=str)}`\n")
    if a.json:
        Path(a.json).write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "md"} for k, v in out.items()},
                                           indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
