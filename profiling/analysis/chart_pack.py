"""Turn the dashboard tables into a chart pack another agent can rebuild as native slide charts, without screenshots.

Every chart is a spec (kind, axes with ticks, series, reference lines, takeaway, source table) plus its plotted
points as tidy CSV: one row per mark, with the data value and the mark's position in the plot area (`pos_x`,
`pos_y` from 0 to 1, origin bottom-left). Mapping onto a slide box is then `left + pos_x*width`,
`top + (1 - pos_y)*height`. Writes chart-pack.md and chart-pack.json; chart-pack.html renders the same JSON.

    uv run python profiling/analysis/chart_pack.py [--out data/gauge/chart-pack]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gauge import paths  # noqa: E402

DASH = paths.DATA / "dashboard"
HTML = ROOT / "profiling" / "report" / "chart-pack.html"
COLS = ["x", "series", "y", "y_min", "y_max", "y_base", "label", "pos_x", "pos_y", "pos_y0", "pos_w"]


def load(name: str):
    f = DASH / name
    return json.loads(f.read_text()) if f.suffix == ".json" else list(csv.DictReader(f.open()))


def num(v):
    if v in (None, ""):
        return None
    return float(v)


def r4(v):
    """Four significant digits; positions and values both stay readable in CSV."""
    if v is None or isinstance(v, str):
        return v
    return float(f"{v:.4g}")


# ---------- scales ----------

def nice(lo: float, hi: float, n: int = 5) -> tuple[float, float, list[float]]:
    """Round the domain out to a 1/2/5 step so tick labels are values a reader would choose."""
    if hi == lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    a, b = math.floor(lo / step) * step, math.ceil(hi / step) * step
    ticks = [round(a + i * step, 10) for i in range(int(round((b - a) / step)) + 1)]
    return a, b, ticks


class Axis:
    def __init__(self, label: str, unit: str = "", scale: str = "linear", lo=None, hi=None, ticks=None):
        self.label, self.unit, self.scale = label, unit, scale
        self.lo, self.hi, self.ticks, self.cats = lo, hi, ticks, None

    def fit(self, values, zero: bool = True):
        vs = [v for v in values if v is not None]
        if self.scale == "category":
            self.cats = list(dict.fromkeys(vs))
            return self
        if self.scale.startswith("log"):
            base = 2 if self.scale == "log2" else 10
            lo = base ** math.floor(math.log(min(vs), base) + 1e-9)
            hi = base ** math.ceil(math.log(max(vs), base) - 1e-9)
            self.lo, self.hi = self.lo or lo, self.hi or hi
            k = math.log(self.hi / self.lo, base)
            stride = max(1, math.ceil(k / 8))
            self.ticks = self.ticks or [self.lo * base ** i for i in range(0, int(round(k)) + 1, stride)]
            return self
        lo = min(vs + ([0] if zero else []))
        hi = max(vs + ([0] if zero else []))
        a, b, t = nice(self.lo if self.lo is not None else lo, self.hi if self.hi is not None else hi)
        self.lo, self.hi, self.ticks = a, b, self.ticks or t
        return self

    def pos(self, v):
        if v is None:
            return None
        if self.scale.startswith("log"):
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def spec(self) -> dict:
        d = {"label": self.label, "unit": self.unit, "scale": self.scale}
        if self.cats is not None:
            n = len(self.cats)
            d["categories"] = [{"value": c, "pos": r4((i + 0.5) / n)} for i, c in enumerate(self.cats)]
        else:
            d["domain"] = [r4(self.lo), r4(self.hi)]
            d["ticks"] = [{"value": r4(t), "pos": r4(self.pos(t))} for t in self.ticks]
        return d


# ---------- chart assembly ----------

CHARTS: list[dict] = []


def add(cid, section, title, takeaway, kind, rows, x: Axis | None = None, y: Axis | None = None,
        refs=(), events=(), bands=(), notes="", source="", columns=None):
    """Place every row on the plot (pos_*), then record the chart. Rows use the COLS keys."""
    series = list(dict.fromkeys(r.get("series", "") for r in rows))
    if kind in ("bar", "stacked_bar"):
        x.fit([r["x"] for r in rows])
        if kind == "stacked_bar":
            base = defaultdict(float)
            for r in rows:
                r["y_base"] = base[r["x"]]
                base[r["x"]] += r["y"] or 0
            y.fit([*base.values(), 0])
        else:
            y.fit([v for r in rows for v in (r["y"], r.get("y_max"))] + [f["y"] for f in refs])
        n, k = len(x.cats), (1 if kind == "stacked_bar" else len(series))
        band = 0.8 / n
        for r in rows:
            i = x.cats.index(r["x"])
            j = 0 if kind == "stacked_bar" else series.index(r.get("series", ""))
            w = band / k
            r["pos_x"] = (i + 0.1) / n + (j + 0.5) * w
            r["pos_w"] = w
            b = r.get("y_base") or 0
            r["pos_y0"] = y.pos(max(b, y.lo))
            r["pos_y"] = y.pos((r["y"] or 0) + (b if kind == "stacked_bar" else 0))
    elif kind in ("line", "scatter"):
        x.fit([r["x"] for r in rows if r["y"] is not None] + [e["x"] for e in events], zero=False)
        y.fit([v for r in rows for v in (r["y"], r.get("y_max"))] + [f["y"] for f in refs])
        for r in rows:
            r["pos_x"], r["pos_y"] = x.pos(r["x"]), y.pos(r["y"])
    elif kind == "heatmap":
        x.fit([r["x"] for r in rows])
        y.fit(series)
        for r in rows:
            r["pos_x"] = (x.cats.index(r["x"]) + 0.5) / len(x.cats)
            r["pos_y"] = 1 - (y.cats.index(r["series"]) + 0.5) / len(y.cats)
            r["pos_w"] = 1 / len(x.cats)
    spec = {
        "id": cid, "section": section, "title": title, "takeaway": takeaway, "kind": kind,
        "x": x.spec() if x else None, "y": y.spec() if y else None, "series": series if kind != "table" else None,
        "refs": [{**f, "pos": r4(y.pos(f["y"]))} for f in refs],
        "events": [{**e, "pos": r4(x.pos(e["x"]))} for e in events],
        "bands": [{**b, "pos_from": r4(x.pos(b["from"])), "pos_to": r4(x.pos(b["to"]))} for b in bands],
        "notes": notes, "source": source,
    }
    if kind == "heatmap":
        spec["color"] = {"domain": [0, 1], "meaning": "share of the ceiling, 0 = idle, 1 = at the ceiling"}
    cols = columns or [c for c in COLS if any(r.get(c) not in (None, "") for r in rows)]
    spec["columns"] = cols
    spec["rows"] = [[r4(r.get(c)) for c in cols] for r in rows]
    CHARTS.append(spec)


def bars(items, series="", **kw):
    return [{"x": x, "series": series, "y": y, **kw} for x, y in items]


# ---------- the charts ----------

RES = {"sm_busy": "SM busy", "memctl_busy": "memory controller busy", "power": "power / limit", "vram": "VRAM / 16 GiB",
       "cpu": "CPU / 16 threads", "nvenc": "NVENC busy", "nvdec": "NVDEC busy", "pcie": "PCIe / link max"}


def build():
    f = load("findings.json")
    add("headline", "Overview", "Headline numbers", "The ten figures the rest of the pack backs up.", "kpi",
        [{"x": r["id"], "y": r["figure"], "series": r["unit"], "label": f'{r["label"]}; {r["detail"]}'} for r in f],
        columns=["x", "y", "series", "label"], source="tables/findings.json",
        notes="kpi: x = id, y = figure, series = unit, label = what it means")

    p = load("progress.json")
    add("progress", "Overview", "Runs per experiment", "Every pre-registered experiment finished; C8 had 15 failed runs, "
        "all out-of-memory walls or the since-fixed NVDEC loop.", "stacked_bar",
        [*bars([(r["experiment"], r["succeeded"]) for r in p], "succeeded"),
         *bars([(r["experiment"], r["failed"]) for r in p], "failed")],
        Axis("experiment", scale="category"), Axis("runs", "runs"), source="tables/progress.json")

    pr = load("profiler.json")
    add("c1_slowdown", "C1 profiler cost", "Nsight slows steady-state work",
        "Tracing or GPU metrics cost 4.5-7.8% of steady-state speed, so every rate in this pack comes from profiler-off runs.",
        "bar", [{"x": r["case"], "y": 100 * abs(r["steady_ratio"] - 1), "label": f'{r["off"]:.4g} off vs {r["on"]:.4g} on, {r["metric"]}'}
                for r in pr], Axis("case", scale="category"), Axis("steady-state slowdown", "%"),
        source="tables/profiler.json", notes="n = 3-5 repeats per arm")
    add("c1_wall", "C1 profiler cost", "Nsight multiplies wall time",
        "Fixed start-up cost: a profiled LLM run takes 2.6-2.8x as long end to end.", "bar",
        bars([(r["case"], r["wall_ratio"]) for r in pr]), Axis("case", scale="category"), Axis("wall time, profiled / off", "x"),
        refs=[{"y": 1, "label": "no change"}], source="tables/profiler.json")

    wc = [r for r in load("warmcold.json") if r["task"] != "node run"]
    ph = [("imports_s", "imports"), ("framework_s", "framework init"), ("weights_s", "weights load"),
          ("first_result_s", "first call")]
    add("c7_phases", "C7 cold vs warm", "Where a cold GPU call spends its time",
        "With a fresh CUDA JIT cache whisper's first call takes 10.8 s; a shared JIT cache cuts it to 1.6 s. "
        "Page cache barely matters. The mask cost is framework import plus weights.", "stacked_bar",
        [{"x": r["condition"], "series": lab, "y": r[k] or 0} for r in wc for k, lab in ph],
        Axis("condition", scale="category"), Axis("seconds before the first result", "s"),
        source="tables/warmcold.json", notes="worker rows have no separate framework mark (0 here)")
    nr = {r["jit_cache"]: r["wall_s"] for r in load("warmcold.json") if r["task"] == "node run"}
    w = next(r for r in load("warmcold.json") if r["condition"] == "worker (page warm, jit shared)")
    add("c7_six_jobs", "C7 cold vs warm", "Six whisper jobs: cold runs vs one warm worker",
        "A warm worker finishes 6 jobs 7.5x faster than 6 cold node runs.", "bar",
        bars([("6 cold node runs, fresh JIT", 6 * nr["none"]), ("6 cold node runs, shared JIT", 6 * nr["shared"]),
              ("1 warm worker, 6 jobs", w["wall_s"])]),
        Axis("setup", scale="category"), Axis("total wall time", "s"), source="tables/warmcold.json",
        notes="cold node-run totals are 6 x the measured single run (14.55 s / 5.38 s)")

    inp = load("inputs.json")
    add("c5_inputs", "C5 input size", "Load vs call time per GPU node",
        "Model load (3-23 s) dwarfs a warm call (0.1-2.3 s) at every input size; call time grows with input, VRAM barely moves.",
        "bar", [{"x": f'{r["node"]}: {r["input"]}', "series": lab, "y": r[k],
                 **({"y_min": r["later_min"], "y_max": r["later_max"]} if k == "later_s" else {})}
                for r in inp for k, lab in (("load_s", "model load"), ("first_s", "first call"), ("later_s", "later calls"))],
        Axis("node and input", scale="category"), Axis("seconds", "s"), source="tables/inputs.json")
    add("c5_vram", "C5 input size", "VRAM per node and input", "VRAM is set by the model, not the input.", "bar",
        bars([(f'{r["node"]}: {r["input"]}', r["vram_gib"]) for r in inp]),
        Axis("node and input", scale="category"), Axis("peak VRAM", "GiB"), source="tables/inputs.json")

    rs = load("resident.json")
    add("c0_resident", "C0 resident VRAM", "VRAM held by each wonhee GPU model",
        f'Together {sum(r["vram_gib"] for r in rs):.2f} GiB of 16, measured one at a time (not tested co-resident).', "bar",
        bars([(r["model"], r["vram_gib"]) for r in rs]), Axis("model", scale="category"), Axis("steady VRAM", "GiB"),
        refs=[{"y": 16, "label": "card VRAM, 16 GiB"}], source="tables/resident.json")

    it = load("interference.json")
    add("c3_interference", "C3 interference", "What each side keeps when LLM decode shares the GPU",
        "LLM decode keeps 0.73 of its speed beside tiny kernels, 0.37 beside a bandwidth hog, 0.29 beside dense compute, "
        "0.13 beside training. Partners keep 0.81-0.89, except the launch stressor (0.32).", "bar",
        [*[{"x": r["partner"], "series": "LLM decode keeps", "y": r["llm_keeps"], "y_min": r["llm_min"], "y_max": r["llm_max"]} for r in it],
         *[{"x": r["partner"], "series": "partner keeps", "y": r["partner_keeps"], "y_min": r["partner_min"], "y_max": r["partner_max"]} for r in it]],
        Axis("partner", scale="category"), Axis("share of solo speed", "x", lo=0, hi=1),
        refs=[{"y": 1, "label": "solo speed"}], source="tables/interference.json")

    sh = load("shapes.json")
    keys = [("power", "power / limit"), ("power_capped", "time power-capped"), ("memctl_busy", "memory controller busy"),
            ("sm_busy", "SM busy"), ("sm_clock", "SM clock / max")]
    add("c3_shapes", "C3 interference", "Resource shape of each kind of work",
        "Dense matmul lives on the power cap; streaming memory pins the memory controller at half the power; "
        "tiny kernels use almost nothing yet still slow LLM decode by 27%.", "heatmap",
        [{"x": lab, "series": r["work"], "y": r[k], "label": f'{r["rate"]:.4g} {r["unit"]}'} for r in sh for k, lab in keys],
        Axis("resource", scale="category"), Axis("work", scale="category"), source="tables/shapes.json",
        notes="SM busy is NVML's 'any kernel running' share, not occupancy")

    sg = load("sharing.json")
    add("c2_sharing", "C2 sharing", "Four requests: batched vs separate processes vs MPS",
        "Batching gives 645 tok/s; 4 time-sliced processes give 252, below a single request (295). "
        "MPS lifts them to 364, recovering 29% of the gap.", "bar",
        [{"x": r["arm"], "y": r["agg_tok_s"], "y_min": r["agg_min"], "y_max": r["agg_max"]} for r in sg],
        Axis("arm", scale="category"), Axis("aggregate decode", "tok/s"), source="tables/sharing.json",
        notes="qwen2.5:3b, 16 prompt tokens; n = 3 per arm")
    add("c2_vram", "C2 sharing", "VRAM per sharing arm", "Separate processes hold 3.7x the VRAM of one batched process.",
        "bar", bars([(r["arm"], r["vram_gib"]) for r in sg]), Axis("arm", scale="category"), Axis("VRAM", "GiB"),
        source="tables/sharing.json")

    md = load("media.json")
    add("c6_media_cpu", "C6 media", "CPU seconds per media stage",
        "Moving decode to NVDEC drops it from 85.6 to 1.6 CPU-s; x264 transcode costs 214 CPU-s.", "bar",
        bars([(r["case"], r["cpu_s"]) for r in md]), Axis("stage and backend", scale="category"), Axis("CPU time", "CPU-s"),
        source="tables/media.json", notes="1080p30 clip; backend_ok checked from NVENC/NVDEC engine use")
    add("c6_media_wall", "C6 media", "Wall time per media stage",
        "Solo, the all-GPU transcode is slower (11.3 s) than CPU decode + NVENC (7.3 s).", "bar",
        bars([(r["case"], r["wall_s"]) for r in md]), Axis("stage and backend", scale="category"), Axis("wall time", "s"),
        source="tables/media.json")
    mp = load("media_parallel.json")
    add("c6_parallel", "C6 media", "Four transcodes at once: CPU decode vs NVDEC",
        "About the same time per copy, but 387 vs 33 CPU-s.", "bar",
        [*bars([(r["decode"], r["cpu_s_all4"]) for r in mp], "CPU-s, all 4"),
         *bars([(r["decode"], r["per_copy_wall_s"]) for r in mp], "wall s per copy")],
        Axis("decode", scale="category"), Axis("seconds", "s"), source="tables/media_parallel.json")

    cap, lad = load("capacity.json"), {r["ladder"]: r for r in load("ladders.json")}
    by = defaultdict(list)
    for r in cap:
        by[r["ladder"]].append(r)
    rows = []
    for name, rs_ in by.items():
        if "prefill" in name:
            continue
        ok = [r for r in sorted(rs_, key=lambda r: r["level"]) if r["rate"]]
        rows += [{"x": r["level"], "series": name, "y": r["rate"] / ok[0]["rate"]} for r in ok]
    add("c8_speedup", "C8 capacity", "Throughput gain as load doubles",
        "Batched LLM decode keeps scaling (27-31x); whisper batching stops at 2.5x, copies of a process at 1.1-1.9x.",
        "line", rows, Axis("load: batch size or concurrent copies", scale="log2"), Axis("throughput / first level", "x", scale="log2"),
        source="tables/capacity.json", notes="median over repeats; prefill ladder left out (its x is prompt tokens)")
    for i, (name, rs_) in enumerate(by.items()):
        L = lad[name]
        rs_ = sorted(rs_, key=lambda r: r["level"])
        rows = []
        peak = max(r["rate"] for r in rs_ if r["rate"])
        rows += [{"x": r["level"], "series": f"throughput / peak ({L['unit']})", "y": r["rate"] / peak,
                  "label": f'{r["rate"]:.4g} {L["unit"]}'} for r in rs_ if r["rate"]]
        for k, lab in RES.items():
            vals = [r.get(k) for r in rs_ if r["rate"]]
            if max((v or 0) for v in vals) >= 0.05:
                rows += [{"x": r["level"], "series": lab, "y": r[k]} for r in rs_ if r["rate"] and r.get(k) is not None]
        capped = [r for r in rs_ if (r.get("power_capped_share") or 0) >= 0.5]
        ev = [{"x": r["level"], "label": "out of memory"} for r in rs_ if not r["rate"]]
        ev += [{"x": r["level"], "label": f'power-capped {r["power_capped_share"]:.0%} of the time'} for r in capped]
        add(f"c8_ladder_{i + 1:02d}", "C8 capacity", f"What tops out first: {name}",
            f'Peak {L["peak_rate"]:.4g} {L["unit"]} at {L["peak_level"]} ({L["speedup"]}x the first level); '
            + (f'plateau from {L["plateau_level"]}' if L["plateau_level"] else "no plateau") + f'; limiter: {L["limiter"]}'
            + (f'; out of memory at {L["wall_at"]}' if L["wall_at"] else "") + ".",
            "line", rows, Axis("prompt tokens" if "prompt" in name else "concurrent copies" if "concurrent" in name
                               else "batch size", scale="log2"), Axis("share of ceiling", "x", lo=0, hi=1),
            refs=[{"y": 0.9, "label": "limiter threshold 0.9"}], events=ev, source="tables/capacity.json, tables/ladders.json",
            notes="device-wide NVML readings, median over repeats; prefill bursts are shorter than NVML's ~1 s averaging"
            if "prefill" in name else "device-wide NVML readings, median over repeats")
    add("c8_limiters", "C8 capacity", "Limiter per ladder", "The pre-registered rule's verdict for each ladder.", "table",
        [{"x": r["ladder"], "y": r["peak_rate"], "series": r["unit"], "label": r["limiter"], "pos_x": r["speedup"],
          "pos_y": r["plateau_level"], "pos_w": r["wall_at"]} for r in load("ladders.json")],
        columns=["x", "y", "series", "pos_x", "pos_y", "pos_w", "label"], source="tables/ladders.json",
        notes="table columns: x = ladder, y = peak rate, series = unit, pos_x = speedup, pos_y = plateau level, "
              "pos_w = out-of-memory level, label = limiter")

    rows = []
    for r in load("packing.json"):
        for part in r["shares"].split(", "):
            task, v = part.rsplit(" ", 1)
            rows.append({"x": r["mix"], "series": task, "y": float(v)})
    add("c8_packing", "C8 capacity", "Packing tasks with different limiters",
        "Bars sum each task's share of its solo speed: 4 mixed tasks give 3.08x the work of one; LLM + mask lose almost nothing.",
        "stacked_bar", rows, Axis("mix", scale="category"), Axis("sum of shares of solo speed", "x"),
        refs=[{"y": 1, "label": "one task alone"}], source="tables/packing.json")

    bt = [r for r in load("batching.csv") if r["experiment"] == "C8 capacity" and r["profile"] == "none"]
    g = defaultdict(list)
    for r in bt:
        g[(r["model"], int(r["sequences"]))].append(float(r["per_sequence_tok_s"]))
    add("c8_per_request", "C8 capacity", "Per-request speed as the batch grows",
        "Each request slows as more share the batch, while the total keeps rising.", "line",
        [{"x": s, "series": m, "y": statistics.median(v)} for (m, s), v in sorted(g.items())],
        Axis("sequences in the batch", scale="log2"), Axis("decode speed per sequence", "tok/s", scale="log2"),
        source="tables/batching.csv")

    ss = load("stress_series.csv")
    for mode, unit in (("launch", "kernel/s"), ("bw", "GB/s"), ("compute", "TFLOP/s")):
        rows, win = [], []
        for paired, lab in ((False, "solo"), (True, "beside LLM decode")):
            b = defaultdict(list)
            for r in ss:
                if r["mode"] == mode and (r["llm_from"] != "") == paired:
                    b[round(float(r["t"]))].append(float(r["rate"]))
                    if paired:
                        win.append((float(r["llm_from"]), float(r["llm_to"])))
            rows += [{"x": t, "series": lab, "y": statistics.mean(v)} for t, v in sorted(b.items())]
        lo, hi = statistics.median(w[0] for w in win), statistics.median(w[1] for w in win)
        add(f"c3_stressor_{mode}", "C3 interference", f"{mode} stressor's own rate, solo vs beside LLM decode",
            "The shaded window is when the LLM ran; the drop inside it is what the stressor lost.", "line", rows,
            Axis("time since the stressor started", "s"), Axis("stressor rate", unit),
            bands=[{"from": max(lo, 0), "to": hi, "label": "LLM decode running"}], source="tables/stress_series.csv",
            notes="mean of 3 runs per arm in 1 s bins")

    ob = load("observer.json")
    ns = [r for r in ob if r["effect"] == "Nsight in the container"]
    add("c9_nsight", "C9 observer effect", "Nsight's cost grows with kernel launch rate",
        "Steady-state cost ranges 0-20%; launch-heavy jobs pay more (Spearman -0.48, moderate).", "scatter",
        [{"x": float(re.search(r"([\d,]+) kernel launches/s", r["detail"])[1].replace(",", "")), "series": r["condition"].split(" (")[1][:-1],
          "y": r["value"], "label": r["condition"]} for r in ns],
        Axis("kernel launches per second", scale="log10"), Axis("steady-state rate change", "%"),
        refs=[{"y": 0, "label": "no change"}], source="tables/observer.json")
    ip = [r for r in ob if r["effect"] == "Idle power floor"]
    add("c9_idle_power", "C9 observer effect", "Idle GPU power under each kind of observation",
        "The sampler at any interval leaves idle power at ~16 W; Nsight GPU metrics add ~15 W.", "bar",
        [{"x": r["condition"].replace("C9 observer · ", "").replace("X0 idle floor · ", "earlier X0: "), "y": r["value"],
          "label": r["detail"]} for r in ip],
        Axis("observation", scale="category"), Axis("idle power", "W"), refs=[{"y": 16.8, "label": "unobserved idle 16.8 W"}],
        source="tables/observer.json")
    sp = [r for r in ob if r["effect"] == "Sampler interval"]
    add("c9_sampler", "C9 observer effect", "Does a faster sampler change the result?",
        "Against 1 s sampling, 100 ms changes nothing beyond tolerance except x264 at 20 ms (+1.9%).", "bar",
        [{"x": r["condition"].split(" @ ")[0], "series": r["condition"].split(" @ ")[1], "y": r["value"], "label": r["verdict"]} for r in sp],
        Axis("workload (metric)", scale="category"), Axis("change vs 1 s sampling", "%"),
        refs=[{"y": 1, "label": "+1% tolerance"}, {"y": -1, "label": "-1% tolerance"}], source="tables/observer.json")
    cl = sorted([r for r in ob if r["effect"] == "Container launch"], key=lambda r: r["value"])
    add("c9_launch", "C9 observer effect", "Time to first output by launch path",
        "A container adds 95-190 ms per start over native (1.6 ms); bursts of 16 reach 333 ms.", "bar",
        bars([(r["condition"], r["value"]) for r in cl]), Axis("launch path", scale="category"), Axis("time to first output", "ms"),
        source="tables/observer.json")
    add("c9_verdicts", "C9 observer effect", "Every observer measurement with its verdict", "", "table",
        [{"x": r["effect"], "series": r["condition"], "y": r["value"], "label": f'{r["unit"]}; {r["detail"]}', "pos_x": r["verdict"]}
         for r in ob], columns=["x", "series", "y", "label", "pos_x"], source="tables/observer.json",
        notes="table columns: x = effect, series = condition, y = value, label = unit and detail, pos_x = verdict")


def to_csv(spec) -> str:
    s = io.StringIO()
    w = csv.writer(s, lineterminator="\n")
    w.writerow(spec["columns"])
    w.writerows(spec["rows"])
    return s.getvalue()


FORMAT = """Each chart has a spec and a CSV of plotted marks, one row per mark (tidy / long format):

- `x`, `series`, `y`: the data. Pivot on `series` for a wide table. `y_min`/`y_max` are the range over repeats.
- `pos_x`, `pos_y`: where the mark sits in the plot area, 0-1, origin bottom-left. On a slide box at
  (left, top, width, height): `px = left + pos_x*width`, `py = top + (1 - pos_y)*height`.
- Bars: `pos_x` is the bar centre, `pos_w` its width, `pos_y0` its bottom, `pos_y` its top (stacked bars
  carry `y_base`, the value where the segment starts).
- Heatmaps: `pos_x`/`pos_y` are cell centres; `y` is the colour value (0-1, share of the ceiling).
- Axes in the spec carry `domain` and `ticks` (value + pos), or `categories` (value + pos) for category axes.
  `log2`/`log10` axes place positions on the log scale. `refs` are horizontal lines, `events` vertical markers,
  `bands` shaded x ranges, each with positions.
- `kpi` and `table` charts are tables: their columns are explained in `notes`."""


def to_md(meta) -> str:
    out = [f"# GPU workload profiling: chart pack\n\n{meta}\n\n## How to read this\n\n{FORMAT}\n\n## Charts\n"]
    out += [f"- [{c['title']}](#{c['id']}) `{c['id']}` ({c['kind']})" for c in CHARTS]
    for c in CHARTS:
        spec = {k: v for k, v in c.items() if k not in ("rows", "columns", "title", "takeaway")}
        out.append(f"\n<a id=\"{c['id']}\"></a>\n## {c['title']}\n\n**{c['section']}.** {c['takeaway']}\n\n"
                   f"```json\n{json.dumps(spec, ensure_ascii=False)}\n```\n\n```csv\n{to_csv(c)}```")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.DATA / "chart-pack"))
    a = ap.parse_args()
    build()
    meta = ("RTX 5070 Ti (16 GiB, 300 W limit), Ryzen 7 3700X, measured 2026-10-16 to 2026-10-19 with gauge "
            "(one job per container, NVML + cgroup sampled every 100 ms). Rates come from profiler-off runs; "
            "device readings are whole-GPU. Source tables are in gpu-profiling-2026-10-19.zip under tables/.")
    pack = {"title": "GPU workload profiling: chart pack", "about": meta, "format": FORMAT, "charts": CHARTS}
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(pack, ensure_ascii=False))
    out.with_suffix(".md").write_text(to_md(meta))
    page = HTML.read_text().replace("/*PACK*/null", json.dumps(pack, ensure_ascii=False).replace("</", "<\\/"))
    out.with_suffix(".html").write_text(page)
    print(f"{len(CHARTS)} charts, {sum(len(c['rows']) for c in CHARTS)} marks -> {out}.{{md,json,html}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
