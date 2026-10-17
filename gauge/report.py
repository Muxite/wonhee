"""Static HTML report: aggregate scores, predicted-vs-actual scatter per target, run table."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

from gauge.evaluate import aggregate
from gauge.workloads.base import get

COLORS = {"prior": "#c2410c", "history": "#2563eb", "calibrated": "#059669"}
UNITS = {"runtime_s": (1, "s"), "cpu_time_s": (1, "s"), "ram_peak_bytes": (2**30, "GiB"),
         "vram_peak_bytes": (2**30, "GiB")}


def _num(v, d=3):
    if v is None:
        return "–"
    return f"{v:.{d}g}" if isinstance(v, float) else str(v)


def scatter(points: list[tuple[float, float, str, bool]], unit: str, size=300) -> str:
    """Log-log predicted (x) vs actual (y). Hollow = censored. Diagonal = perfect."""
    pts = [(x, y, c, cen) for x, y, c, cen in points if x and y and x > 0 and y > 0]
    if not pts:
        return "<p class=muted>no data</p>"
    lo = math.log10(min(min(p[0] for p in pts), min(p[1] for p in pts))) - 0.1
    hi = math.log10(max(max(p[0] for p in pts), max(p[1] for p in pts))) + 0.1
    pad = 34
    f = lambda v: pad + (math.log10(v) - lo) / (hi - lo) * (size - 2 * pad)  # noqa: E731
    out = [f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img">',
           f'<line x1="{pad}" y1="{size - pad}" x2="{size - pad}" y2="{pad}" class="diag"/>',
           f'<text x="{size / 2}" y="{size - 6}" class="ax">predicted ({unit})</text>',
           f'<text x="10" y="{size / 2}" class="ax" transform="rotate(-90 10 {size / 2})">actual</text>']
    for e in range(math.ceil(lo), math.floor(hi) + 1):
        v = 10 ** e
        out.append(f'<text x="{f(v)}" y="{size - pad + 14}" class="tick">{v:g}</text>')
        out.append(f'<text x="{pad - 4}" y="{size - f(v) + 3}" class="tick" text-anchor="end">{v:g}</text>')
    for x, y, c, cen in pts:
        fill = "none" if cen else c
        out.append(f'<circle cx="{f(x):.1f}" cy="{size - f(y):.1f}" r="3.5" fill="{fill}" stroke="{c}"/>')
    return "".join(out) + "</svg>"


def corun_slowdowns(conn) -> list[dict]:
    """Each co-run job vs the median of isolated successful runs with the identical spec."""
    import statistics

    out = []
    rows = conn.execute("SELECT r.id, r.corun_group, r.workload, r.spec_hash, r.spec_json, r.failure_class, s.json"
                        " FROM run r JOIN summary s ON s.run_id=r.id WHERE r.corun_group IS NOT NULL"
                        " ORDER BY r.corun_group, r.id").fetchall()
    for r in rows:
        s = json.loads(r["json"])
        w = get(r["workload"])
        norm = lambda spec: w.validate(json.loads(spec)["params"]).model_dump()  # noqa: E731  defaults filled in
        mine = norm(r["spec_json"])
        iso = [json.loads(x["json"]) for x in conn.execute(
            "SELECT r.spec_json, s.json FROM run r JOIN summary s ON s.run_id=r.id WHERE r.workload=?"
            " AND r.corun_group IS NULL AND r.failure_class='success'", (r["workload"],)) if norm(x["spec_json"]) == mine]
        med = lambda k: statistics.median([x[k] for x in iso if x.get(k)]) if any(x.get(k) for x in iso) else None  # noqa: E731
        base_rt, base_vram = med("runtime_s"), med("vram_peak_bytes")
        out.append({"group": r["corun_group"], "run": r["id"], "workload": r["workload"],
                    "params": json.loads(r["spec_json"])["params"], "outcome": r["failure_class"],
                    "runtime_s": s.get("runtime_s"), "isolated_runtime_s": base_rt, "n_isolated": len(iso),
                    "slowdown": s["runtime_s"] / base_rt if base_rt and s.get("runtime_s") else None,
                    "vram_peak_bytes": s.get("vram_peak_bytes"), "isolated_vram_bytes": base_vram})
    return out


def corun_section(conn) -> list[str]:
    rows = corun_slowdowns(conn)
    if not rows:
        return []
    parts = ["<h2>Co-run trials</h2><p class=muted>slowdown = co-run runtime / median isolated runtime of the "
             "identical spec. Device-scope GPU metrics in these runs are shared, so they are not per-job.</p>",
             "<table><tr><th>group</th><th>run</th><th>workload</th><th>params</th><th>outcome</th><th>runtime s</th>"
             "<th>isolated s (n)</th><th>slowdown</th><th>vram GiB</th><th>isolated vram GiB</th></tr>"]
    gib = lambda v: _num(v / 2**30, 3) if v else "–"  # noqa: E731
    for r in rows:
        parts.append(f"<tr><td>{r['group']}</td><td>{r['run']}</td><td>{r['workload']}</td>"
                     f"<td><code>{html.escape(json.dumps(r['params']))}</code></td><td>{r['outcome']}</td>"
                     f"<td>{_num(r['runtime_s'])}</td><td>{_num(r['isolated_runtime_s'])} ({r['n_isolated']})</td>"
                     f"<td>{_num(r['slowdown'], 3)}</td><td>{gib(r['vram_peak_bytes'])}</td>"
                     f"<td>{gib(r['isolated_vram_bytes'])}</td></tr>")
    return parts + ["</table>"]


def write_report(conn, out: Path) -> Path:
    agg = aggregate(conn)
    ev = conn.execute(
        "SELECT e.*, p.predictor, r.workload FROM evaluation e JOIN prediction p ON p.id=e.prediction_id"
        " JOIN run r ON r.id=e.run_id WHERE r.corun_group IS NULL").fetchall()
    runs = conn.execute("SELECT r.*, s.json AS sj FROM run r LEFT JOIN summary s ON s.run_id=r.id"
                        " ORDER BY r.id").fetchall()
    parts = ["<h2>Prediction scores (isolated runs)</h2>",
             "<p class=muted>coverage = share of non-censored runs where actual ≤ upper bound. "
             "mape = mean |actual−estimate|/actual. Upper bounds are heuristics, not calibrated quantiles.</p>",
             "<table><tr><th>workload</th><th>target</th><th>predictor</th><th>n</th><th>censored</th>"
             "<th>coverage</th><th>mape</th><th>actual/pred</th><th>mean under</th><th>mean excess</th></tr>"]
    for r in agg:
        sc, u = UNITS.get(r["target"], (1, ""))
        parts.append(
            f"<tr><td>{r['workload']}</td><td>{r['target']}</td><td style='color:{COLORS.get(r['predictor'])}'>"
            f"{r['predictor']}</td><td>{r['n']}</td><td>{r['censored']}</td><td>{_num(r['coverage'], 2)}</td>"
            f"<td>{_num(r['mape'], 2)}</td><td>{_num(r['mean_ratio'], 3)}</td>"
            f"<td>{_num((r['mean_under'] or 0) / sc, 3)}{u}</td><td>{_num((r['mean_excess'] or 0) / sc, 3)}{u}</td></tr>")
    parts.append("</table><h2>Predicted vs actual</h2><p class=muted>log-log; diagonal is perfect; "
                 "hollow = censored (OOM/timeout). Point = estimate if the predictor gives one, else the upper "
                 "bound.</p>")
    for wl in sorted({e["workload"] for e in ev}):
        parts.append(f"<h3>{html.escape(wl)}</h3><div class=grid>")
        for target, (sc, unit) in UNITS.items():
            pts = [((e["estimate"] if e["estimate"] is not None else e["upper"]) / sc if (
                e["estimate"] if e["estimate"] is not None else e["upper"]) else None,
                    e["actual"] / sc if e["actual"] else None, COLORS.get(e["predictor"], "#666"), bool(e["censored"]))
                   for e in ev if e["workload"] == wl and e["target"] == target]
            if pts:
                parts.append(f"<figure><figcaption>{target}</figcaption>{scatter(pts, unit)}</figure>")
        parts.append("</div>")
    parts += corun_section(conn)
    parts.append("<h2>Runs</h2><table><tr><th>id</th><th>workload</th><th>params</th><th>outcome</th>"
                 "<th>runtime s</th><th>cpu s</th><th>ram peak GiB</th><th>vram peak GiB</th><th>dev util</th>"
                 "<th>samples</th><th>corun</th></tr>")
    for r in runs:
        s = json.loads(r["sj"]) if r["sj"] else {}
        params = json.loads(r["spec_json"])["params"]
        gib = lambda v: _num(v / 2**30, 3) if v else "–"  # noqa: E731
        parts.append(
            f"<tr><td>{r['id']}</td><td>{r['workload']}</td><td><code>{html.escape(json.dumps(params))}</code></td>"
            f"<td>{r['failure_class'] or r['status']}</td><td>{_num(s.get('runtime_s'))}</td>"
            f"<td>{_num(s.get('cpu_time_s'))}</td><td>{gib(s.get('ram_peak_bytes'))}</td>"
            f"<td>{gib(s.get('vram_peak_bytes'))}</td><td>{_num(s.get('dev_util_mean'), 2)}</td>"
            f"<td>{s.get('sample_count', '–')}</td><td>{r['corun_group'] or ''}</td></tr>")
    parts.append("</table>")
    page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>gauge report</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#d6d3d1}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1917;--fg:#e7e5e4;--muted:#a8a29e;--line:#44403c}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1200px;padding:0 16px}}
table{{border-collapse:collapse;font-size:12.5px;display:block;overflow-x:auto}}
td,th{{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;white-space:nowrap}}
.muted{{color:var(--muted)}} .grid{{display:flex;flex-wrap:wrap;gap:12px}}
figure{{margin:0}} figcaption{{font-size:12px;color:var(--muted)}}
svg .diag{{stroke:var(--line);stroke-dasharray:4}} svg .ax,svg .tick{{fill:var(--muted);font-size:10px}}
svg .ax{{text-anchor:middle}} code{{font-size:11px}}
</style></head><body><h1>gauge</h1>
<p class=muted>prior <span style="color:{COLORS['prior']}">●</span> history
<span style="color:{COLORS['history']}">●</span> calibrated <span style="color:{COLORS['calibrated']}">●</span></p>
{''.join(parts)}</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
