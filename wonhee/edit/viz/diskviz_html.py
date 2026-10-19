"""Emit a self-contained HTML treemap/circle-pack. Both layouts consume the same hierarchy
via d3.hierarchy(data).sum(sizeAccessor) — sizeAccessor is a swappable functor (default
d => d.value), not hardcoded to bytes, so the same renderer could size by another metric later.
"""
from __future__ import annotations

import json
from typing import Any

from wonhee.edit.viz.themes import theme as get_theme

_LABEL_MIN_AREA = 2200  # px^2; cells smaller than this skip the baked-in text label

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
  html, body {{ margin: 0; height: 100%; background: {bg}; overflow: hidden;
                font-family: system-ui, sans-serif; }}
  #hint {{ position: fixed; top: 10px; left: 12px; color: {hint}; font-size: 12px; z-index: 2; display: {hint_display}; }}
  #tooltip {{
    position: fixed; pointer-events: none; background: rgba(20,22,30,0.95); color: #eee;
    padding: 6px 9px; border-radius: 6px; font-size: 12px; line-height: 1.4; display: none;
    border: 1px solid rgba(255,255,255,0.15); max-width: 320px; z-index: 3;
  }}
  .cell-label {{ fill: {label}; font-size: 11px; pointer-events: none; }}
</style>
</head>
<body>
<div id="hint">{node_count} files &middot; {total_human} &middot; hover a cell</div>
<div id="tooltip"></div>
<svg id="stage" width="100%" height="100%"></svg>
<script>
const data = {tree_json};
const layoutMode = {layout_mode!r};

function hashHue(name) {{
  let h = 5381;
  for (let i = 0; i < name.length; i++) {{ h = ((h * 33) + name.charCodeAt(i)) >>> 0; }}
  return Math.floor((h * 137.508) % 360);
}}
function hashColor(name) {{ return `hsl(${{hashHue(name)}}, 65%, {node_lightness})`; }}
function humanSize(n) {{
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = n;
  while (v >= 1024 && i < units.length - 1) {{ v /= 1024; i++; }}
  return `${{v.toFixed(v < 10 && i > 0 ? 1 : 0)}} ${{units[i]}}`;
}}

const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("#stage");
const view = svg.append("g");
const zoomBehavior = d3.zoom().scaleExtent([0.3, 8]).on("zoom", (ev) => view.attr("transform", ev.transform));
svg.call(zoomBehavior);

const sizeAccessor = d => d.value;  // swappable: any function of a leaf datum -> number
const root = d3.hierarchy(data, d => d.children).sum(sizeAccessor).sort((a, b) => b.value - a.value);

const tooltip = d3.select("#tooltip");
const leaves = root.leaves();

function labelFor(d) {{
  return `${{d.data.name}} &middot; ${{humanSize(d.value)}}`;
}}

if (layoutMode === "rect") {{
  d3.treemap().size([width, height]).paddingInner(1.5).paddingOuter(2).round(true)(root);
  const cell = view.selectAll("g").data(leaves).join("g")
    .attr("transform", d => `translate(${{d.x0}},${{d.y0}})`);
  cell.append("rect")
    .attr("width", d => Math.max(0, d.x1 - d.x0))
    .attr("height", d => Math.max(0, d.y1 - d.y0))
    .attr("fill", d => hashColor(d.data.ext || "(none)"))
    .attr("fill-opacity", 0.85)
    .attr("stroke", "{stroke}").attr("stroke-width", 1)
    .on("mouseenter", (ev, d) => {{
      tooltip.style("display", "block").html(`<b>${{d.data.path}}</b><br>${{humanSize(d.value)}}`);
    }})
    .on("mousemove", (ev) => tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY + 10) + "px"))
    .on("mouseleave", () => tooltip.style("display", "none"));
  cell.filter(d => (d.x1 - d.x0) * (d.y1 - d.y0) > {label_min_area})
    .append("text").attr("class", "cell-label").attr("x", 4).attr("y", 14)
    .html(d => labelFor(d));
}} else {{
  d3.pack().size([width, height]).padding(2)(root);
  const cell = view.selectAll("g").data(leaves).join("g")
    .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
  cell.append("circle")
    .attr("r", d => d.r)
    .attr("fill", d => hashColor(d.data.ext || "(none)"))
    .attr("fill-opacity", 0.85)
    .attr("stroke", "{stroke}").attr("stroke-width", 1)
    .on("mouseenter", (ev, d) => {{
      tooltip.style("display", "block").html(`<b>${{d.data.path}}</b><br>${{humanSize(d.value)}}`);
    }})
    .on("mousemove", (ev) => tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY + 10) + "px"))
    .on("mouseleave", () => tooltip.style("display", "none"));
  cell.filter(d => Math.PI * d.r * d.r > {label_min_area})
    .append("text").attr("class", "cell-label").attr("x", d => -d.r * 0.6).attr("y", 4)
    .html(d => labelFor(d));
}}
</script>
</body>
</html>
"""


def render_html(tree: dict[str, Any], title: str, layout: str = "rect", theme: str = "dark",
                show_hint: bool = True, labels: bool = True) -> str:
    """labels=False bakes no file names into the picture (for private repos)."""
    def _count_and_sum(node: dict[str, Any]) -> tuple[int, int]:
        children = node.get("children")
        if not children:
            return 1, node["value"]
        count, total = 0, 0
        for c in children:
            c_count, c_total = _count_and_sum(c)
            count += c_count
            total += c_total
        return count, total

    node_count, total_bytes = _count_and_sum(tree)
    units = ["B", "KB", "MB", "GB", "TB"]
    v, i = float(total_bytes), 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    total_human = f"{v:.1f} {units[i]}" if i > 0 else f"{int(v)} {units[i]}"

    return _TEMPLATE.format(
        **get_theme(theme),
        hint_display="block" if show_hint else "none",
        title=title,
        node_count=node_count,
        total_human=total_human,
        tree_json=json.dumps(tree),
        layout_mode=layout,
        label_min_area=_LABEL_MIN_AREA if labels else "Infinity",
    )
