"""Emit a single self-contained HTML file: D3 force-directed graph (d3 from cdnjs), no build step, opens
directly in a browser; export.py screenshots it. Node size = LOC (log-scaled), color = hash(language), shape mixes
circles/rounded-rects by language, with jitter so dense clusters don't look like a rigid grid.
"""
from __future__ import annotations

import json

from wonhee.edit.viz.themes import theme as get_theme
from wonhee.edit.viz.graph import Graph

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
  html, body {{ margin: 0; height: 100%; background: {bg}; overflow: hidden; }}
  #hint {{ position: fixed; top: 10px; left: 12px; color: {hint}; font: 12px/1.4 system-ui, sans-serif; display: {hint_display}; }}
  #tooltip {{
    position: fixed; pointer-events: none; background: rgba(20,22,30,0.95); color: #eee;
    padding: 6px 9px; border-radius: 6px; font: 12px/1.4 system-ui, sans-serif; display: none;
    border: 1px solid rgba(255,255,255,0.15); max-width: 320px;
  }}
</style>
</head>
<body>
<div id="hint">{node_count} files &middot; {loc_total} LOC &middot; drag to pan, scroll to zoom, hover a node</div>
<div id="tooltip"></div>
<svg id="stage" width="100%" height="100%"></svg>
<script>
const graph = {graph_json};

function hashHue(name) {{
  let h = 5381;
  for (let i = 0; i < name.length; i++) {{ h = ((h * 33) + name.charCodeAt(i)) >>> 0; }}
  return Math.floor((h * 137.508) % 360);
}}
function hashColor(name) {{ return `hsl(${{hashHue(name)}}, 65%, {node_lightness})`; }}
function jitter(id, scale) {{
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) {{ h = Math.imul(h ^ id.charCodeAt(i), 16777619); }}
  const a = ((h >>> 0) % 1000) / 1000 - 0.5;
  const b = (((h >>> 8) >>> 0) % 1000) / 1000 - 0.5;
  return [a * scale, b * scale];
}}
function shapeFor(language) {{
  return hashHue(language) % 2 === 0 ? "circle" : "rect";
}}

const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("#stage");
const view = svg.append("g");
const zoomBehavior = d3.zoom().scaleExtent([0.05, 6]).on("zoom", (ev) => view.attr("transform", ev.transform));
svg.call(zoomBehavior);

const locExtent = d3.extent(graph.nodes, d => d.loc || 1);
const radius = d3.scaleLog()
  .domain([Math.max(1, locExtent[0]), Math.max(2, locExtent[1])])
  .range([4, 26]).clamp(true);

const nodes = graph.nodes.map(n => {{
  const [jx, jy] = jitter(n.id, 40);
  return {{...n, x: width / 2 + jx, y: height / 2 + jy}};
}});
const nodeById = new Map(nodes.map(n => [n.id, n]));
const links = graph.edges
  .filter(e => nodeById.has(e.from) && nodeById.has(e.to))
  .map(e => ({{source: e.from, target: e.to}}));

const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(50).strength(0.15))
  .force("charge", d3.forceManyBody().strength(-90).distanceMax(400))
  .force("x", d3.forceX(width / 2).strength(0.04))
  .force("y", d3.forceY(height / 2).strength(0.04))
  .force("collide", d3.forceCollide(d => radius(d.loc || 1) + 3));

const link = view.append("g")
  .attr("stroke", "{edge}").attr("stroke-opacity", {edge_opacity})
  .selectAll("line").data(links).join("line").attr("stroke-width", 1);

const tooltip = d3.select("#tooltip");

const node = view.append("g")
  .selectAll("g").data(nodes).join("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", (ev, d) => {{ if (!ev.active) sim.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y; }})
    .on("drag", (ev, d) => {{ d.fx = ev.x; d.fy = ev.y; }})
    .on("end", (ev, d) => {{ if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }}))
  .on("mouseenter", (ev, d) => {{
    tooltip.style("display", "block")
      .html(`<b>${{d.path}}</b><br>${{d.language}} &middot; ${{d.loc}} LOC`);
  }})
  .on("mousemove", (ev) => {{
    tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY + 10) + "px");
  }})
  .on("mouseleave", () => tooltip.style("display", "none"));

node.each(function(d) {{
  const r = radius(d.loc || 1);
  const g = d3.select(this);
  const color = hashColor(d.language);
  if (shapeFor(d.language) === "circle") {{
    g.append("circle").attr("r", r).attr("fill", color).attr("fill-opacity", 0.85)
      .attr("stroke", "{stroke}").attr("stroke-width", 1);
  }} else {{
    g.append("rect").attr("x", -r).attr("y", -r * 0.75).attr("width", r * 2).attr("height", r * 1.5)
      .attr("rx", r * 0.35).attr("fill", color).attr("fill-opacity", 0.85)
      .attr("stroke", "{stroke}").attr("stroke-width", 1);
  }}
}});

sim.on("tick", () => {{
  link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});

function fitToView() {{
  // bounds include each node's radius, so big nodes at the edge are not clipped
  const r = d => radius(d.loc || 1);
  const x0 = Math.min(...nodes.map(d => d.x - r(d))), x1 = Math.max(...nodes.map(d => d.x + r(d)));
  const y0 = Math.min(...nodes.map(d => d.y - r(d))), y1 = Math.max(...nodes.map(d => d.y + r(d)));
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  const scale = Math.min(6, 0.85 / Math.max(w / width, h / height));
  const tx = width / 2 - scale * (x0 + x1) / 2;
  const ty = height / 2 - scale * (y0 + y1) / 2;
  svg.transition().duration(400)
    .call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}}
// export.py waits for this flag, so a PNG is never taken mid-layout
window.__vizkitSettled = false;
sim.on("end", () => {{ fitToView(); setTimeout(() => {{ window.__vizkitSettled = true; }}, 500); }});
// also fit once early, in case the sim runs long enough that "settle" screenshots happen first
setTimeout(fitToView, 900);

</script>
</body>
</html>
"""


def render_html(graph: Graph, title: str = "depgraph", theme: str = "dark", show_hint: bool = True) -> str:
    stats = graph.stats()
    return _TEMPLATE.format(
        **get_theme(theme),
        hint_display="block" if show_hint else "none",
        title=title,
        node_count=stats["file_count"],
        loc_total=stats["total_loc"],
        graph_json=json.dumps(graph.to_dict()),
    )
