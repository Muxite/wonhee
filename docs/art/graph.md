# Graph and DAG art (`graph-*` effects, `wonhee/art/graphkit.py`)

Owner direction (2026-11-03): strengthen the DAG and graph images. All of them obey the library rules: no focal point
(detail spread over the whole frame, structure bleeds off every edge, nothing centred), readable at 288x192 (big shapes,
edges >= 0.4% of the frame width, no text), one hue family with contrast from lightness, a *thin* accent that crosses the
whole frame, <= 1 s per 1500x1000, deterministic in (params, seed, size, palette, data).

```
PYTHONPATH=<checkout> python -m wonhee.art render graph-dag-flow --out a.png --seed 1 --palette ember
PYTHONPATH=<checkout> python -m wonhee.art sheet  graph-ring-wheel --out s.png --seeds 0,1,2,3 --palettes ember,ultramarine,violet-dusk
```

## graphkit API

| Area | Names |
|---|---|
| Graph model | `Graph(ids, edges, group, weight, eweight, directed, group_names, layer)`; `.n .m .degree() .subgraph(keep)` |
| Loaders | `as_graph(data)`, `from_repo(path, max_nodes)`, `from_ts(path)`, `from_json(path)`; synthetic `sbm()`, `random_layered()` |
| Layouts | `layered(g, rng)` (cycle removal, longest-path layering or `g.layer`, dummy waypoints for long edges, median crossing-reduction sweeps, neighbour relaxation blended with even spacing), `force()`, `grid_anchors()`, `tree_layout()`, `matrix_order()` (group, then reverse Cuthill-McKee), `communities()` (label propagation), `longest_path()` (critical path) |
| Routing | `s_curve`, `flow_path` (S-curves through waypoints), `ortho_path`, `bundle_curve` (Holten-style pull to a gate), `catmull`, `bezier` |
| Raster | `Canvas` (float canvas, `paint`, additive-style `glow` that also works on light canvases), `Batch` (collect masks per colour and z, flush once), `m_line m_disc m_ring m_rrect m_poly`, `sector_pts`, `arc_pts` |
| Node styles | `draw_node(batch, pal, style, c, size, colour, hot)` with `box pill dot ring square` |
| Colour | `tone(pal, t)` (0 canvas, 0.5 structure, 1 accent *lightness*, hue stays the structure's above 0.5), `group_tone()` |

`ctx.data` shapes accepted by every network effect (`as_graph`):

* `{"nodes":[{"id","group"?,"weight"?}], "edges":[{"source","target","weight"?}]}` (also `links`, `from`/`to`, pair lists; FlowSpec `layer` is kept as the fixed layer)
* `{"a.py": ["b.py","c.py"]}` module -> imports (path-like keys group by top directory)
* networkx-like adjacency `{"a": {"b": {"weight": 2}}}`, or a real networkx graph
* `[("a","b"), ...]`, a euglena-style trace `{"events":[{"type":"job_created","jobId","parentJobId"}]}`, a `Graph`

Real-data helpers:

* `from_repo("/path/to/repo")`: Python (`ast`) and TS/JS (relative and `@/` imports) import graph; nodes = files, group = top directory, weight = bytes; keeps the `max_nodes` best-connected files.
* `from_ts(".../nexusGraphData.ts")` parses the muksite projectViz modules (`entity('id','TYPE',..)`, `{ id, type }`, `{ source, target }`, FlowSpec `{ id, layer }` + `{ from, to }`); `from_json(".../fixtures/02-....json")` reads euglena traces.

Proven on: `muksite/src` import graph (110 files, `graph-dag-flow`, `graph-critical-path`, `graph-force-glow`, `graph-matrix`), the NEXUS knowledge graph
(`nexusGraphData.ts`, 46 nodes) and the `wonhee` package itself (30 modules). In the DAG effects a real graph's longest chain becomes the accent
path (`longest_path`, expanded through long-edge waypoints); synthetic ones use a maximum-vertical-travel path so it weaves across the frame.

## Effects (23)

Every effect has `depth`/`bleed`/`max_nodes` style params listed by `python -m wonhee.art list --family graph`.

Layered DAG (engine `graph_dag.draw_dag`; ctx.data = graph):
* `graph-dag-flow`: boxes with label bars, S-curves, blurred ghost graph behind. Params `layers 10, wmin 5, wmax 8`. Best: ember, ultramarine, teal-coral.
* `graph-dag-pills`: chunky two-socket pills, fewer bigger shapes (the best thumbnail legibility). `layers 9, wmin 4, wmax 6`.
* `graph-dag-lattice`: dense round-node lattice, even texture (`layers 15, wmin 7, wmax 11`).
* `graph-dag-diamond`: widths swell then taper, ring nodes, off-centre swell, strong bleed.
* `graph-dag-bundle`: metro-map edge bundling: edges snap onto horizontal rails per layer gap (`lanes 5`).
* `graph-dag-swimlane`: full-bleed lanes (`lanes 5`), square nodes snapped to lane.
* `graph-dag-tiered`: full-height tier stripes, pills (`layers 10`).
* `graph-dag-dots`: lattice DAG over a halftone dot matrix (`step 0.03`).
* `graph-dag-field`: box DAG over a wave-ripple interference field rendered via compose, remapped to one hue (`field 0.8`).
* `graph-critical-path`: dim dense lattice, one thin glowing path through the whole frame (`layers 18`).

Ring buffers (cells oldest -> newest, free outlined, body lightness rising, accent head; ctx.data = `{"values":[fill 0..1 per buffer], "head":[position 0..1]}`):
* `graph-ring-snake` (`cols 13`), `graph-ring-wheel` (`rings`, `thick`; hub off-frame, arcs sweep the frame; best of the family),
  `graph-ring-lanes` (`lanes 8`), `graph-ring-spiral` (`pitch 0.20`; cells grow with radius), `graph-ring-gauges` (`cols 4`, `segments 18`).

Networks (ctx.data = graph; synthetic = seeded stochastic block model):
* `graph-matrix`: block-ordered adjacency matrix, dot texture in empty cells, accent cells scattered (`cols 30`).
* `graph-force-glow`: knots on a jittered grid, bloom edges, one accent node per region (`nodes 132`, `clusters 12`).
* `graph-constellation`: best-candidate star field, straight links, halo rings (`nodes 96`, `clusters 10`).

Structures (ctx.data = `{"keys":[...]}` where it applies):
* `graph-btree` (`fanout 3`, `levels 3`, `keys_per_node 3`), `graph-heap` (`levels 5`), `graph-hash` (`buckets 15`, `load 9`), `graph-skiplist` (`levels 7`, `towers 26`, `p 0.58`; the classic search staircase is the accent).

## Palettes

Dark, single-family palettes read best: `ember`, `ultramarine`, `violet-dusk`, `teal-coral`, `green-gold`. Light palettes work for the DAG and ring
families (glows blend toward the colour, so they do not blow out) but low-contrast structure edges are weaker; check them with `check`.

## Failure modes

* Real graphs with very few layers (bipartite-like, e.g. NEXUS) leave one side with only edge bundles. Use `graph-force-glow`, `graph-constellation` or the matrix for those.
* `graph-matrix` on real data is a square N x N matrix cropped to the frame; small N is wrapped periodically. Fine for texture, not a faithful matrix view.
* Hub-heavy import graphs make `graph-force-glow` / constellation edge-heavy; cross-cluster edges are dimmed by edge count but not bundled.
* `graph-heap` is the weakest effect (few columns, sparse right side); `graph-ring-snake` is plain at small sizes.
* Edge widths are thick on purpose; with >300 nodes they merge into texture (that is the "river" look of `graph-dag-flow` on a real repo).

## Note for a future skill: turning a project's real structure into an image

1. Get a graph: `gk.from_repo(path)` for a codebase; a pipeline/dependency YAML or JSON -> `as_graph`; a knowledge graph or trace -> `{nodes, edges}`. Keep `group` meaningful (module directory, node type): it sets the lightness stripes and the cluster knots.
2. Pick the look from the data's shape: deep layered flow (build chains, pipelines) -> `graph-dag-flow` or `graph-critical-path` (the longest chain *is* the story); modular code -> `graph-matrix` or `graph-force-glow`; entity/knowledge graphs -> `graph-constellation`; ring/queue/cache projects -> a ring effect with real fill levels in `data.values`; data-structure projects -> the matching structure effect with `data.keys`.
3. `compose.render_effect(name, size=(1500,1000), seed=s, palette=p, data=graph)`, run `check` (coverage >= 0.9, center_bias <= 1.6, focal_ratio <= 3), look at the 288x192 downscale, vary seed (it only changes jitter and synthetic parts; the real structure stays).
4. Never invent nodes: cap with `max_nodes` (best-connected files) instead; say in the card copy what the image is (e.g. "import graph of 110 files").
