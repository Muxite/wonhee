"""Layered-DAG family: the DAG engine (`draw_dag`) and eight looks.

All of them lay the graph out with graphkit.layered() (Sugiyama-style), scale the layout past the frame so
structure bleeds off every edge, and run a thin accent critical path through the whole width. Pass a real
graph as ``ctx.data`` (see graphkit.as_graph for the accepted shapes) and the same look draws it.
"""
from __future__ import annotations

import numpy as np
import cv2

from .. import graphkit as gk
from ..registry import effect

DATA_DOC = ("ctx.data: optional real graph. {'nodes':[{'id','group'?,'weight'?}],'edges':[{'source','target'}]}, "
            "or module -> imports dict, or networkx adjacency dict, or a graphkit.Graph. Without it a seeded synthetic DAG is used.")


def pick_graph(ctx, make):
    rng = ctx.rng
    if ctx.data is not None:
        g = gk.as_graph(ctx.data)
        mx = int(ctx.p("max_nodes", 140))
        if g.n > mx:
            deg = g.degree(); keep = np.sort(np.argsort(-deg, kind="stable")[:mx]); g = g.subgraph(keep)
        return g
    return make(rng)


def edge_tone(pal, x, lo=0.30, hi=0.44):
    return gk.tone(pal, lo + (hi - lo) * float(x))


def view_for(ctx, W, H, bleed=0.05, shift=True):
    """A view rectangle slightly larger than the frame, nudged by the seed so nothing is centred."""
    rng = np.random.default_rng(ctx.seed + 991)
    b = float(ctx.p("bleed", bleed))
    sx = (rng.random() - 0.5) * 0.04 * W if shift else 0
    sy = (rng.random() - 0.5) * 0.04 * H if shift else 0
    return (-b * W + sx, -b * H + sy, (1 + b) * W + sx, (1 + b) * H + sy)


def swing_path(lay: gk.Layered) -> list[int]:
    """Critical path with maximal vertical travel: longest by layer, ties (all) broken by sum |dy| so the path
    weaves through the whole frame height."""
    N = len(lay.layer)
    down = [[] for _ in range(N)]
    indeg = np.zeros(N, int)
    for ch in lay.chains:
        for u, v in zip(ch[:-1], ch[1:]):
            if v not in down[u]:
                down[u].append(v); indeg[v] += 1
    order = np.argsort(lay.layer, kind="stable")
    best = np.zeros(N); prev = -np.ones(N, int)
    for v in order:
        for w in down[v]:
            sc = best[v] + 1000 + abs(lay.y[w] - lay.y[v]) * 10
            if sc > best[w]:
                best[w] = sc; prev[w] = v
    v = int(np.argmax(best)); path = [v]
    while prev[v] >= 0:
        v = int(prev[v]); path.append(v)
    return path[::-1]


def expand_path(lay: gk.Layered, real_path) -> list[int]:
    """Real-node path -> path through dummy waypoints (so long edges on the critical path light up whole)."""
    chains = {(ch[0], ch[-1]): ch for ch in lay.chains}
    out = [int(real_path[0])]
    for a, b in zip(real_path[:-1], real_path[1:]):
        ch = chains.get((int(a), int(b))) or (chains.get((int(b), int(a))) or [None])[::-1]
        out += list(ch[1:]) if ch and ch[0] is not None else [int(b)]
    return out


def draw_dag(cv: gk.Canvas, pal, g: gk.Graph, lay: gk.Layered, px: np.ndarray, *, node="box", edge="s", node_size=1.0,
             edge_w=1.0, hot=True, hot_nodes=None, glow=0.55, casing=None, edge_alpha=1.0, node_alpha=1.0, node_tone=(0.38, 0.60),
             edge_tone_rng=(0.30, 0.44), hot_w=1.8, aspect=1.7, node_fill_canvas=False, group_color=True, rails=None, accent_nodes=True, path_mode="swing", node_ring=None):
    """Draw edges, nodes and the hot path of a laid-out DAG. ``px`` (N,2) are pixel coords for real+dummy nodes."""
    W, H = cv.W, cv.H
    nl = lay.nlayers
    counts = np.bincount(lay.layer[:lay.n_real], minlength=nl)
    sx = (px[:, 0].max() - px[:, 0].min()) / max(nl - 1, 1)
    sy = (px[:, 1].max() - px[:, 1].min()) / max(counts.max(), 1)
    unit = min(sy * 0.40, sx * 0.20) * node_size
    unit = max(unit, W * 0.006)
    ew = max(2.0, 0.0052 * W * edge_w)
    if hot_nodes is not None:
        path = expand_path(lay, hot_nodes)
    elif not hot:
        path = []
    elif path_mode == "longest":
        path = expand_path(lay, gk.longest_path(g))
    else:
        path = swing_path(lay)
    hot_pairs = set(zip(path[:-1], path[1:]))
    b = gk.Batch(cv)
    hot_b = gk.Batch(cv)
    hot_mask = cv.mask()
    seen = set()
    for ci, ch in enumerate(lay.chains):
        for u, v in zip(ch[:-1], ch[1:]):
            if (u, v) in seen:
                continue
            seen.add((u, v))
            p, q = px[u], px[v]
            ish = (u, v) in hot_pairs
            if edge == "line":
                pts = np.array([p, q])
            elif edge == "ortho":
                pts = gk.ortho_path(p, q, 0.3 + 0.4 * ((u * 7 + v * 13) % 9) / 8.0)
            elif edge == "bundle" and rails is not None:
                gy = rails(u, v); dx = q[0] - p[0]
                a1, a2 = np.array([p[0] + dx * 0.32, gy]), np.array([p[0] + dx * 0.68, gy])
                pts = np.concatenate([gk.s_curve(p, a1, True, 12, 0.5), gk.s_curve(a2, q, True, 12, 0.5)]) if False else np.concatenate([gk.s_curve(p, a1, True, 14, 0.5), gk.s_curve(a1, a2, True, 4, 0.5)[1:], gk.s_curve(a2, q, True, 14, 0.5)[1:]])
            else:
                pts = gk.s_curve(p, q, True, 20, 0.5)
            if ish:
                gk.m_line(hot_mask, pts, ew * hot_w)
                continue
            xr = ((p[0] + q[0]) / 2 - px[:, 0].min()) / max(np.ptp(px[:, 0]), 1)
            col = edge_tone(pal, xr, *edge_tone_rng)
            if casing is not None:
                gk.m_line(b.mask(casing), pts, ew + 5)
            gk.m_line(b.mask(col, edge_alpha), pts, ew)
    b.flush()
    if casing is not None:
        pass
    # nodes
    ng = int(g.group.max()) + 1 if g.n else 1
    for i in range(lay.n_real):
        c = px[i]
        if c[0] < -4 * unit or c[0] > W + 4 * unit or c[1] < -4 * unit or c[1] > H + 4 * unit:
            continue
        ishot = i in set(path)
        gi = int(g.group[i])
        col = gk.group_tone(pal, gi, ng, *node_tone) if group_color else gk.tone(pal, 0.5 * (node_tone[0] + node_tone[1]))
        sz = unit * (0.7 + 0.5 * min(1.0, g.weight[i] / (g.weight.max() + 1e-9))) if g.weight.max() > 1.5 else unit
        if node == "box" or node == "pill":
            pass
        gk.draw_node(hot_b if ishot else b, pal, node, c, sz, col, ishot and accent_nodes, aspect=aspect)
    b.flush()
    # hot path on top, nodes after edges
    cv.paint(hot_mask, pal.accent, 1.0)
    if glow:
        cv.glow(hot_mask, pal.accent, ew * 2.2, glow)
    hot_b.flush()
    return unit


def pm(ctx):
    return "longest" if ctx.data is not None else "swing"


def ghost_layer(ctx, cv, W, H, rng):
    """Out-of-focus oversized graph behind the main one: depth without a second hue."""
    pal = ctx.palette
    gc = gk.Canvas((W, H), pal.canvas)
    g = gk.random_layered(rng, [int(v) for v in rng.integers(4, 8, 9)], 0.5)
    lay = gk.layered(g, rng)
    view = (-0.2 * W + rng.random() * 0.1 * W, -0.15 * H, 1.2 * W, 1.15 * H)
    px = gk.layered_to_px(lay, view)
    draw_dag(gc, pal, g, lay, px, node="box", edge_w=1.6, hot=False, node_size=1.0, glow=0, edge_tone_rng=(0.16, 0.22),
             node_tone=(0.20, 0.26), accent_nodes=False)
    k = max(3, int(W * 0.006)) | 1
    gc.img[:] = cv2.GaussianBlur(gc.img, (0, 0), W * 0.003)
    cv.img[:] = gc.img


def make_bg(ctx, kind="flat"):
    W, H = ctx.size; pal = ctx.palette
    cv = gk.Canvas((W, H), pal.canvas)
    return cv


# ---------------------------------------------------------------------------------------------------------------
def _widths(rng, L, lo, hi):
    return [int(v) for v in rng.integers(lo, hi + 1, L)]


def _render(ctx, *, widths, node, edge="s", node_size=1.0, edge_w=1.0, aspect=1.7, extra=0.5, ghost=True, lanes=4,
            bands=None, bleed=0.05, glow=0.55, node_tone=(0.38, 0.60), casing=False, bg=None, hot_w=1.8, edge_tone_rng=(0.30, 0.44)):
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    cv = bg(ctx) if bg else gk.Canvas((W, H), pal.canvas)
    if ghost and bg is None and ctx.p("depth", True):
        ghost_layer(ctx, cv, W, H, np.random.default_rng(ctx.seed + 17))
    g = pick_graph(ctx, lambda r: gk.random_layered(r, widths(r) if callable(widths) else widths, extra, lanes=lanes))
    lay = gk.layered(g, rng)
    view = view_for(ctx, W, H, bleed)
    if bands == "lane":
        pass
    px = gk.layered_to_px(lay, view)
    if bands:
        bands(ctx, cv, g, lay, px, view)
    cas = pal.canvas if casing else None
    draw_dag(cv, pal, g, lay, px, node=node, edge=edge, node_size=node_size, edge_w=edge_w, aspect=aspect, glow=glow,
             node_tone=node_tone, casing=cas, hot_w=hot_w, edge_tone_rng=edge_tone_rng, path_mode=pm(ctx))
    return cv.out()


P_COMMON = {"depth": (True, "blurred oversized second graph behind for depth"), "bleed": (0.05, "how far the layout runs past the frame edges (fraction)"),
            "max_nodes": (140, "cap when drawing real ctx.data")}


@effect("graph-dag-flow", family="graph", tags=["dag", "graph", "all-over", "flow"],
        params={**P_COMMON, "layers": (10, "layers (columns)"), "wmin": (5, "min nodes per layer"), "wmax": (8, "max nodes per layer")})
def dag_flow(ctx):
    """Layered DAG: box nodes joined by S-curves, a thin accent critical path weaving the whole width, a blurred ghost graph behind. ctx.data: a real graph, see graphkit.as_graph."""
    return _render(ctx, widths=lambda r: _widths(r, int(ctx.p("layers")), int(ctx.p("wmin")), int(ctx.p("wmax"))), node="box", node_size=1.0, edge_w=1.0)


@effect("graph-dag-pills", family="graph", tags=["dag", "graph", "all-over", "pill"],
        params={**P_COMMON, "layers": (9, "layers"), "wmin": (4, "min width"), "wmax": (6, "max width")})
def dag_pills(ctx):
    """Chunky pill nodes (two sockets each) on a 7-layer DAG with orthogonal-ish S routing, fewer and bigger shapes for thumbnails. ctx.data: a real graph, see graphkit.as_graph."""
    return _render(ctx, widths=lambda r: _widths(r, int(ctx.p("layers")), int(ctx.p("wmin")), int(ctx.p("wmax"))), node="pill", node_size=1.0, edge_w=1.2, aspect=2.0, extra=0.35)


@effect("graph-dag-lattice", family="graph", tags=["dag", "graph", "all-over", "dense"],
        params={**P_COMMON, "layers": (15, "layers"), "wmin": (7, "min width"), "wmax": (11, "max width")})
def dag_lattice(ctx):
    """Dense lattice DAG: many round nodes and crossing edges, an even texture; the accent path is the only bright line. ctx.data: a real graph, see graphkit.as_graph."""
    return _render(ctx, widths=lambda r: _widths(r, int(ctx.p("layers")), int(ctx.p("wmin")), int(ctx.p("wmax"))), node="dot", node_size=0.9, edge_w=0.75, extra=0.7, bleed=0.04)


@effect("graph-dag-diamond", family="graph", tags=["dag", "graph", "all-over", "ring"],
        params={**P_COMMON})
def dag_diamond(ctx):
    """A diamond-profile DAG (widths swell then taper) scaled so the swell sits off-centre and both tapers bleed off the edges; ring nodes. ctx.data: a real graph, see graphkit.as_graph."""
    def widths(r):
        peak = int(r.integers(5, 8)); w = [2, 3, 4, 5, 7, 9, 8, 6, 5, 4, 3]
        return [max(2, v + int(r.integers(-1, 2))) for v in w[peak - 5:] + w[:peak - 5]] if False else [max(2, v + int(r.integers(-1, 2))) for v in w]
    return _render(ctx, widths=widths, node="ring", node_size=1.1, edge_w=1.0, extra=0.55, bleed=0.12)


def _bundle_rails(ctx):
    pass


@effect("graph-dag-bundle", family="graph", tags=["dag", "graph", "all-over", "bundle"],
        params={**P_COMMON, "layers": (9, "layers"), "lanes": (5, "rails per layer gap")})
def dag_bundle(ctx):
    """Edge-bundled DAG: every edge between two layers is pulled through one of a few horizontal rails, so edges merge into
    thick parallel bands (metro-map look). Rails are spread over the full height. ctx.data: a real graph, see graphkit.as_graph."""
    W, H = ctx.size
    lanes = int(ctx.p("lanes"))

    def rails_for(view):
        y0, y1 = view[1], view[3]
        return lambda u, v: y0 + (y1 - y0) * (((u * 5 + v * 3) % lanes) + 0.5) / lanes
    return _render_rails(ctx, rails_for, lanes)


def _render_rails(ctx, rails_for, lanes):
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    cv = gk.Canvas((W, H), pal.canvas)
    if ctx.p("depth", True):
        ghost_layer(ctx, cv, W, H, np.random.default_rng(ctx.seed + 17))
    g = pick_graph(ctx, lambda r: gk.random_layered(r, _widths(r, int(ctx.p("layers")), 5, 9), 0.8, lanes=lanes))
    lay = gk.layered(g, rng)
    view = view_for(ctx, W, H, 0.05)
    px = gk.layered_to_px(lay, view)
    rails = rails_for(view)
    # rail membership uses the destination lane so bundles are coherent
    draw_dag(cv, pal, g, lay, px, node="square", edge="bundle", rails=lambda u, v: view[1] + (view[3] - view[1]) * ((int(lay.y[u] * lanes) % lanes) + 0.5) / lanes,
             node_size=0.8, edge_w=1.7, edge_alpha=0.7, glow=0.5, path_mode=pm(ctx))
    return cv.out()


def _lane_bands(ctx, cv, g, lay, px, view):
    pal = ctx.palette; W, H = cv.W, cv.H
    L = int(ctx.p("lanes", 5))
    y0, y1 = view[1], view[3]
    b = gk.Batch(cv)
    for i in range(L):
        t = 0.10 + 0.07 * (i % 2)
        top = y0 + (y1 - y0) * i / L + 8; bot = y0 + (y1 - y0) * (i + 1) / L - 8
        gk.m_rrect(b.mask(gk.tone(pal, t * 0.9)), -40, top, W + 40, bot, 14)
    b.flush()


@effect("graph-dag-swimlane", family="graph", tags=["dag", "graph", "all-over", "lanes"],
        params={**P_COMMON, "layers": (14, "layers"), "lanes": (5, "swim lanes (full-width horizontal bands)")})
def dag_swimlane(ctx):
    """Swim-lane DAG: full-bleed horizontal lanes (tinted bands), square nodes snapped to their lane, S-curves crossing lanes. ctx.data: a real graph, see graphkit.as_graph."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    L = int(ctx.p("lanes"))
    cv = gk.Canvas((W, H), pal.canvas)
    g = pick_graph(ctx, lambda r: gk.random_layered(r, _widths(r, int(ctx.p("layers")), 3, 6), 0.5, lanes=L))
    g.group = np.minimum(g.group, L - 1) if ctx.data is None else (g.group % L)
    lay = gk.layered(g, rng)
    # snap y to lane, spread same-(layer,lane) nodes inside the lane; dummies interpolate
    cnt: dict[tuple, list] = {}
    for i in range(g.n):
        cnt.setdefault((int(lay.layer[i]), int(g.group[i])), []).append(i)
    for (l, ln), ids in cnt.items():
        ids.sort(key=lambda i: lay.y[i])
        for k, i in enumerate(ids):
            lay.y[i] = (ln + (k + 0.5) / len(ids)) / L
    for ch in lay.chains:
        a, bnode = ch[0], ch[-1]
        for k, d in enumerate(ch[1:-1], 1):
            lay.y[d] = lay.y[a] + (lay.y[bnode] - lay.y[a]) * k / (len(ch) - 1)
    view = view_for(ctx, W, H, 0.03)
    px = gk.layered_to_px(lay, view)
    _lane_bands(ctx, cv, g, lay, px, view)
    draw_dag(cv, pal, g, lay, px, node="square", node_size=1.0, edge_w=1.1, glow=0.5, node_tone=(0.42, 0.62), casing=None, path_mode=pm(ctx))
    return cv.out()


@effect("graph-dag-tiered", family="graph", tags=["dag", "graph", "all-over", "tiers"],
        params={**P_COMMON, "layers": (10, "tiers (full-height columns)")})
def dag_tiered(ctx):
    """Tiered DAG: each layer sits in a full-height column stripe that brightens left to right; pill nodes. ctx.data: a real graph, see graphkit.as_graph."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    cv = gk.Canvas((W, H), pal.canvas)
    g = pick_graph(ctx, lambda r: gk.random_layered(r, _widths(r, int(ctx.p("layers")), 4, 7), 0.4))
    lay = gk.layered(g, rng)
    view = view_for(ctx, W, H, 0.06)
    px = gk.layered_to_px(lay, view)
    nl = lay.nlayers; sx = (view[2] - view[0]) / max(nl - 1, 1)
    b = gk.Batch(cv)
    for l in range(nl):
        t = 0.09 + 0.10 * l / max(nl - 1, 1) + 0.04 * (l % 2)
        cx = view[0] + l * sx
        gk.m_rrect(b.mask(gk.tone(pal, t)), cx - sx * 0.46, -20, cx + sx * 0.46, H + 20, 10)
    b.flush()
    draw_dag(cv, pal, g, lay, px, node="pill", node_size=1.0, edge_w=1.2, aspect=2.0, glow=0.5, node_tone=(0.44, 0.62), casing=pal.canvas, path_mode=pm(ctx))
    return cv.out()
