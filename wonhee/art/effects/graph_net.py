"""Network family: adjacency matrix, glowing force graph with spread clusters, knowledge-graph constellation.

``ctx.data``: a real graph (graphkit.as_graph accepts {nodes, edges}, module -> imports, adjacency dicts...).
Without data a seeded stochastic block model is used. No hubs: clusters sit on a jittered grid that covers the frame.
"""
from __future__ import annotations

import math

import numpy as np

from .. import graphkit as gk
from ..registry import effect


def _net(ctx, n, k, p_in, p_out, directed=False):
    rng = ctx.rng
    if ctx.data is not None:
        g = gk.as_graph(ctx.data, directed)
        mx = int(ctx.p("max_nodes", 220))
        if g.n > mx:
            deg = g.degree(); g = g.subgraph(np.sort(np.argsort(-deg, kind="stable")[:mx]))
        if len(set(g.group.tolist())) < 8:            # few / no groups: refine with detected communities
            comm = gk.communities(g, rng)
            if comm.max() + 1 > len(set(g.group.tolist())):
                g.group = comm
        _, g.group = np.unique(g.group, return_inverse=True)          # compact ids 0..k-1
        return g
    return gk.sbm(rng, n, k, p_in, p_out, directed)


def _dots(cv, pal, step, jit_rng=None, tone_t=0.17, r=0.11, off=(0, 0)):
    """Quiet micro-dot lattice (the owner's liked texture): even detail everywhere."""
    b = gk.Batch(cv); m = b.mask(gk.tone(pal, tone_t))
    for y in np.arange(off[1], cv.H + step, step):
        for x in np.arange(off[0], cv.W + step, step):
            gk.m_disc(m, (x, y), step * r)
    b.flush()


# ------------------------------------------------------------------------------------------------------------------
@effect("graph-matrix", family="graph", tags=["matrix", "graph", "all-over", "blocks"],
        params={"cols": (30, "matrix columns across the frame"), "max_nodes": (220, "cap for real data")})
def graph_matrix(ctx):
    """Block-ordered adjacency matrix: dense blocks run corner to corner (bipartite when synthetic), micro-dots fill empty
    cells, a scattering of accent cells. Real data: square N x N matrix (block ordering = group, then reverse Cuthill-McKee), cropped to
    the frame and wrapped if N is small. ctx.data: a graph."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    cols = int(ctx.p("cols"))
    if ctx.data is not None:                                  # real graph: about one cell per node (28..64 across)
        cols = int(np.clip(gk.as_graph(ctx.data).n, 28, 64))
    cs = W / (cols - 0.4); rows = int(math.ceil(H / cs)) + 2
    cv = gk.Canvas((W, H), pal.canvas)
    oy = -cs * rng.uniform(0.1, 0.6); ox = -cs * rng.uniform(0.1, 0.5)
    if ctx.data is None:
        k = 6
        cb = np.sort(rng.integers(0, k, cols + 2)); rb = np.sort(np.minimum(k - 1, (np.arange(rows + 2) / (rows + 2) * k + rng.normal(0, 0.15, rows + 2)).astype(int)))
        cb = np.sort(np.minimum(k - 1, (np.arange(cols + 2) / (cols + 2) * k + rng.normal(0, 0.15, cols + 2)).astype(int)))
        P = np.where(rb[:, None] == cb[None, :], 0.8, 0.16)
        P *= 0.65 + 0.7 * rng.random(P.shape)
        A = (rng.random(P.shape) < P) * (0.35 + 0.65 * rng.random(P.shape))
        blk = rb[:, None] == cb[None, :]
        bounds = [(int(np.where(cb == b)[0].min()), int(np.where(cb == b)[0].max()), int(np.where(rb == b)[0].min()), int(np.where(rb == b)[0].max())) for b in range(k) if (cb == b).any() and (rb == b).any()]
    else:
        g = _net(ctx, 0, 0, 0, 0, directed=False)
        order = gk.matrix_order(g, rng); n = g.n
        M = np.zeros((n, n))
        pos = np.empty(n, int); pos[order] = np.arange(n)
        for (a, b), w in zip(g.edges, g.eweight):
            M[pos[a], pos[b]] = max(M[pos[a], pos[b]], 0.5 + 0.5 * min(w, 3) / 3); M[pos[b], pos[a]] = M[pos[a], pos[b]]
        A = M[np.arange(rows + 2)[:, None] % n, np.arange(cols + 2)[None, :] % n]
        gs = g.group[order]
        blk = (gs[np.arange(rows + 2)[:, None] % n] == gs[np.arange(cols + 2)[None, :] % n])
        bounds = []
    b = gk.Batch(cv)
    # block plates
    for (c0, c1, r0, r1) in bounds:
        gk.m_rrect(b.mask(gk.tone(pal, 0.115), 1.0, 0), ox + c0 * cs, oy + r0 * cs, ox + (c1 + 1) * cs, oy + (r1 + 1) * cs, cs * 0.35)
    if not bounds:
        bl = blk.astype(float)
        for r in range(blk.shape[0]):
            for c in range(blk.shape[1]):
                if blk[r, c]:
                    gk.m_rrect(b.mask(gk.tone(pal, 0.10), 1.0, 0), ox + c * cs + 1, oy + r * cs + 1, ox + (c + 1) * cs - 1, oy + (r + 1) * cs - 1, cs * 0.2)
    gap = cs * 0.1
    ys, xs = np.nonzero(A)
    sel = set()
    # accent cells: evenly spread, one per 6x6 cell patch that has a cell
    patch = {}
    for y, x in zip(ys, xs):
        patch.setdefault((y // 5, x // 5), []).append((y, x))
    for key, lst in sorted(patch.items()):
        if rng.random() < 0.5:
            sel.add(lst[int(rng.integers(0, len(lst)))])
    for y, x in zip(ys, xs):
        v = A[y, x]; X = ox + x * cs; Y = oy + y * cs
        isb = blk[y, x]
        t = 0.30 + 0.30 * float(v) if isb else 0.24 + 0.14 * float(v)
        col = pal.accent if (y, x) in sel else gk.tone(pal, t)
        gk.m_rrect(b.mask(col, 1.0, 2 if (y, x) in sel else 1), X + gap, Y + gap, X + cs - gap, Y + cs - gap, cs * 0.22)
    # micro-dots in empty cells
    empty = b.mask(gk.tone(pal, 0.20), 1.0, 0)
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            if A[y, x] == 0:
                gk.m_disc(empty, (ox + (x + 0.5) * cs, oy + (y + 0.5) * cs), cs * 0.09)
    b.flush()
    return cv.out()


# ------------------------------------------------------------------------------------------------------------------
def _spread_layout(g, rng, aspect, pull=0.9):
    """Cluster knots on a jittered grid: each cluster gets its own force layout, scaled to a disc and dropped on its anchor, so
    density is even across the frame and no cluster dominates."""
    k = int(g.group.max()) + 1
    A = gk.grid_anchors(rng, k, aspect, 0.3)
    gx = max(1, int(round(math.sqrt(k * aspect)))); gy = max(1, int(math.ceil(k / gx)))
    cell = min(aspect / gx, 1.0 / gy)
    sizes = np.bincount(g.group, minlength=k)
    out = np.zeros((g.n, 2))
    for c in range(k):
        idx = np.where(g.group == c)[0]
        if not len(idx):
            continue
        sub = g.subgraph(idx)
        if len(idx) == 1:
            loc = np.zeros((1, 2))
        else:
            loc = gk.force(sub, rng, iters=80, aspect=1.0, spread=0.0)
            loc = (loc - loc.mean(0)) / (np.abs(loc - loc.mean(0)).max() + 1e-9)
        R = cell * 0.60 * (0.55 + 0.45 * math.sqrt(sizes[c] / max(sizes.max(), 1)))
        out[idx] = A[c] + loc * R
    return out


def _map_view(p, W, H, aspect, bleed=0.06):
    q = p / np.array([aspect, 1.0])
    return np.stack([(-bleed + q[:, 0] * (1 + 2 * bleed)) * W, (-bleed + q[:, 1] * (1 + 2 * bleed)) * H], 1)


def curved(p, q, rng_off):
    mid = (p + q) / 2; d = q - p
    nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-6)
    return gk.bezier(p, mid + nrm * rng_off * np.hypot(*d) * 0.5, mid + nrm * rng_off * np.hypot(*d) * 0.5, q, 18)


@effect("graph-force-glow", family="graph", tags=["force", "graph", "all-over", "glow", "clusters"],
        params={"nodes": (132, "synthetic node count"), "clusters": (12, "clusters on a jittered grid"), "max_nodes": (220, "cap for real data")})
def graph_force(ctx):
    """Glowing-edge force graph: clusters on a jittered grid so no hub forms, curved bloom edges, dots sized mildly by degree, a few accent nodes (one per region). ctx.data: a graph (groups or detected communities)."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    aspect = W / H
    g = _net(ctx, int(ctx.p("nodes")), int(ctx.p("clusters")), 0.30, 0.0)
    p = _spread_layout(g, rng, aspect)
    px = _map_view(p, W, H, aspect)
    if ctx.data is None:                               # short bridges between neighbouring clusters
        extra = []
        for i in range(g.n):
            if rng.random() < 0.10:
                d = np.hypot(*(px - px[i]).T); d[g.group == g.group[i]] = 1e9
                extra.append((i, int(np.argmin(d))))
        g.edges = np.vstack([g.edges, np.array(extra, int).reshape(-1, 2)]); g.eweight = np.ones(len(g.edges))
    cv = gk.Canvas((W, H), pal.canvas)
    ew = max(3.0, W * 0.0036)
    glow = cv.mask(); core_in = gk.Batch(cv); core_out = gk.Batch(cv)
    ng = int(g.group.max()) + 1
    for k, (a, b) in enumerate(g.edges):
        same = g.group[a] == g.group[b]
        pts = curved(px[a], px[b], (rng.random() - 0.5) * (0.35 if same else 0.5))
        calpha = float(np.clip(150 / max(g.m, 1), 0.25, 0.7))
        if same:
            gk.m_line(core_in.mask(gk.group_tone(pal, int(g.group[a]), ng, 0.38, 0.54)), pts, ew)
        else:
            gk.m_line(core_out.mask(gk.tone(pal, 0.50), calpha), pts, ew * 0.7)
        if same:
            gk.m_line(glow, pts, ew)
    cv.glow(glow, gk.tone(pal, 0.5), W * 0.006, 0.5)
    core_in.flush(); core_out.flush()
    deg = g.degree(); dmax = max(deg.max(), 1)
    nb = gk.Batch(cv)
    best = {}
    for i in range(g.n):                      # one accent node per grid region
        key = (int(px[i, 0] // (W / 4)), int(px[i, 1] // (H / 3)))
        if 0 <= px[i, 0] < W and 0 <= px[i, 1] < H and (key not in best or deg[i] > deg[best[key]]):
            best[key] = i
    hot = set(best.values())
    for i in range(g.n):
        r = W * (0.0075 + 0.0085 * min(1.0, deg[i] / (0.5 * dmax + 1e-9)))
        col = gk.group_tone(pal, int(g.group[i]), ng, 0.46, 0.68)
        if i in hot:
            gk.m_disc(nb.mask(pal.accent, 1.0, 2), px[i], r * 1.35)
        else:
            gk.m_disc(nb.mask(pal.canvas, 1.0, 0), px[i], r * 1.5)
            gk.m_disc(nb.mask(col, 1.0, 1), px[i], r)
    nb.flush()
    return cv.out()


# ------------------------------------------------------------------------------------------------------------------
def _poisson_points(rng, n, aspect, tries=14):
    """Best-candidate sampling: n points evenly spread over [0,aspect]x[0,1]."""
    pts = [rng.random(2) * [aspect, 1]]
    for _ in range(n - 1):
        cand = rng.random((tries, 2)) * [aspect, 1]
        d = np.min(np.linalg.norm(cand[:, None] - np.array(pts)[None], axis=2), axis=1)
        pts.append(cand[int(np.argmax(d))])
    return np.array(pts)


@effect("graph-constellation", family="graph", tags=["knowledge-graph", "graph", "all-over", "constellation", "clusters"],
        params={"nodes": (96, "synthetic star count"), "clusters": (10, "clusters (Voronoi cells of a jittered grid)"), "max_nodes": (220, "cap for real data")})
def graph_constellation(ctx):
    """Knowledge-graph constellation: evenly spread stars (best-candidate sampling), clusters = cells of a jittered grid, straight edges
    inside clusters, a few long cross-links, halo rings on larger stars, a micro-dot sky behind. No hub. ctx.data: a graph (real graphs use the cluster-anchored force layout)."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    aspect = W / H
    cv = gk.Canvas((W, H), pal.canvas)
    _dots(cv, pal, W / 38, tone_t=0.15, off=(rng.random() * 20, rng.random() * 20))
    if ctx.data is None:
        n = int(ctx.p("nodes")); k = int(ctx.p("clusters"))
        P = _poisson_points(rng, n, aspect)
        A = gk.grid_anchors(rng, k, aspect, 0.2)
        grp = np.argmin(np.linalg.norm(P[:, None] - A[None], axis=2), axis=1)
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        pairs = set()
        for i in range(n):
            same = np.where((grp == grp[i]) & (np.arange(n) != i))[0]
            if len(same):
                for j in same[np.argsort(D[i, same])[:2]]:
                    pairs.add((min(i, int(j)), max(i, int(j))))
            if rng.random() < 0.10:
                oth = np.where(grp != grp[i])[0]
                j = oth[np.argsort(D[i, oth])[int(rng.integers(0, 3))]]
                pairs.add((min(i, int(j)), max(i, int(j))))
        ids = [f"s{i}" for i in range(n)]
        g = gk._build(ids, [(ids[a], ids[b]) for a, b in pairs], {ids[i]: int(grp[i]) for i in range(n)}, directed=False)
        g.weight = np.exp(rng.normal(0, 0.55, n))
        px = _map_view(P, W, H, aspect, 0.05)
    else:
        g = _net(ctx, 0, 0, 0, 0)
        px = _map_view(_spread_layout(g, rng, aspect), W, H, aspect, 0.05)
    ng = int(g.group.max()) + 1
    ew = max(3.0, W * 0.0040)
    eb = gk.Batch(cv)
    for a, b in g.edges:
        same = g.group[a] == g.group[b]
        gk.m_line(eb.mask(gk.tone(pal, 0.40 if same else 0.30), 1.0), np.array([px[a], px[b]]), ew if same else ew * 0.8)
    eb.flush()
    nb = gk.Batch(cv); hm = cv.mask()
    w = g.weight / (np.percentile(g.weight, 90) + 1e-9)
    deg = g.degree()
    best = {}
    for i in range(g.n):
        key = (int(px[i, 0] // (W / 4)), int(px[i, 1] // (H / 3)))
        if 0 <= px[i, 0] < W and 0 <= px[i, 1] < H and (key not in best or deg[i] > deg[best[key]]):
            best[key] = i
    hot = set(best.values())
    for i in range(g.n):
        r = W * (0.006 + 0.010 * min(1.4, w[i])) * (1 + 0.04 * min(deg[i], 6))
        col = gk.group_tone(pal, int(g.group[i]), ng, 0.42, 0.66)
        gk.m_disc(nb.mask(pal.canvas, 1.0, 0), px[i], r * 1.7)
        if i in hot:
            gk.m_disc(nb.mask(pal.accent, 1.0, 2), px[i], r * 1.15); gk.m_disc(hm, px[i], r * 1.15)
            gk.m_ring(nb.mask(gk.tone(pal, 0.58), 1.0, 1), px[i], r * 1.75, max(2, W * 0.0025))
        else:
            gk.m_disc(nb.mask(col, 1.0, 1), px[i], r)
            if w[i] > 1.05:
                gk.m_ring(nb.mask(gk.tone(pal, 0.34), 1.0, 1), px[i], r * 1.7, max(2, W * 0.0025))
    nb.flush()
    cv.glow(hm, pal.accent, W * 0.008, 0.45)
    return cv.out()
