"""Moeka: message traffic on the real module graph.

Long-exposure light of messages travelling the import edges of a real repo (``ctx.data``: any graph accepted by
graphkit.as_graph, e.g. ``from_repo('/home/muk/projects/moeka')``). The graph is laid out as spread clusters (no hub),
its edges are faint copper wire, and many seeded random walks (momentum, no immediate backtrack) are exposed onto the
same wires as additive trails that fade along their length, brightest on the graph's longest import chain.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import graphkit as gk
from ..registry import effect
from .graph_net import _map_view, _net, _spread_layout, curved


def _lerp3(L, c0, c1, c2):
    L = L[..., None]
    lo = c0 + (c1 - c0) * np.clip(L * 2.0, 0, 1)
    return lo + (c2 - c1) * np.clip(L * 2.0 - 1.0, 0, 1)


@effect("proj-moeka-traffic", family="graph", tags=["graph", "light", "all-over", "project", "moeka", "long-exposure"],
        params={"max_nodes": (110, "cap for real data"), "walks": (170, "message walks"), "steps": (9, "edges per walk"),
                "exposure": (0.28, "light exposure"), "spread": (0.85, "rank-equalisation of the force layout (even density)")})
def moeka_traffic(ctx):
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    aspect = W / H
    g = _net(ctx, 110, 10, 0.30, 0.0)
    p = gk.force(g, rng, iters=220, aspect=aspect, spread=float(ctx.p("spread"))) if ctx.data is not None else _spread_layout(g, rng, aspect)
    px = _map_view(p, W, H, aspect, bleed=0.07)
    cv = gk.Canvas((W, H), pal.canvas)
    nb = [[] for _ in range(g.n)]
    for a, b in g.edges:
        nb[a].append(b); nb[b].append(a)
    ew = max(2.0, W * 0.0016)
    # curved wire per edge, cached (both directions)
    wire = {}
    for a, b in g.edges:
        pts = curved(px[a], px[b], (rng.random() - 0.5) * 0.4)
        wire[(a, b)] = pts; wire[(b, a)] = pts[::-1]
    # faint wires
    wm = cv.mask()
    for (a, b), pts in wire.items():
        if a < b:
            gk.m_line(wm, pts, ew)
    cv.paint(wm, gk.tone(pal, 0.36), 0.8)
    # light exposure
    light = np.zeros((H, W), np.float32)
    deg = np.array([len(x) for x in nb])
    start_pool = np.where(deg > 0)[0]
    def trail(path, gain, th):
        pts = np.vstack([wire[(path[i], path[i + 1])] for i in range(len(path) - 1)])
        n = len(pts)
        seg = max(1, n // 24)
        for s in range(0, n - 1, seg):
            fade = (0.25 + 0.75 * (s / n)) ** 1.5       # head bright, tail faint
            tmp = np.zeros((H, W), np.uint8)
            cv2.polylines(tmp, [np.round(pts[s:s + seg + 1] * 8).astype(np.int32)], False, 255, int(th), cv2.LINE_AA, 3)
            x, y, w, h = cv2.boundingRect(tmp)
            if w and h:
                light[y:y + h, x:x + w] += tmp[y:y + h, x:x + w].astype(np.float32) * (gain * fade / 255.0)
    steps = int(ctx.p("steps"))
    for _ in range(int(ctx.p("walks"))):
        cur = int(rng.choice(start_pool)); prev = -1; path = [cur]
        for _s in range(steps):
            opts = [q for q in nb[cur] if q != prev] or nb[cur]
            if not opts:
                break
            nxt = int(rng.choice(opts)); prev, cur = cur, nxt; path.append(cur)
        if len(path) > 1:
            trail(path, 0.34 * ctx.p("exposure"), 2)
    try:
        lp = gk.longest_path(g)
        if len(lp) > 1:
            trail(lp, 1.6 * ctx.p("exposure"), 3)
    except Exception:
        pass
    # nodes: small, even
    nm = cv.mask()
    for i in range(g.n):
        gk.m_disc(nm, px[i], W * 0.0055)
    cv.paint(nm, gk.tone(pal, 0.46), 1.0)
    L = 1 - np.exp(-light * 1.1)
    wide = cv2.GaussianBlur(L, (0, 0), W * 0.006)
    tight = cv2.GaussianBlur(L, (0, 0), 1.2)
    Lm = np.clip(0.65 * tight + 0.9 * wide + 0.35 * L, 0, 1)
    img = cv.img
    lit = _lerp3(Lm, np.asarray(gk.tone(pal, 0.40), np.float32), np.asarray(gk.tone(pal, 0.75), np.float32), np.asarray(pal.accent, np.float32))
    a = np.clip(Lm * 1.4, 0, 1)[..., None]
    out = img * (1 - a) + (img + lit) * 0.5 * a + lit * a * 0.5
    return np.clip(out, 0, 1).astype(np.float32)
