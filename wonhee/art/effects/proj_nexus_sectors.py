"""NEXUS card image: the original DYNAMIC KNOWLEDGE GRAPH radial (news centre, entity sectors, market halo) re-tiled.

Instead of one big radial with a bright hub, many small radials (hollow centres, so no hub) sit on a jittered offset lattice
that bleeds off every edge. Each radial keeps the original's anatomy: entity sectors whose angular widths follow the real
entity-subtype counts, a market ring outside, a news ring inside, thousands of fine edges (news -> entity -> market spokes
that stay inside their sector's angle). Sector widths come from ``ctx.data`` (a graph whose node groups are the entity
subtypes; the muksite nexusGraphData graph works), else from the NEXUS legend's 7 subtypes with equal weights.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import graphkit as gk
from ..registry import effect


def _sector_weights(ctx):
    if ctx.data is None:
        return np.ones(7)
    g = gk.as_graph(ctx.data)
    names = list(g.group_names) if g.group_names else []
    cnt = {}
    for i in range(g.n):
        nm = names[int(g.group[i])] if names else str(g.group[i])
        if nm in ("market", "event"):
            continue
        cnt[nm] = cnt.get(nm, 0) + 1
    w = np.array([cnt[k] for k in sorted(cnt)], float)
    return w if len(w) >= 3 else np.ones(7)


@effect("proj-nexus-sectors", family="graph", tags=["knowledge-graph", "graph", "all-over", "radial", "project"],
        params={"cell": (330, "lattice pitch in px at 1500 wide"), "dots": (2600, "dots per radial"),
                "spokes": (170, "fine edges per radial")})
def nexus_sectors(ctx):
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    s = W / 1500.0
    cell = float(ctx.p("cell")) * s
    nd, ne = int(ctx.p("dots")), int(ctx.p("spokes"))
    w = _sector_weights(ctx); w = w / w.sum(); ns = len(w)
    edges_a = np.concatenate([[0], np.cumsum(w)]) * 2 * math.pi
    cols = [gk.group_tone(pal, i, ns, 0.40, 0.68) for i in range(ns)]
    mk = gk.tone(pal, 0.36); nw = gk.tone(pal, 0.78)
    dots = np.zeros((H, W, 3), np.float32); lines = np.zeros((H, W, 3), np.float32)
    rows = int(H / (cell * 0.86)) + 3; ncol = int(W / cell) + 3
    for r in range(-1, rows):
        for c in range(-1, ncol):
            cx = (c + (0.5 if r % 2 else 0.0)) * cell + rng.uniform(-0.12, 0.12) * cell
            cy = r * cell * 0.86 + rng.uniform(-0.12, 0.12) * cell
            R = cell * rng.uniform(0.50, 0.60)
            rot = rng.uniform(0, 2 * math.pi)
            gap = 0.05
            # entity points by sector
            sec = rng.choice(ns, size=nd, p=w)
            lo, hi = edges_a[sec] + gap, edges_a[sec + 1] - gap
            ang = rot + lo + (hi - lo) * rng.beta(1.6, 1.6, nd)
            rad = R * (0.50 + 0.26 * np.sqrt(rng.random(nd)))
            ex, ey = cx + rad * np.cos(ang), cy + rad * np.sin(ang)
            # market halo (outer), news (inner, sparse and hollow-centred)
            ma = rng.uniform(0, 2 * math.pi, nd); mr = R * (0.84 + 0.16 * rng.random(nd) ** 0.8)
            mx, my = cx + mr * np.cos(ma), cy + mr * np.sin(ma)
            na = rng.uniform(0, 2 * math.pi, nd // 2); nr = R * (0.06 + 0.44 * np.sqrt(rng.random(nd // 2)))
            nx, ny = cx + nr * np.cos(na), cy + nr * np.sin(na)
            # spokes: news -> entity -> market, each stays near one sector's angle
            x0, y0, x1, y1 = int(cx - R * 1.1), int(cy - R * 1.1), int(cx + R * 1.1) + 1, int(cy + R * 1.1) + 1
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if xb <= xa or yb <= ya:
                continue
            m = np.zeros((yb - ya, xb - xa, 3), np.uint8)
            for k in range(ne):
                i = rng.integers(nd); a2 = ang[i] + rng.normal(0, 0.035)
                r0 = R * rng.uniform(0.30, 0.50); r1 = R * rng.uniform(0.88, 1.02)
                col = tuple(int(255 * v * 0.55) for v in cols[sec[i]][::-1])
                cv2.line(m, (int(cx + r0 * math.cos(a2) - xa), int(cy + r0 * math.sin(a2) - ya)),
                         (int(cx + r1 * math.cos(a2) - xa), int(cy + r1 * math.sin(a2) - ya)), col, 1, cv2.LINE_AA)
            lines[ya:yb, xa:xb] += m.astype(np.float32)[..., ::-1] / 255.0
            for X, Y, col, a in ((ex, ey, None, 0.9), (mx, my, mk, 0.55), (nx, ny, nw, 0.45)):
                xi, yi = np.round(X).astype(int), np.round(Y).astype(int)
                ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                for t in range(len(xi)):
                    if ok[t]:
                        cc = cols[sec[t]] if col is None else col
                        dots[yi[t], xi[t]] += cc * a
    dots = np.maximum(cv2.dilate(dots, np.ones((2, 2), np.uint8)), dots)
    img = pal.canvas + (1 - np.exp(-(dots * 1.3 + lines * 0.7))) * (1 - pal.canvas)
    img = img + cv2.GaussianBlur(dots, (0, 0), 2.2) * 0.35
    cap = 0.92
    return np.clip(img, 0, cap).astype(np.float32)
