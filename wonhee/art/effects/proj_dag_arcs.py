"""proj-dag-arcs: the real Native DAG (ctx.data = nodes by job kind + parent edges from the euglena
trace fixtures) laid out with graphkit.layered, tiled edge to edge at two scales, and drawn as
plasma arcs: every edge is a flow curve whose path is perturbed by fractal noise and stroked three
times (wide halo, body, hot core). Nodes are discs sized by job kind (plan / search / visit /
verify / synthesize); cancelled (pruned) jobs are hollow dim rings. No labels.

ctx.data: {"nodes":[{"id","group","weight"?}],"edges":[{"source","target"}]}; weight < 0.5 = pruned."""
from __future__ import annotations

import numpy as np
import cv2

from .. import graphkit as gk
from ..registry import effect

SIZE = {"plan": 1.25, "search": 1.0, "visit": 0.8, "verify": 1.1, "synthesize": 1.4}
HOT = {"verify": 1.0, "synthesize": 0.9}


def _noise(rng, n, octaves=4):
    t = np.linspace(0, 1, n)
    out = np.zeros(n)
    for o in range(octaves):
        k = 3 * 2 ** o
        ph = rng.random(k + 2)
        out += np.interp(t * k, np.arange(k + 2), ph * 2 - 1) / (1.6 ** o)
    return out


def _arc(rng, pts, amp):
    p = np.asarray(pts, np.float32)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    n = max(24, int(seg.sum() / 5))
    s = np.r_[0, np.cumsum(seg)]
    u = np.linspace(0, s[-1], n)
    q = np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], 1)
    tg = np.gradient(q, axis=0)
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
    nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-6
    env = np.sin(np.linspace(0, np.pi, n)) ** 0.6
    return q + nr * (_noise(rng, n) * amp * env)[:, None]


@effect("proj-dag-arcs", family="graph", tags=["project", "dag", "plasma", "light", "all-over"],
        params={"tiles": (3, "copies of the DAG across the frame (small scale)"), "arc": (7.0, "plasma jitter amplitude px"),
                "halo": (9.0, "glow sigma px"), "big": (1, "also draw a large dim copy behind (0/1)")})
def proj_dag_arcs(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    g = gk.as_graph(ctx.data) if ctx.data is not None else gk.random_layered(rng, [1, 4, 4, 1])
    kinds = [str(g.group_names[int(k)]) if g.group_names else "visit" for k in g.group]
    pruned = np.asarray(g.weight if g.weight is not None else np.ones(g.n)) < 0.5
    cv = gk.Canvas(ctx.size, P.canvas)
    hal = gk.Canvas(ctx.size, P.canvas)
    scales = [(max(1, int(ctx.p("tiles"))), 1.0, 0.0)]
    if int(ctx.p("big")):
        scales.insert(0, (1, 2.3, 1.0))
    for si, (nt, sc, big) in enumerate(scales):
        tw = W / nt * 1.0
        th = tw * 0.5
        nr = int(np.ceil(H / th)) + 1
        for r in range(nr):
            for c in range(nt + 1):
                lay = gk.layered(g, np.random.default_rng(ctx.seed * 131 + si * 17 + r * 7 + c))
                ox = c * tw - tw * (0.1 + 0.12 * float(rng.random()))
                oy = r * th - th * (0.05 + 0.12 * float(rng.random()))
                vs = 0.8 + 0.4 * float(rng.random())
                view = (ox, oy, ox + tw * 1.02 * vs, oy + th * 0.92 * vs)
                if rng.random() < 0.5:
                    view = (view[2], view[1], view[0], view[3])
                if rng.random() < 0.5:
                    view = (view[0], view[3], view[2], view[1])
                if big:
                    view = (ox * sc - tw, oy * sc - th, ox * sc + tw * sc * 0.96 - tw, oy * sc + th * sc * 0.9 - th)
                px = gk.layered_to_px(lay, view)
                for chain in lay.chains:
                    path = gk.flow_path(px[chain], n=14)
                    pa = _arc(rng, path, float(ctx.p("arc")) * (1.7 if big else 1.0))
                    pr = pruned[chain[-1]] if chain[-1] < g.n else False
                    lev = 0.30 if big else (0.34 if pr else 0.5)
                    w = (6 if big else 3)
                    m = cv.mask(); gk.m_line(m, pa, w); 
                    col = gk.tone(P, lev + 0.1 * float(rng.random()))
                    cv.paint(m, col, 0.55 if big else 0.9)
                    if not big and not pr:
                        m2 = cv.mask(); gk.m_line(m2, _arc(rng, path, float(ctx.p("arc")) * 0.8), 1.6)
                        cv.paint(m2, gk.tone(P, 0.9), 0.95)
                        hal.paint(m2, gk.tone(P, 0.62), 1.0)
                        hal.img = hal.img
                for i in range(g.n):
                    ctr = px[i]
                    k = kinds[i]
                    rad = (14 if not big else 30) * SIZE.get(k, 1.0) * (tw / 500)
                    m = cv.mask()
                    if big:
                        gk.m_ring(m, ctr, rad * 0.5, max(2.0, rad * 0.1)); cv.paint(m, gk.tone(P, 0.33), 0.6)
                    elif pruned[i]:
                        gk.m_ring(m, ctr, rad, max(1.5, rad * 0.16)); cv.paint(m, gk.tone(P, 0.38), 0.8)
                    else:
                        gk.m_disc(m, ctr, rad); cv.paint(m, gk.tone(P, 0.46 if big else 0.55), 1.0)
                        mi = cv.mask(); gk.m_disc(mi, ctr, rad * 0.45)
                        cv.paint(mi, gk.tone(P, 1.0 if k in HOT else 0.8), 1.0)
                        if not big:
                            hal.paint(mi, gk.tone(P, 0.75), 1.0)
    img = cv.img.copy()
    # halo from the bright cores
    d = np.abs(hal.img - np.asarray(P.canvas, np.float32)).sum(2)
    core = (np.clip(d / (d.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
    b = cv2.GaussianBlur(core.astype(np.float32) / 255, (0, 0), float(ctx.p("halo")))[..., None]
    img += (gk.tone(P, 0.8) - img) * np.clip(b * 1.4, 0, 0.7)
    return np.clip(img, 0, 1).astype(np.float32)
