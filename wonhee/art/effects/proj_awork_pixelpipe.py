"""awork card image: quadtree pixel waves with the project's real pipeline DAG tiled across the frame.

ctx.data: a graph (graphkit.as_graph shapes); without it a small stand-in chain DAG is used. The graph is laid out
(longest-path layering), tiled in `rows` bands (odd bands mirrored) and the longest chain is joined into ONE coral
serpentine line across the whole frame.
"""
from __future__ import annotations

import numpy as np

import dataclasses

from .. import compose, graphkit as gk
from ..palette import oklch_to_rgb, rgb_to_oklch
from ..registry import effect
from .graph_dag import expand_path


@effect("proj-awork-pixelpipe", family="graph", tags=["project", "awork", "dag", "pixel-waves", "all-over"],
        params={"rows": (3, "tiled DAG bands"), "cell": (0.06, "pixel-wave cell / width"), "depth": (2, "quartering levels"),
                "edge": (3.4, "edge width px at 1500"), "node": (0.30, "node radius / band height share per row slot"),
                "bg_alpha": (1.0, "pixel field strength"), "accent_hue": (28.0, "coral accent hue (oklch deg)"),
                "wave_deep": (0.55, "tone of the pixel field's densest cells (teal)")})
def pixelpipe(ctx):
    W, H = ctx.size
    pal = ctx.palette
    rng = ctx.rng
    La, Ca, _ = rgb_to_oklch(pal.accent)
    coral = oklch_to_rgb(0.62, 0.17, float(ctx.p("accent_hue"))).astype(np.float32)
    bgpal = dataclasses.replace(pal, accent=gk.tone(pal, float(ctx.p("wave_deep"))))
    bg = compose.render_effect("dots-pixel-waves", size=(W, H), seed=ctx.seed, palette=bgpal, source=ctx.source,
                               params={"cell": float(ctx.p("cell")), "depth": int(ctx.p("depth"))})
    cv = gk.Canvas((W, H), pal.canvas)
    cv.img[:] = pal.canvas + (bg - pal.canvas) * float(ctx.p("bg_alpha"))
    data = ctx.data if ctx.data is not None else {"nodes": [{"id": str(i)} for i in range(8)],
                                                  "edges": [{"source": str(i), "target": str(i + 1)} for i in range(7)]}
    g = gk.as_graph(data)
    lay = gk.layered(g, rng)
    chain = expand_path(lay, gk.longest_path(g))
    rows = int(ctx.p("rows"))
    bh = H / rows
    sc = W / 1500.0
    ew = float(ctx.p("edge")) * sc
    soft = pal.canvas + (pal.structure - pal.canvas) * 0.9
    dark = gk.tone(pal, 0.22)
    cb = gk.Batch(cv)       # casings (canvas colour halo so lines read over the squares)
    eb = gk.Batch(cv)
    nb = gk.Batch(cv)
    hot = cv.mask()
    hotn = []
    prev_end = None
    ng = int(g.group.max()) + 1
    for r in range(rows):
        y0 = -0.04 * bh + r * bh
        view = (-0.05 * W, y0, 1.05 * W, y0 + 1.08 * bh)
        px = gk.layered_to_px(lay, view)
        if r % 2:
            px = px.copy(); px[:, 0] = view[0] + view[2] - px[:, 0]
        hp = set(zip(chain[:-1], chain[1:]))
        for ch in lay.chains:
            for u, v in zip(ch[:-1], ch[1:]):
                pts = gk.s_curve(px[u], px[v], True, 24, 0.5)
                if (u, v) in hp:
                    gk.m_line(hot, pts, ew * 1.5)
                else:
                    gk.m_line(cb.mask(pal.canvas), pts, ew + 4 * sc)
                    gk.m_line(eb.mask(dark), pts, ew)
        for i in range(lay.n_real):
            c = px[i]
            rad = 0.062 * bh
            if i in chain:
                hotn.append((c, rad * 1.15))
            else:
                gk.m_disc(nb.mask(pal.canvas, 1.0, 0), c, rad + 3 * sc)
                gk.m_disc(nb.mask(dark, 1.0, 1), c, rad)
        a, z = px[chain[0]], px[chain[-1]]
        if prev_end is not None:
            gk.m_line(hot, gk.s_curve(prev_end, a, False, 30, 0.6), ew * 1.5)
        prev_end = z
    cb.flush(); eb.flush(); nb.flush()
    cv.paint(hot, coral, 1.0)
    cv.glow(hot, coral, ew * 2.0, 0.25)
    hb = gk.Batch(cv)
    for c, rad in hotn:
        gk.m_disc(hb.mask(pal.canvas, 1.0, 1), c, rad)
        gk.m_ring(hb.mask(coral, 1.0, 2), c, rad * 0.86, max(2.0, rad * 0.28))
    hb.flush()
    return cv.out()
