"""dots-gradient-sweeps: a patchwork of dot gradients. Each BSP panel is its
own screen (square / hex / rotated lattice, round / square / diamond / line /
plus dots) whose dots sweep from tiny to full along a direction (and back),
like halftone gradient swatches. Panel lightness steps along the palette hue.

ctx.source (optional): luminance modulates the sweep in every other panel.
ctx.data unused."""
from __future__ import annotations

import math

import numpy as np

from .. import dotkit as dk
from ..registry import effect


@effect("dots-gradient-sweeps", family="dots", tags=["dots", "gradient", "halftone", "panels", "all-over"],
        params={"panels": (10, "number of panels"), "pitch": (0.016, "base dot pitch, fraction of width"), "cycles": (1.0, "sweeps per panel")})
def dots_gradient_sweeps(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    cv = dk.new_canvas(ctx.size, P.canvas)
    ramp = dk.palette_ramp(P, 64, ("edge", "structure", "accent"))
    src = dk.source_field(ctx, 1.0) if ctx.source is not None else None
    panels = dk.bsp_panels(ctx.size, rng, int(ctx.p("panels")), 0.2)
    g = max(3, int(W * 0.006))
    shapes = ["round", "square", "diamond", "line", "plus", "round"]
    lats = ["hex", "square", "rotated", "hex", "square"]
    for i, (x, y, w, h) in enumerate(panels):
        if w <= 2 * g or h <= 2 * g:
            continue
        w2, h2 = w - g, h - g
        a = rng.random() * math.pi
        yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float32)
        u = (xx * math.cos(a) + yy * math.sin(a)) / (abs(w2 * math.cos(a)) + abs(h2 * math.sin(a)))
        t = (u * float(ctx.p("cycles")) + rng.random()) % 1.0
        f = t if rng.random() < 0.6 else 1 - np.abs(2 * t - 1)
        if src is not None and i % 2 == 1:
            f = np.clip(0.6 * f + 0.4 * dk.norm01(src[y:y + h2, x:x + w2]), 0, 1)
        pitch = float(ctx.p("pitch")) * W * float(rng.choice([0.8, 1.0, 1.0, 1.3, 1.7]))
        shape = shapes[i % len(shapes)] if rng.random() < 0.75 else str(rng.choice(shapes))
        kind = lats[i % len(lats)]
        lat = dk.make_lattice(kind, (w2, h2), pitch, rng, angle=float(rng.choice([0, 15, 30, 45])) if kind != "hex" else 0)
        tcol = [0.5, 0.42, 0.58, 0.46, 0.64, 0.54][i % 6] if i % 5 != 3 else 0.74
        col = ramp[int(tcol * 63)]
        bgc = P.canvas + (P.structure - P.canvas) * (0.07 if i % 2 else 0.03)
        sub = dk.new_canvas((w2, h2), bgc)
        ang = np.full_like(f, a + math.pi / 2) if shape in ("line", "diamond", "square", "plus") else None
        dk.render_dots(f, (w2, h2), lat, canvas=sub, color=col, shape=shape, r_min=0.04, r_max=0.52, gamma=1.1, angle_by=ang, thick=0.5, normalize=False)
        cv[y + g:y + h, x + g:x + w] = sub if False else cv[y + g:y + h, x + g:x + w]
        cv[y:y + h2, x:x + w2] = sub
    return cv
