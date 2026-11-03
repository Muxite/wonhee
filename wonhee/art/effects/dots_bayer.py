"""dots-bayer: ordered-dither (Bayer matrix) panels. The frame is split into
rectangles; each is a sawtooth/triangle gradient at its own angle, dithered
with a 2/4/8 Bayer matrix at its own pixel size, in its own lightness of the
palette hue. Some panels use error diffusion for a contrast of grain. Like a
1-bit graphics-card test card, evenly busy everywhere.

ctx.source (optional): shown dithered in one panel in three (luminance).
ctx.data unused."""
from __future__ import annotations

import math

import numpy as np

from .. import dotkit as dk
from ..registry import effect


@effect("dots-bayer", family="dots", tags=["dots", "bayer", "dither", "1-bit", "panels", "all-over"],
        params={"panels": (11, "number of panels"), "pixel": (0.0085, "dither pixel, fraction of width"), "cycles": (1.6, "gradient cycles per panel"),
                "diffuse": (0.15, "share of panels using error diffusion")})
def dots_bayer(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    cv = dk.new_canvas(ctx.size, P.canvas)
    px = max(2.0, float(ctx.p("pixel")) * W)
    panels = dk.bsp_panels(ctx.size, rng, int(ctx.p("panels")), 0.2)
    ramp = dk.palette_ramp(P, 64, ("canvas", "structure", "accent"))
    src = dk.source_field(ctx, 1.0) if ctx.source is not None else None
    for i, (x, y, w, h) in enumerate(panels):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.random() * math.pi
        u = (xx * math.cos(a) + yy * math.sin(a)) / (abs(w * math.cos(a)) + abs(h * math.sin(a)))
        tri = np.abs(((u * float(ctx.p("cycles")) + rng.random()) % 1.0) * 2 - 1)
        f = tri if rng.random() < 0.5 else (u * float(ctx.p("cycles")) + rng.random()) % 1.0
        f = f * 0.9 + 0.05
        if src is not None and i % 3 == 0:
            f = dk.norm01(src[y:y + h, x:x + w])
        mode = "fs" if rng.random() < float(ctx.p("diffuse")) else "bayer"
        cell = px * float(rng.choice([1.0, 1.0, 1.5, 2.0]))
        bn = int(rng.choice([2, 4, 8, 8]))
        m = dk.halftone_mask(f, (w, h), mode, cell=cell, bayer_n=bn, normalize=False)
        # panel colour: lightness band of the ramp (structure family, a few accents)
        t = [0.45, 0.55, 0.65, 0.5, 0.7, 0.6][i % 6] if i % 5 != 4 else 0.97
        fg = ramp[int(t * 63)]
        bgc = P.canvas + (P.structure - P.canvas) * (0.10 if i % 2 else 0.04)
        panel = dk.mix_mask(m, bgc, fg)
        cv[y:y + h, x:x + w] = panel
    # hairline gutters in canvas colour
    g = max(2, int(W * 0.004))
    for (x, y, w, h) in panels:
        cv[y:y + g, x:x + w] = P.canvas
        cv[y:y + h, x:x + g] = P.canvas
    return cv
