"""dots-pixel-waves: the owner's 'pixel waves'. A quadtree of squares driven
by a wave-interference field: quiet regions stay as large sparse squares, busy
regions quarter into dense small ones. Fill, colour and alpha rise with the
field; the crests take the accent; troughs show as outlines. Wave sources are
spread over the frame, so density varies everywhere with no centre.

ctx.source (optional): its luminance replaces the wave field, giving a quadtree
mosaic of the picture. ctx.data (optional): 2D array used as the field."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import dotkit as dk
from ..registry import effect


@effect("dots-pixel-waves", family="dots", tags=["dots", "quadtree", "wave", "all-over", "owner-liked"],
        params={"cell": (0.07, "largest square, fraction of width"), "depth": (2, "max quartering levels"), "sources": (16, "wave sources"),
                "wavelength": (0.22, "wavelength, fraction of width"), "outline": (0.35, "share of quiet squares drawn as outlines")})
def dots_pixel_waves(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    f = None
    if ctx.data is not None and np.ndim(ctx.data) == 2:
        f = dk.norm01(dk.resize_field(np.asarray(ctx.data, np.float32), ctx.size))
    elif ctx.source is not None:
        f = dk.norm01(dk.source_field(ctx, 2.0))
    if f is None:
        w = dk.wave_field((H // 3, W // 3), rng, int(ctx.p("sources")), float(ctx.p("wavelength")) * W / 3, decay=0.0)
        f = dk.norm01(np.abs(w), 1, 99) ** 1.4
        f = dk.resize_field(f, ctx.size)
    maxc = float(ctx.p("cell")) * W
    minc = maxc / (2 ** int(ctx.p("depth")))
    q = dk.quadtree(f, ctx.size, maxc, minc * 0.99, rng)
    v = dk.sample(f, q, ctx.size)
    ramp = dk.palette_ramp(P)
    cv = dk.new_canvas(ctx.size, P.canvas)
    gap = 0.07
    quiet = v < 0.16
    outl = quiet & (rng.random(len(q)) < float(ctx.p("outline")))
    hide = quiet & ~outl & (rng.random(len(q)) < 0.4)
    col = dk.ramp_lookup(ramp, np.clip(v * 1.0, 0, 1) ** 0.9)
    r = q.cell * (0.5 - gap)
    # outlines: coloured square, then canvas square inside
    o = outl
    edge = P.canvas + (P.structure - P.canvas) * 0.45
    dk.draw_dots(cv, q.x[o], q.y[o], r[o], edge, 1.0, "square")
    dk.draw_dots(cv, q.x[o], q.y[o], r[o] - np.maximum(1.2, q.cell[o] * 0.06), P.canvas, 1.0, "square")
    fill = ~outl & ~hide
    al = 0.55 + 0.45 * v[fill]
    dk.draw_dots(cv, q.x[fill], q.y[fill], r[fill], col[fill], al, "square")
    return cv
