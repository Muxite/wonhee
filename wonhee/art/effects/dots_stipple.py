"""dots-stipple: weighted Voronoi stipple (Lloyd-relaxed blue-noise dots) of
an image or, without one, of a multi-scale noise field with a few wave
ripples. Dots are uniform small discs whose DENSITY follows the field, like an
engraver's stipple; brightness follows density through the palette ramp.

ctx.source (optional): the picture to stipple (luminance). ctx.data (optional):
2D array used as the density field."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dk
from ..registry import effect


@effect("dots-stipple", family="dots", tags=["dots", "stipple", "voronoi", "source", "all-over"],
        params={"dots": (5500, "number of dots"), "size": (0.42, "dot radius as fraction of local spacing"),
                "contrast": (1.3, "density contrast (gamma)"), "invert": (False, "dense where the field is dark")})
def dots_stipple(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    if ctx.data is not None and np.ndim(ctx.data) == 2:
        f = dk.norm01(np.asarray(ctx.data, np.float32))
    elif ctx.source is not None:
        f = dk.norm01(dk.source_field(ctx, 1.5))
    else:
        wv = dk.wave_field((H // 4, W // 4), rng, 8, 0.2 * W / 4)
        f = dk.norm01(0.65 * dk.smooth_noise((H // 4, W // 4), 40, rng, 3) + 0.35 * dk.norm01(wv))
        f = dk.resize_field(f, ctx.size)
    if ctx.p("invert"):
        f = 1 - f
    f = 0.12 + 0.88 * np.power(f, float(ctx.p("contrast")))
    n = int(ctx.p("dots") * (W * H) / 1.5e6)
    n = max(n, 300)
    lat = dk.stipple(f, ctx.size, n, rng, iters=6)
    ramp = dk.palette_ramp(P)
    d = dk.sample(f, lat, ctx.size)
    cv = dk.render_dots(f, ctx.size, lat, bg=P.canvas, ramp=ramp, shape="round", size_const=True, r_max=float(ctx.p("size")), color_by=d ** 0.8, normalize=False)
    return cv
