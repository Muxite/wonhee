"""dots-dash-flow: a needle / dash field. Every site of a square lattice holds
a short line segment ('dot' shape = line) rotated to the local direction of a
wave field's gradient and sized by its amplitude, so the interference pattern
appears as flowing hatch strokes. Colour by phase.

ctx.source (optional): the gradient of its luminance orients the dashes.
ctx.data (optional): 2D array used as the potential field."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dk
from ..registry import effect


@effect("dots-dash-flow", family="dots", tags=["dots", "line", "flow", "hatch", "wave", "all-over"],
        params={"pitch": (0.022, "lattice pitch, fraction of width"), "sources": (6, "wave sources"), "wavelength": (0.24, "wavelength, fraction of width"),
                "shape": ("line", "line | plus | diamond")})
def dots_dash_flow(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    s = 3
    if ctx.data is not None and np.ndim(ctx.data) == 2:
        pot = dk.resize_field(np.asarray(ctx.data, np.float32), (W // s, H // s))
    elif ctx.source is not None:
        pot = dk.resize_field(dk.source_field(ctx, 2.0), (W // s, H // s))
    else:
        pot = dk.wave_field((H // s, W // s), rng, int(ctx.p("sources")), float(ctx.p("wavelength")) * W / s)
    pot = dk.resize_field(pot, ctx.size)
    from scipy import ndimage
    ps = ndimage.gaussian_filter(pot, W * 0.004)
    gy, gx = np.gradient(ps)
    ang = np.arctan2(gy, gx) + np.pi / 2
    lat = dk.square(ctx.size, float(ctx.p("pitch")) * W, 0)
    amp = np.hypot(gx, gy)
    ramp = dk.palette_ramp(P)
    return dk.render_dots(amp, ctx.size, lat, bg=P.canvas, ramp=ramp, shape=str(ctx.p("shape")), size_by=amp, color_by=pot, angle_by=ang,
                          r_min=0.12, r_max=0.48, gamma=0.7, thick=0.28)
