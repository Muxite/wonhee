"""dots-hex-field: a hexagonal dot lattice driven by a wave-interference
field. Dot SIZE follows the wave amplitude, dot COLOUR follows the signed
phase (two-field control), so the interference pattern reads both as moire of
sizes and as a colour swell. Sources are spread over the frame.

ctx.source (optional): luminance drives the size instead of the waves (colour
still from the waves). ctx.data (optional): 2D array used as the size field."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dk
from ..registry import effect


@effect("dots-hex-field", family="dots", tags=["dots", "hex", "wave", "interference", "all-over"],
        params={"pitch": (0.018, "dot pitch, fraction of width"), "sources": (7, "wave sources"), "wavelength": (0.16, "wavelength, fraction of width"),
                "shape": ("round", "round | diamond | square"), "angle": (0.0, "lattice rotation, degrees")})
def dots_hex_field(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    s = 3
    w = dk.wave_field((H // s, W // s), rng, int(ctx.p("sources")), float(ctx.p("wavelength")) * W / s)
    w = dk.resize_field(w, ctx.size)
    size_f = np.abs(w)
    if ctx.data is not None and np.ndim(ctx.data) == 2:
        size_f = np.asarray(ctx.data, np.float32)
    elif ctx.source is not None:
        size_f = dk.source_field(ctx)
    lat = dk.hexagonal(ctx.size, float(ctx.p("pitch")) * W, float(ctx.p("angle")))
    ramp = dk.palette_ramp(P)
    return dk.render_dots(size_f, ctx.size, lat, bg=P.canvas, ramp=ramp, shape=str(ctx.p("shape")), color_by=w, size_by=size_f,
                          r_min=0.10, r_max=0.54, gamma=0.8)
