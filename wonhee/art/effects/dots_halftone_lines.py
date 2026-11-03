"""dots-halftone-lines: a line-screen halftone. Parallel lines whose width
follows a field and whose path is bent by a second warp field, so the screen
swells and thins over the whole frame (engraving / banknote look). Line colour
steps across a few lightness bands of the palette hue; a thin share of the
swells take the accent.

ctx.source (optional): luminance drives the line width. ctx.data (optional):
2D array used as the width field."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dk
from ..registry import effect


@effect("dots-halftone-lines", family="dots", tags=["dots", "halftone", "lines", "engraving", "all-over"],
        params={"pitch": (0.012, "line spacing, fraction of width"), "angle": (18.0, "screen angle, degrees"),
                "warp": (1.6, "line bend in line-spacings"), "mode": ("line", "line | round | hex"), "gamma": (1.0, "tone gamma")})
def dots_halftone_lines(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    cell = float(ctx.p("pitch")) * W
    s = 3
    w = dk.wave_field((H // s, W // s), rng, 6, 0.2 * W / s)
    n1 = dk.smooth_noise((H // s, W // s), 55, rng, 3)
    f = dk.norm01(0.55 * dk.norm01(w) + 0.45 * n1)
    if ctx.data is not None and np.ndim(ctx.data) == 2:
        f = dk.norm01(np.asarray(ctx.data, np.float32))
    elif ctx.source is not None:
        f = dk.norm01(dk.source_field(ctx, 1.0))
    warp = (dk.norm01(dk.smooth_noise((H // s, W // s), 90, rng, 2)) - 0.5) * 2 * float(ctx.p("warp")) * cell * 2.5
    mode = str(ctx.p("mode"))
    m = dk.halftone_mask(f, ctx.size, mode, cell=cell, angle=float(ctx.p("angle")), gamma=float(ctx.p("gamma")), thickness=1.05, warp=warp if mode == "line" else None)
    # colour by slow noise through the dim->structure->accent ramp
    cn = dk.resize_field(dk.norm01(dk.smooth_noise((H // s, W // s), 130, rng, 2)), ctx.size)
    ramp = dk.palette_ramp(P)
    t = 0.15 + 0.7 * cn
    t = np.where(cn > 0.9, 0.97, t * 0.8)
    fg = dk.ramp_lookup(ramp, t)
    return dk.mix_mask(m, P.canvas, fg)
