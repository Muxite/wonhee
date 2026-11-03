"""dots-rings: several systems of polar dot rings spread over the frame (a
jittered grid of centres, never one centre). Every dot belongs to its nearest
system, so the systems tile like a mosaic of ripples. Dot size swells along
each ring set with a radial wave; colour follows a second field (a global wave)
through the palette ramp.

ctx.data (optional): list of (x, y) centres in 0..1 frame fractions.
ctx.source unused."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .. import dotkit as dk
from ..registry import effect


@effect("dots-rings", family="dots", tags=["dots", "polar", "rings", "ripple", "all-over"],
        params={"centers": (7, "number of ring systems"), "pitch": (0.020, "ring spacing, fraction of width"), "pulse": (3.2, "ring cycles of the size wave"),
                "shape": ("round", "round | square | diamond")})
def dots_rings(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    n = int(ctx.p("centers"))
    if ctx.data is not None and np.ndim(ctx.data) == 2 and np.shape(ctx.data)[1] == 2:
        C = np.asarray(ctx.data, np.float32) * [W, H]
    else:
        gx = int(math.ceil(math.sqrt(n * W / H))); gy = int(math.ceil(n / gx))
        cells = [(i, j) for i in range(gx) for j in range(gy)]
        rng.shuffle(cells)
        C = np.array([[(i + 0.2 + 0.6 * rng.random()) / gx * W, (j + 0.2 + 0.6 * rng.random()) / gy * H] for i, j in cells[:n]], np.float32)
    n = len(C)
    tree = cKDTree(C)
    parts, rv, ph = [], [], []
    rp = float(ctx.p("pitch")) * W
    for k, (cx, cy) in enumerate(C):
        pk = rp * (0.8 + 0.5 * rng.random())
        d = dk.polar(ctx.size, (cx, cy), pk, pk, phase=rng.random() * 6.28, twist=0.35 * (rng.random() - 0.5), r_min=0.0)
        _, nn = tree.query(np.stack([d.x, d.y], 1))
        d = d.subset(nn == k)
        r = np.hypot(d.x - cx, d.y - cy)
        parts.append(d)
        pulse = float(ctx.p("pulse")) * (0.7 + 0.6 * rng.random())
        rv.append(0.5 + 0.5 * np.cos(2 * math.pi * pulse * r / (pk * 18) + rng.random() * 6.28))
        ph.append(np.full(len(d), k / n) + 0.25 * np.sin(r / (pk * 5)))
    lat = dk.Dots.concat(parts)
    sv = np.concatenate(rv).astype(np.float32)
    cvv = np.concatenate(ph).astype(np.float32)
    ramp = dk.palette_ramp(P)
    return dk.render_dots(sv, ctx.size, lat, bg=P.canvas, ramp=ramp, shape=str(ctx.p("shape")), size_by=sv, color_by=cvv * 0.55 + 0.45 * sv,
                          r_min=0.12, r_max=0.48, normalize=False)
