"""Many-source wave interference (analytic Huygens sum, no time stepping):
crests glow where the phase of all sources agrees. Modes: scattered points,
a hexagonal lattice (Talbot / moire-like carpets) and two facing slit arrays."""
from __future__ import annotations

import numpy as np

from ..registry import effect
from .physics_common import fit, get_points, lookup


@effect("physics-interference", family="physics", tags=["wave", "physics", "all-over", "interference", "analytic"],
        params={"mode": ("points", "points | lattice | slits"), "sources": (4, "number of point sources (points mode)"),
                "wavelength": (0.06, "wavelength as a fraction of frame height"),
                "gain": (2.0, "crest sharpening"), "spacing": (3.2, "lattice/slit spacing in wavelengths")})
def interference(ctx):
    """data: (x, y, w) in 0..1 = wave sources (w = amplitude, negative = antiphase)."""
    W, H = ctx.size
    rng = ctx.rng
    sw = min(W, 1100)
    sh = max(8, round(sw * H / W))
    lam = float(ctx.p("wavelength")) * sh
    k = 2 * np.pi / lam
    mode = str(ctx.p("mode"))
    if ctx.data is not None or mode == "points":
        xs, ys, ws = get_points(ctx, int(ctx.p("sources")), rng, signed=False)
        xs, ys = xs * sw, ys * sh
        ph = rng.random(len(xs)) * 6.283
    elif mode == "lattice":
        d = float(ctx.p("spacing")) * lam
        pts = []
        row = 0
        y = rng.random() * d
        while y < sh + d:
            x = (rng.random() * 0 + (0.5 * d if row % 2 else 0.0)) + 0.13 * d
            while x < sw + d:
                pts.append((x, y)); x += d
            y += d * 0.866; row += 1
        pts = np.array(pts)
        xs, ys = pts[:, 0], pts[:, 1]
        ws = np.ones(len(xs)); ph = np.zeros(len(xs))
    else:  # slits
        d = float(ctx.p("spacing")) * lam * 0.7
        xs_row = np.arange(-d, sw + d, d)
        xs = np.concatenate([xs_row, xs_row + 0.37 * d])
        ys = np.concatenate([np.full(len(xs_row), -0.03 * sh), np.full(len(xs_row), 1.03 * sh)])
        ws = np.ones(len(xs)); ph = np.concatenate([np.zeros(len(xs_row)), np.full(len(xs_row), 1.7)])
    X, Y = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
    f = np.zeros((sh, sw), np.float32)
    r0 = 0.12 * sh
    for x, y, w, p in zip(xs, ys, ws, ph):
        r = np.hypot(X - x, Y - y)
        f += (w / np.sqrt(1 + r / r0)).astype(np.float32) * np.cos(k * r - p)
    f /= f.std() + 1e-9
    g = float(ctx.p("gain"))
    t = 0.5 + 0.5 * np.tanh(g * f)
    t = 0.03 + 0.8 * t ** 1.4
    return fit(lookup(t, ctx.palette), W, H)
