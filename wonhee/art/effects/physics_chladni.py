"""Chladni plate figures: a rectangular plate driven near one resonance. The
field is the superposition of all eigenmodes cos(m pi x) cos(n pi y) whose
frequency (m/A)^2 + n^2 lies in a narrow band; sand collects on the nodal
lines (u = 0), drawn as bright filaments over a two-tone plate."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, get_points, lookup, source_lum, smoothstep


@effect("physics-chladni", family="physics", tags=["wave", "physics", "all-over", "eigenmode", "nodal"],
        params={"order": (17.0, "resonance index: larger = finer figure (nodal lines per frame ~ order)"),
                "band": (0.7, "mode-selection half-width in frequency units; wider = more modes, busier"),
                "sand": (0.17, "nodal line width (fraction of rms amplitude)"),
                "warp": (0.035, "source image: coordinate warp strength (luminance bends the plate)")})
def chladni(ctx):
    """Chladni sand figure. data: (x, y, w) points in 0..1 = drive points (mode
    amplitude = response at those points). source: luminance warps the plate."""
    W, H = ctx.size
    rng = ctx.rng
    sw = min(W, 1000)
    sh = max(8, round(sw * H / W))
    A = W / H
    X = (np.arange(sw) + 0.5) / sw
    Y = (np.arange(sh) + 0.5) / sh
    Xg, Yg = np.meshgrid(X, Y)
    lum = source_lum(ctx, sw, sh)
    if lum is not None:
        lum = cv2.GaussianBlur(lum, (0, 0), sw / 80)
        Xg = Xg + float(ctx.p("warp")) * (lum - lum.mean())
        Yg = Yg + float(ctx.p("warp")) * (np.roll(lum, 7, 0) - lum.mean())
    order = float(ctx.p("order"))
    band = float(ctx.p("band"))
    modes = []
    mmax = int(order * A * 1.3) + 2
    nmax = int(order * 1.3) + 2
    for m in range(0, mmax):
        for n in range(0, nmax):
            if m == 0 and n == 0:
                continue
            if abs(np.hypot(m / A, n) - order) < band:
                modes.append((m, n))
    if len(modes) < 3:
        modes = [(int(order * A * 0.6), int(order * 0.8)), (int(order * 0.8), int(order * 0.6) + 1), (3, int(order))]
    xs, ys, ws = get_points(ctx, 0, rng) if ctx.data is not None else (None, None, None)
    u = np.zeros((sh, sw), np.float32)
    for m, n in modes:
        if xs is not None:
            a = float(np.sum(ws * np.cos(m * np.pi * xs) * np.cos(n * np.pi * ys)))
        else:
            a = rng.normal()
        u += a * np.cos(m * np.pi * Xg) * np.cos(n * np.pi * Yg)
    u /= u.std() + 1e-9
    line = np.exp(-(u / float(ctx.p("sand"))) ** 2)
    plate = 0.5 + 0.5 * np.tanh(1.4 * u)
    t = 0.08 + 0.32 * smoothstep(0.0, 1.0, plate) + 0.05 * np.tanh(np.abs(u)) + 0.62 * line
    return fit(lookup(np.clip(t, 0, 1), ctx.palette), W, H)
