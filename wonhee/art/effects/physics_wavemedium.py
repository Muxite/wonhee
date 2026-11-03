"""Waves in a heterogeneous medium: the 2D wave equation u_tt = c(x,y)^2 lap u
with a smooth random speed map. Wavelength shrinks where waves slow down and
fronts refract around the fast and slow patches, so the picture shows the
medium as much as the sources. Time-stepped (leapfrog), absorbing sponge."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, get_points, lookup, sim_size, smooth_noise, source_lum


@effect("physics-wavemedium", family="physics", tags=["wave", "physics", "all-over", "refraction", "simulation"],
        params={"sources": (6, "point sources (jittered grid)"), "contrast": (1.1, "speed-map contrast (0 = homogeneous medium)"),
                "wavelength": (0.07, "mean wavelength (fraction of frame height)"), "steps": (700, "time steps"), "patch": (0.2, "size of speed patches (fraction of height)")})
def wavemedium(ctx):
    """data: (x, y, w) points in 0..1 = sources (w amplitude). source: luminance IS the speed map (bright = fast)."""
    W, H = ctx.size
    rng = ctx.rng
    sw0, sh0 = sim_size(ctx, 420)
    pad = max(8, sh0 // 14)
    sw, sh = sw0 + 2 * pad, sh0 + 2 * pad
    xs, ys, ws = get_points(ctx, int(ctx.p("sources")), rng)
    xs = (xs * sw0 + pad) / sw; ys = (ys * sh0 + pad) / sh
    sp = smooth_noise(rng, sh, sw, sh * float(ctx.p("patch")))
    lum = source_lum(ctx, sw, sh)  # padded size
    if lum is not None:
        sp = (lum - lum.mean()) / (lum.std() + 1e-9)
    c = np.clip(1.0 + float(ctx.p("contrast")) * 0.5 * np.tanh(sp), 0.35, 1.8)
    c2 = (0.2 * c * c).astype(np.float32)          # courant number^2 <= 0.5 for stability
    lam_px = float(ctx.p("wavelength")) * sh
    freq = np.sqrt(0.2) / lam_px                  # cycles per step in the c=1 medium
    sponge = np.ones((sh, sw), np.float32)
    m = pad
    ramp = np.linspace(0, 1, m) ** 2
    for i in range(m):
        s = 1 - 0.09 * (1 - ramp[i])
        sponge[i, :] *= s; sponge[-1 - i, :] *= s; sponge[:, i] *= s; sponge[:, -1 - i] *= s
    u0 = np.zeros((sh, sw), np.float32); u1 = u0.copy()
    sx = np.clip((xs * sw).astype(int), 2, sw - 3); sy = np.clip((ys * sh).astype(int), 2, sh - 3)
    ph = rng.random(len(sx)) * 6.283
    env = np.zeros((sh, sw), np.float32)
    steps = int(ctx.p("steps"))
    for t in range(steps):
        lap = cv2.Laplacian(u1, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT)
        u2 = 2 * u1 - u0 + c2 * lap
        ramp_t = min(1.0, t / 80.0)
        for x, y, w, p in zip(sx, sy, ws, ph):
            u2[y - 1:y + 2, x - 1:x + 2] += ramp_t * 0.25 * w * np.sin(2 * np.pi * freq * t + p)
        u2 *= sponge
        u0, u1 = u1, u2
        if t > steps - 60:
            env = np.maximum(env, np.abs(u1))
    sl = (slice(pad, pad + sh0), slice(pad, pad + sw0))
    u1 = u1[sl]; c = c[sl]
    f = u1 / (np.percentile(np.abs(u1), 97) + 1e-9)
    t01 = 0.5 + 0.5 * np.tanh(1.5 * f)
    cn = (c - c.min()) / (np.ptp(c) + 1e-9)
    t = 0.03 + 0.8 * t01 ** 1.5 + 0.05 * cn
    return fit(lookup(np.clip(t, 0, 1), ctx.palette), W, H)
