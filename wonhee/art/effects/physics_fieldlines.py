"""Electrostatic field of many point charges spread over the frame, drawn as
line-integral-convolution streamlines (flow along E) overlaid with glowing
equipotential contours. Bright where lines are dense, dark in the far field."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, get_points, lookup, pnorm, sim_size


@effect("physics-fieldlines", family="physics", tags=["field", "physics", "all-over", "electrostatics", "lic"],
        params={"charges": (14, "number of point charges (jittered grid, mixed sign)"), "length": (40, "streamline half-length in steps"),
                "equipotentials": (0.15, "equipotential line strength (0 = off)"), "soft": (0.04, "core softening (fraction of height)")})
def fieldlines(ctx):
    """data: (x, y, q) in 0..1 = charges (q signed; positive/negative are drawn alike; sign sets line winding)."""
    W, H = ctx.size
    rng = ctx.rng
    sw, sh = sim_size(ctx, 900)
    xs, ys, qs = get_points(ctx, int(ctx.p("charges")), rng, signed=True)
    X, Y = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
    eps = float(ctx.p("soft")) * sh
    Ex = np.zeros((sh, sw), np.float32); Ey = np.zeros_like(Ex); phi = np.zeros_like(Ex)
    for x, y, q in zip(xs * sw, ys * sh, qs):
        dx, dy = X - x, Y - y
        r2 = dx * dx + dy * dy + eps * eps
        Ex += q * dx / r2; Ey += q * dy / r2
        phi += q * 0.5 * np.log(r2)
    mag = np.hypot(Ex, Ey) + 1e-9
    ux, uy = Ex / mag, Ey / mag
    noise = rng.random((sh, sw)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.3)
    acc = noise.copy(); wsum = np.ones_like(noise)
    L = int(ctx.p("length"))
    for sgn in (1, -1):
        px, py = X.copy(), Y.copy()
        for i in range(1, L + 1):
            vx = cv2.remap(ux, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            vy = cv2.remap(uy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            px = px + sgn * vx * 1.2; py = py + sgn * vy * 1.2
            wgt = np.exp(-(i / (0.55 * L)) ** 2)
            acc += wgt * cv2.remap(noise, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum += wgt
    lic = acc / wsum
    lic = (lic - lic.mean()) / (lic.std() + 1e-9)
    lic = 0.5 + 0.5 * np.tanh(1.5 * lic)
    strength = np.clip(np.log1p(mag * sh * 0.6) / 3.0, 0, 1)
    t = 0.08 + 0.72 * lic * (0.45 + 0.55 * pnorm(strength, 2, 98))
    eq = float(ctx.p("equipotentials"))
    if eq > 0:
        pp = phi / (np.ptp(phi) + 1e-9) * 22
        bands = np.exp(-((pp - np.round(pp)) / 0.11) ** 2)
        t = t + eq * 0.45 * bands
    return fit(lookup(np.clip(t, 0, 1), ctx.palette), W, H)
