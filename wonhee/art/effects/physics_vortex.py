"""2D incompressible vortex dynamics (Euler / Navier-Stokes, pseudo-spectral):
many Gaussian vortices of both signs on a periodic domain stir a striped dye.
Vortices roll the stripes into spirals, merge and shear (Kelvin-Helmholtz
roll-up between opposite-sign neighbours). Semi-Lagrangian advection."""
from __future__ import annotations

import cv2
import numpy as np
import scipy.fft as sfft

from ..registry import effect
from .physics_common import fit, get_points, lookup, sim_size, source_lum


@effect("physics-vortex", family="physics", tags=["fluid", "physics", "all-over", "vortex", "simulation"],
        params={"vortices": (26, "number of vortices (jittered grid, mixed sign)"), "steps": (55, "time steps"),
                "stripes": (14.0, "dye stripes per frame height"), "strength": (1.0, "vortex strength (more = more winding)"),
                "radius": (0.085, "vortex core radius (fraction of height)")})
def vortex(ctx):
    """data: (x, y, w) in 0..1 = vortices (w = signed circulation). source: its luminance is the dye (recoloured through the palette), so the image is stirred and wound into spirals."""
    W, H = ctx.size
    rng = ctx.rng
    sw, sh = sim_size(ctx, 560)
    xs, ys, ws = get_points(ctx, int(ctx.p("vortices")), rng, signed=True)
    X, Y = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
    R = float(ctx.p("radius")) * sh
    w = np.zeros((sh, sw), np.float32)
    for x, y, g in zip(xs * sw, ys * sh, ws):
        for ox in (-sw, 0, sw):
            for oy in (-sh, 0, sh):
                w += g * np.exp(-((X - x - ox) ** 2 + (Y - y - oy) ** 2) / (2 * R * R)).astype(np.float32)
    w *= float(ctx.p("strength"))
    kx = 2 * np.pi * sfft.fftfreq(sw)[None, :]; ky = 2 * np.pi * sfft.fftfreq(sh)[:, None]
    k2 = (kx ** 2 + ky ** 2).astype(np.float32); k2[0, 0] = 1
    nu = 0.0015
    decay = np.exp(-nu * k2 * 20).astype(np.float32)
    lum = source_lum(ctx, sw, sh)
    n = float(ctx.p("stripes"))
    if lum is not None:
        dye = lum.copy()
    else:
        dye = (0.5 + 0.5 * np.sin(2 * np.pi * n * (Y / sh))).astype(np.float32)
    # velocity scale so a unit vortex rotates ~ 1.5 rad over the run
    wh = sfft.fft2(w)
    def vel(wh):
        ps = -wh / k2
        u = sfft.ifft2(1j * ky * ps).real.astype(np.float32)   # u = dpsi/dy
        v = (-sfft.ifft2(1j * kx * ps).real).astype(np.float32)
        return u, v
    u0, v0 = vel(wh)
    scale = 0.8 * R / (np.abs(np.hypot(u0, v0)).max() + 1e-9) * 1.0
    steps = int(ctx.p("steps"))
    for _ in range(steps):
        u, v = vel(wh)
        mx = (X - u * scale * 1.3).astype(np.float32)
        my = (Y - v * scale * 1.3).astype(np.float32)
        dye = cv2.remap(dye, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        wr = cv2.remap(sfft.ifft2(wh).real.astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        wh = sfft.fft2(wr) * decay
    t = 0.06 + 0.78 * np.clip(dye, 0, 1)
    return fit(lookup(t, ctx.palette), W, H)
