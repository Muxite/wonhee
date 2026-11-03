"""Swift-Hohenberg pattern formation (Rayleigh-Benard convection / Faraday
patterns): u_t = r u - (1+lap)^2 u + g u^2 - u^3. A spatially varying quadratic
term g makes hexagon cells melt into stripes with dislocations."""
from __future__ import annotations

import numpy as np

from ..registry import effect
from .physics_common import fit, lookup, sim_size, smooth_noise, source_lum


@effect("physics-swift-hohenberg", family="physics", tags=["reaction", "physics", "all-over", "turing", "convection", "simulation"],
        params={"pattern": ("mixed", "hex | stripes | mixed"), "wavelength": (0.034, "pattern wavelength (fraction of frame height)"),
                "r": (0.35, "control parameter (distance above onset)"), "steps": (420, "time steps")})
def swift_hohenberg(ctx):
    """source: luminance sets the hex-vs-stripe parameter g (bright = hexagons)."""
    W, H = ctx.size
    rng = ctx.rng
    sw, sh = sim_size(ctx, 420)
    lam_px = max(6.0, float(ctx.p("wavelength")) * sh)
    dx = 2 * np.pi / lam_px
    kx = 2 * np.pi * np.fft.fftfreq(sw, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(sh, d=dx)
    k2 = (kx[None, :] ** 2 + ky[:, None] ** 2).astype(np.float32)
    r = float(ctx.p("r"))
    dt = 0.8
    den = (1 + dt * ((1 - k2) ** 2 - r)).astype(np.float32)
    mode = str(ctx.p("pattern"))
    if mode == "hex":
        g = np.full((sh, sw), 1.0, np.float32)
    elif mode == "stripes":
        g = np.zeros((sh, sw), np.float32)
    else:
        n = smooth_noise(rng, sh, sw, sh / 5)
        g = np.clip(0.55 + 0.55 * n, 0, 1.2).astype(np.float32)
    lum = source_lum(ctx, sw, sh)
    if lum is not None:
        g = (1.1 * (lum - lum.min()) / (np.ptp(lum) + 1e-6)).astype(np.float32)
    u = (0.1 * rng.normal(0, 1, (sh, sw))).astype(np.float32)
    for _ in range(int(ctx.p("steps"))):
        N = g * u * u - u ** 3
        u = np.fft.ifft2((np.fft.fft2(u) + dt * np.fft.fft2(N)) / den).real.astype(np.float32)
    u /= u.std() + 1e-9
    t = 0.5 + 0.5 * np.tanh(1.2 * u)
    t = 0.05 + 0.95 * t ** 1.3
    return fit(lookup(fit(t, W, H), ctx.palette), W, H)
