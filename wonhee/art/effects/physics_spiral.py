"""Spiral waves of the complex Ginzburg-Landau equation (an oscillatory
medium: chemical BZ reaction, cardiac tissue). Started from noise; defects
nucleate spiral cores everywhere. Colour = oscillation phase (cyclic bands),
darkened where the amplitude collapses at the cores."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from ..registry import effect
from .physics_common import fit, lookup, sim_size, source_lum


@effect("physics-spiral", family="physics", tags=["reaction", "physics", "all-over", "spiral", "simulation"],
        params={"alpha": (0.4, "dispersion of diffusion (spiral tightness)"), "beta": (-0.7, "nonlinear frequency shift"),
                "steps": (700, "time steps"), "cell": (15.0, "arm size in sim px (bigger = coarser spirals, fewer of them)"),
                "bands": (1.0, "phase cycles per colour cycle")})
def spiral(ctx):
    """source: luminance modulates the local frequency detuning (arms bend around bright areas)."""
    W, H = ctx.size
    rng = ctx.rng
    sw, sh = sim_size(ctx, 340)
    dx = 1.15 * 11.0 / float(ctx.p("cell"))
    kx = 2 * np.pi * np.fft.fftfreq(sw, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(sh, d=dx)
    k2 = (kx[None, :] ** 2 + ky[:, None] ** 2).astype(np.float32)
    al, be = float(ctx.p("alpha")), float(ctx.p("beta"))
    dt = 0.12
    lin = np.exp(dt * (1 - (1 + 1j * al) * k2)).astype(np.complex64)
    A = (rng.normal(0, 0.5, (sh, sw)) + 1j * rng.normal(0, 0.5, (sh, sw))).astype(np.complex64)
    lum = source_lum(ctx, sw, sh)
    det = None if lum is None else (0.35 * (lum - lum.mean())).astype(np.float32)
    for _ in range(int(ctx.p("steps"))):
        A = sfft.ifft2(sfft.fft2(A) * lin).astype(np.complex64)
        m2 = np.abs(A) ** 2
        s = 1 + 2 * m2 * dt
        A = A / np.sqrt(s) * np.exp(-1j * (be / 2) * np.log(s))
        if det is not None:
            A = A * np.exp(1j * det * dt)
    phase = np.angle(A) * float(ctx.p("bands"))
    amp = np.clip(np.abs(A), 0, 1)
    base = 0.5 + 0.5 * np.cos(phase)
    t = (0.06 + 0.94 * base ** 1.5) * (0.25 + 0.75 * np.clip(amp * 1.1, 0, 1) ** 0.5)
    return fit(lookup(t, ctx.palette), W, H)
