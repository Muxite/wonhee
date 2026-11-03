"""Reference effect: a 2D wave-equation ripple tank (the physics behind the
WaveSim card, which the owner called excellent). Several sources at spread
positions, stepped with the leapfrog finite-difference scheme, coloured by
field value through the palette ramp. Detail is spread over the whole frame
(sources are placed on a jittered grid, never centred)."""
from __future__ import annotations

import numpy as np

from ..registry import effect


@effect("wave-ripple", family="physics", tags=["wave", "physics", "all-over", "simulation"],
        params={"sources": (7, "number of point sources (jittered grid)"), "steps": (260, "time steps before the still"),
                "freq": (0.045, "source frequency (cycles per step)"), "scale": (4, "simulation downscale factor")})
def wave_ripple(ctx):
    W, H = ctx.size
    s = int(ctx.p("scale"))
    w, h = W // s, H // s
    rng = ctx.rng
    n = int(ctx.p("sources"))
    gx = int(np.ceil(np.sqrt(n * w / h))); gy = int(np.ceil(n / gx))
    cells = [(i, j) for i in range(gx) for j in range(gy)]
    rng.shuffle(cells)
    srcs = []
    for i, j in cells[:n]:
        x = int((i + 0.2 + 0.6 * rng.random()) / gx * w); y = int((j + 0.2 + 0.6 * rng.random()) / gy * h)
        srcs.append((x, y, 0.6 + 0.8 * rng.random(), 0.7 + 0.6 * rng.random(), rng.random() * 6.28))
    u0 = np.zeros((h, w), np.float32); u1 = np.zeros_like(u0)
    c2 = 0.24
    for t in range(int(ctx.p("steps"))):
        lap = (np.roll(u1, 1, 0) + np.roll(u1, -1, 0) + np.roll(u1, 1, 1) + np.roll(u1, -1, 1) - 4 * u1)
        u2 = 2 * u1 - u0 + c2 * lap
        for x, y, amp, fr, ph in srcs:
            u2[y, x] += amp * np.sin(2 * np.pi * ctx.p("freq") * fr * t + ph)
        u2 *= 0.9995
        u0, u1 = u1, u2
    field = u1 / (np.percentile(np.abs(u1), 98) + 1e-6)
    t01 = np.clip(0.5 + 0.5 * field, 0, 1)
    ramp = ctx.palette.ramp(256, ("canvas", "structure", "accent"))
    small = ramp[(t01 * 255).astype(int)]
    from scipy import ndimage
    big = ndimage.zoom(small, (H / h, W / w, 1), order=1)
    return big[:H, :W]
