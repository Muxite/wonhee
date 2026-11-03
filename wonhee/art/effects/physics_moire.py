"""Moire interference of line gratings: two or three families of curved
(radial / warped) fine lines, multiplied like overlaid transparencies. The
slow beat pattern between nearly equal spatial frequencies is the picture."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, lookup, smooth_noise, source_lum


@effect("physics-moire", family="physics", tags=["interference", "physics", "all-over", "moire", "optics"],
        params={"gratings": (2, "number of overlaid gratings"), "pitch": (0.02, "line pitch as fraction of frame height"),
                "warp": (0.9, "low-frequency warp of the lines (pitch units x 40)"), "duty": (0.5, "line duty cycle"),
                "image_warp": (0.05, "source image: line displacement (frame height) by luminance")})
def moire(ctx):
    """source: luminance displaces the lines (line-halftone), so the moire carries the picture."""
    W, H = ctx.size
    rng = ctx.rng
    sw = min(W, 1000)
    sh = max(8, round(sw * H / W))
    A = sw / sh
    Y, X = np.mgrid[0:sh, 0:sw].astype(np.float32)
    X = X / sh; Y = Y / sh
    pitch = float(ctx.p("pitch"))
    lum = source_lum(ctx, sw, sh)
    n = int(ctx.p("gratings"))
    T = np.ones((sh, sw), np.float32)
    duty = float(ctx.p("duty"))
    wsmooth = 1.2 * pitch
    for i in range(n):
        wx = smooth_noise(rng, sh, sw, sh * 0.22) * float(ctx.p("warp")) * pitch * 40 / 40
        wy = smooth_noise(rng, sh, sw, sh * 0.22) * float(ctx.p("warp")) * pitch * 40 / 40
        kind = i % 3
        p = pitch * (1 + 0.03 * (i - 1) + 0.02 * rng.normal())
        if kind == 0:                      # linear
            th = rng.random() * np.pi
            u = (X + wx) * np.cos(th) + (Y + wy) * np.sin(th)
        elif kind == 1:                    # radial around an off-frame centre
            cx, cy = A * (0.5 + rng.choice([-1, 1]) * (0.7 + 0.5 * rng.random())), 0.5 + rng.choice([-1, 1]) * (0.5 + 0.6 * rng.random())
            u = np.hypot(X + wx - cx, Y + wy - cy)
        else:                              # sinusoidal lines
            th = rng.random() * np.pi
            u = (X + wx) * np.cos(th) + (Y + wy) * np.sin(th) + 0.04 * np.sin(2 * np.pi * (X * np.sin(th) - Y * np.cos(th)) * 2.5 + rng.random() * 6)
        if lum is not None and i == 0:
            u = u + float(ctx.p("image_warp")) * (lum - lum.mean())
        c = 0.5 + 0.5 * np.cos(2 * np.pi * u / p)
        thr = np.cos(np.pi * duty)
        g = np.clip((c - (0.5 + 0.5 * thr)) / 0.14 + 0.5, 0, 1)
        T *= 1 - 0.85 * (1 - g)
    sm = cv2.GaussianBlur(T, (0, 0), max(1.0, pitch * sh * 1.4))
    from .physics_common import pnorm
    t = 0.10 + 0.82 * pnorm(0.25 * T + 0.75 * sm, 1, 99.5)
    return fit(lookup(np.clip(t, 0, 1), ctx.palette), W, H)
