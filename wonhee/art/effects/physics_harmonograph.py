"""Harmonographs: damped coupled-pendulum curves x = sum A sin(f t + p) e^{-d t},
one per tile of a grid that covers the frame, each with its own near-rational
frequency ratio (rosettes, lace, knots). Density accumulated with sub-pixel
splatting and a two-scale glow."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, lookup


@effect("physics-harmonograph", family="physics", tags=["oscillator", "physics", "all-over", "lissajous", "tiles"],
        params={"tiles": (15, "number of harmonograph tiles"), "samples": (90000, "samples per curve"),
                "detune": (0.012, "frequency detuning (higher = faster rotating rosettes)"), "decay": (0.5, "damping (higher = tighter spiral-in)"),
                "gap": (0.05, "gap between tiles as a fraction of tile size")})
def harmonograph(ctx):
    """data: a list of numbers (e.g. one series per tile) sets each tile's frequency-ratio seed; length = number of tiles."""
    W, H = ctx.size
    rng = ctx.rng
    n = int(ctx.p("tiles"))
    if ctx.data is not None and not np.isscalar(ctx.data) and len(np.asarray(ctx.data).shape) == 1:
        d = np.asarray(ctx.data, float); n = len(d)
    else:
        d = None
    cols = int(round(np.sqrt(n * W / H)))
    cols = max(1, cols)
    rows = int(np.ceil(n / cols))
    cw, ch = W / cols, H / rows
    S = int(ctx.p("samples"))
    dens = np.zeros((H, W), np.float32)
    ratios = [(1, 2), (2, 3), (3, 4), (3, 5), (1, 3), (4, 5), (2, 5), (5, 6), (3, 7), (5, 7)]
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        sub = np.random.default_rng(ctx.seed * 1000 + (i if d is None else int(abs(d[i % len(d)]) * 1e4) % 997 + i))
        a, b = ratios[sub.integers(len(ratios))]
        det = float(ctx.p("detune"))
        f = np.array([a, a + det * (1 + sub.random()), b, b + det * (1 + sub.random())]) * (3.0 + sub.random())
        ph = sub.random(4) * 6.283
        dk = float(ctx.p("decay")) * (0.6 + 0.8 * sub.random(4))
        T = 4.0 / dk.min()
        t = np.linspace(0, T, S)
        T_scale = 1.0
        x = np.sin(f[0] * t + ph[0]) * np.exp(-dk[0] * t * 0.5) + np.sin(f[1] * t + ph[1]) * np.exp(-dk[1] * t * 0.5)
        y = np.sin(f[2] * t + ph[2]) * np.exp(-dk[2] * t * 0.5) + np.sin(f[3] * t + ph[3]) * np.exp(-dk[3] * t * 0.5)
        rx, ry = np.abs(x).max(), np.abs(y).max()
        m = 0.5 * (1 - float(ctx.p("gap")))
        px = c * cw + cw / 2 + x / max(rx, 1e-6) * cw * m
        py = r * ch + ch / 2 + y / max(ry, 1e-6) * ch * m
        # speed-normalised weight so slow parts of the curve do not saturate
        sp = np.hypot(np.gradient(px), np.gradient(py)) + 0.05
        wgt = (sp / sp.mean()).astype(np.float32)
        ix, iy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
        fx, fy = (px - ix).astype(np.float32), (py - iy).astype(np.float32)
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            jx, jy = ix + dx, iy + dy
            ok = (jx >= 0) & (jx < W) & (jy >= 0) & (jy < H)
            dens += np.bincount(jy[ok] * W + jx[ok], weights=(ww * wgt)[ok] * (1.0 if True else 0), minlength=W * H).reshape(H, W).astype(np.float32)
    dens = dens / (np.percentile(dens[dens > 0], 90) if (dens > 0).any() else 1)
    sharp = cv2.GaussianBlur(dens, (0, 0), 0.8)
    glow = cv2.GaussianBlur(dens, (0, 0), 5.0) * 1.5 + cv2.GaussianBlur(dens, (0, 0), 14.0) * 1.2
    core = 1 - np.exp(-1.1 * sharp)
    t = 0.05 + 0.40 * (1 - np.exp(-glow * 0.9)) + 0.62 * core
    # faint tile shading keeps empty tile interiors from reading as dead space
    return lookup(np.clip(t, 0, 1), ctx.palette).astype(np.float32)
