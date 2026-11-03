"""Long-exposure light painting: bundles of strands following smooth random
curves across the whole frame, summed additively like light on a sensor."""
from __future__ import annotations

import math

import numpy as np

from ..registry import effect
from . import light_common as lc


@effect("light-ribbons", family="light", tags=["light", "long-exposure", "all-over", "ribbons"],
        params={"ribbons": (15, "number of ribbons, stratified over the frame height"),
                "strands": (34, "strands per ribbon (bundle thickness)"),
                "swing": (0.16, "curve amplitude as a fraction of frame height"),
                "twist": (1.6, "bundle pinch/twist count along a ribbon"),
                "exposure": (1.0, "overall exposure"),
                "fringe": (0.25, "chromatic fringe 0..1")})
def light_ribbons(ctx):
    """Light painting. Each ribbon is a bundle of strands around a smooth
    random centre curve (sum of sines, random tilt); the bundle pinches and
    opens (twist). Strand brightness ~ 1/speed along the curve (a slow hand
    exposes more), faded at the ends. data: points steer ribbon anchors
    (one ribbon passes through each y of the point set); source: luminance
    scales exposure per ribbon (a ribbon over bright image glows more)."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    n = int(ctx.p("ribbons")); K = max(8, int(int(ctx.p("strands")) * min(1.0, sc * 1.5)))
    buf = np.zeros((H, W), np.float32)
    acc = lc.SegAcc()
    lum = lc.source_lum(ctx)
    dp = lc.data_points(ctx)
    ys = (np.arange(n) + 0.5) / n * H + (rng.random(n) - 0.5) * H / n * 0.6
    if dp is not None:
        ys = np.sort(np.resize(dp[0][:, 1], n))
    U = 900
    u = np.linspace(0, 1, U)
    for i in range(n):
        th = rng.normal(0, 0.22)
        L = W * 1.5
        cx, cy = W * (0.5 + (rng.random() - 0.5) * 0.15), ys[i]
        tx, ty = math.cos(th), math.sin(th)
        nx, ny = -ty, tx
        off = np.zeros(U)
        for _ in range(3):
            f = rng.uniform(0.5, 2.4); a = ctx.p("swing") * H * rng.uniform(0.35, 1.0) / (1 + 0.5 * _)
            off += a * np.sin(2 * np.pi * f * u + rng.uniform(0, 6.28))
        base_x = cx + (u - 0.5) * L * tx + off * nx
        base_y = cy + (u - 0.5) * L * ty + off * ny
        # bundle spread
        g = ctx.p("twist"); ph = rng.uniform(0, 6.28)
        spread = (0.03 + 0.97 * np.abs(np.sin(np.pi * g * u + ph)) ** 1.5) * rng.uniform(0.5, 1.0) * H * 0.045
        # speed along the curve
        dx, dy = np.gradient(base_x), np.gradient(base_y)
        sp = np.hypot(dx, dy) * U / L
        env = np.clip(np.minimum(u, 1 - u) * 6, 0, 1) ** 1.5
        k = (np.arange(K) / (K - 1) * 2 - 1)
        k = np.sign(k) * np.abs(k) ** 1.2
        wob = rng.uniform(0.0, 0.25)
        gain = 0.018 * rng.uniform(0.6, 1.4)
        if lum is not None:
            gain *= 0.4 + 1.6 * float(lum[np.clip(int(cy), 0, H - 1)].mean())
        for j in range(K):
            o = k[j] * spread * (1 + wob * np.sin(2 * np.pi * (u * rng.uniform(2, 6)) + rng.uniform(0, 6.28)))
            # normal of the base curve
            nxu, nyu = -dy / np.maximum(np.hypot(dx, dy), 1e-6), dx / np.maximum(np.hypot(dx, dy), 1e-6)
            px = base_x + nxu * o; py = base_y + nyu * o
            w = gain * env / np.maximum(sp, 0.4)
            acc.polyline(np.stack([px, py], 1), w * 6)
    acc.flush(buf, 0.8)
    img = lc.develop(ctx, buf * 1.0, exposure=2.2 * ctx.p("exposure"), glow=0.9, glow_sigmas=(1.6, 6, 20),
                     fringe=float(ctx.p("fringe")), base=lc.ground(ctx))
    return img
