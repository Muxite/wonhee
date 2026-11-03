"""Aurora curtains and long-exposure star trails."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..registry import effect
from . import light_common as lc


@effect("aurora-curtains", family="light", tags=["aurora", "all-over", "curtains", "atmosphere"],
        params={"curtains": (5, "stacked curtain layers (spread over the frame height)"),
                "rays": (150, "vertical ray density across the width"),
                "fold": (0.07, "curtain fold amplitude (fraction of height)"),
                "exposure": (1.0, "exposure")})
def aurora_curtains(ctx):
    """Aurora: each curtain is a wavy lower edge y_c(x) (sum of sines) with
    field-aligned rays (vertical striations from noise) whose brightness
    decays exponentially upward from the edge, as emission does along a
    magnetic field line, and with a thin bright lower rim. Several curtains
    sit at stratified heights so the whole frame is lit. Folds foreshorten
    the curtain (brightness ~ 1/|slope|). source: luminance modulates ray
    brightness along x; data: point x positions seed bright ray clusters."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    x = np.arange(W, dtype=np.float32)
    yy = np.arange(H, dtype=np.float32)[:, None]
    I = np.zeros((H, W), np.float32)
    n = int(ctx.p("curtains"))
    lum = lc.source_lum(ctx)
    dp = lc.data_points(ctx)
    for i in range(n):
        base = (i + 0.9) / (n + 0.3) * H
        yc = np.full(W, base, np.float32)
        for k in range(3):
            f = rng.uniform(0.6, 2.8) * (1 + k * 0.8)
            yc += ctx.p("fold") * H * rng.uniform(0.3, 1) / (1 + k) * np.sin(2 * np.pi * f * x / W + rng.uniform(0, 6.28))
        # rays: 1D noise with sharp ridges
        r = rng.standard_normal(W).astype(np.float32)
        r = cv2.GaussianBlur(r[None], (0, 0), max(1.0, W / float(ctx.p("rays")) * 0.5))[0]
        r2 = cv2.GaussianBlur(rng.standard_normal(W).astype(np.float32)[None], (0, 0), W / 22)[0]
        rays = np.clip(0.55 + 0.9 * r / (r.std() + 1e-6) + 0.7 * r2 / (r2.std() + 1e-6), 0, None) ** 2.2
        if lum is not None:
            rays = rays * (0.4 + 1.2 * lum.mean(0))
        if dp is not None:
            bump = np.zeros(W, np.float32)
            for px in dp[0][:, 0]:
                bump += np.exp(-((x - px) / (W * 0.03)) ** 2)
            rays = rays * (0.5 + bump)
        slope = np.abs(np.gradient(yc)) + 0.15
        bright = rays / np.sqrt(slope) * rng.uniform(0.6, 1.1)
        height = H * rng.uniform(0.16, 0.26) * (0.7 + 0.6 * np.clip(rays / (rays.mean() + 1e-6), 0, 2) / 2)
        d = yc[None, :] - yy                    # >0 above the lower edge
        fall = np.where(d > 0, np.exp(-d / height[None, :]), np.exp(-(-d) / (H * 0.012)))
        I += (bright[None, :] * fall * 0.10).astype(np.float32)
    # soft vertical smear for the veil
    I = cv2.GaussianBlur(I, (0, 0), 1.0 * sc)
    return lc.develop(ctx, I, exposure=0.95 * ctx.p("exposure"), glow=0.35, glow_sigmas=(4, 14, 40), base=lc.ground(ctx), grain=0.01)


@effect("star-trails", family="light", tags=["stars", "long-exposure", "all-over", "astro"],
        params={"stars": (2600, "number of stars"),
                "arc": (0.30, "exposure length as rotation angle in radians"),
                "pole": (1.6, "pole distance in frame widths (outside the frame, so arcs cross evenly)"),
                "exposure": (1.0, "exposure")})
def star_trails(ctx):
    """Earth's rotation turns every star into a circular arc about the
    celestial pole. The pole is placed outside the frame (default 1.6 frame
    widths away, angled), so the arcs sweep the whole frame like parallel
    ribbons. Star magnitudes follow a power law (many faint, few bright),
    trail brightness ~ flux, with colour from the palette ramp by flux.
    data: points are extra bright stars; source: luminance scales star flux
    (an image seen as night sky)."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    ang = rng.uniform(0, 2 * math.pi)
    # pole off the frame, direction random but biased off-axis
    pc = np.array([W / 2 + math.cos(ang) * W * float(ctx.p("pole")), H / 2 + math.sin(ang) * W * float(ctx.p("pole"))])
    n = int(int(ctx.p("stars")) * max(0.2, sc))
    pad = 0.5 * W
    P = np.stack([rng.uniform(-pad * 0.2, W + pad * 0.2, n), rng.uniform(-pad * 0.2, H + pad * 0.2, n)], 1)
    flux = rng.pareto(1.8, n) * 0.11 + 0.01
    dp = lc.data_points(ctx)
    if dp is not None:
        P = np.vstack([P, dp[0]]); flux = np.concatenate([flux, np.full(len(dp[0]), 2.5) * dp[2] / dp[2].mean()])
    lum = lc.source_lum(ctx)
    if lum is not None:
        ix = np.clip(P[:, 0].astype(int), 0, W - 1); iy = np.clip(P[:, 1].astype(int), 0, H - 1)
        flux = flux * (0.3 + 1.4 * lum[iy, ix])
    n = len(P)
    arc = float(ctx.p("arc"))
    r = P - pc
    th0 = np.arctan2(r[:, 1], r[:, 0]); rad = np.hypot(r[:, 0], r[:, 1])
    # sample each arc: points every ~0.7 px of arc length
    L = rad * arc
    cnt = np.maximum(2, np.ceil(L / 0.7).astype(int))
    idx = np.repeat(np.arange(n), cnt)
    first = np.cumsum(cnt) - cnt
    t = (np.arange(len(idx)) - first[idx]) / (cnt[idx] - 1)
    th = th0[idx] + arc * t
    xs = pc[0] + rad[idx] * np.cos(th); ys = pc[1] + rad[idx] * np.sin(th)
    # per-unit-length intensity: flux, with a gentle twinkle taper at the ends
    env = np.clip(np.minimum(t, 1 - t) * 8, 0, 1) ** 0.7
    buf = np.zeros((H, W), np.float32)
    lc.splat(buf, xs, ys, flux[idx] * env * (L[idx] / cnt[idx]) * 0.5 / 3.0)
    glowl = cv2.GaussianBlur(buf, (0, 0), 0.7 * sc)
    return lc.develop(ctx, glowl * 1.2, exposure=6.0 * ctx.p("exposure"), glow=0.6, glow_sigmas=(1.5, 5, 16), base=lc.ground(ctx), grain=0.01)
