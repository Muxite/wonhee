"""Ray-based light: laser billiards (reflected beams) and a refractive caustic pool."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..registry import effect
from . import light_common as lc


@effect("laser-billiards", family="light", tags=["laser", "reflection", "billiards", "all-over", "physics"],
        params={"emitters": (9, "laser sources, spread over the frame"),
                "rays": (18, "beams per emitter"),
                "mirrors": (17, "circular mirrors (Sinai scatterers) spread over the frame"),
                "bounces": (9, "reflections traced per beam"),
                "exposure": (1.0, "exposure")})
def laser_billiards(ctx):
    """Geometric optics in a mirrored chamber with convex circular mirrors (a
    Sinai billiard). Beams leave many emitters at random angles and reflect
    specularly (d' = d - 2(d.n)n) off the chamber walls and the mirrors; each
    segment is drawn additively and loses 12% per bounce. The convex mirrors
    scatter beams chaotically, which builds a dense web with bright
    caustic-like crossings. data: points are the emitters; source: luminance
    attenuates beam brightness (the chamber glass is the image)."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    dp = lc.data_points(ctx)
    em = dp[0] if dp is not None else lc.jitter_grid(rng, int(ctx.p("emitters")), W, H, 0.9)
    mir = lc.blue_noise(rng, int(ctx.p("mirrors")), W, H, 0.04)
    rad = H * rng.uniform(0.035, 0.085, len(mir))
    # keep emitters out of mirrors
    for i, e in enumerate(em):
        for c, r in zip(mir, rad):
            if np.hypot(*(e - c)) < r * 1.2:
                em[i] = c + (e - c) / max(np.hypot(*(e - c)), 1e-6) * r * 1.4
    nr = int(ctx.p("rays"))
    P = np.repeat(em, nr, 0).astype(np.float64)
    th = rng.uniform(0, 2 * math.pi, len(P))
    D = np.stack([np.cos(th), np.sin(th)], 1)
    x0, x1, y0, y1 = -0.02 * W, 1.02 * W, -0.02 * H, 1.02 * H
    buf = np.zeros((H, W), np.float32)
    lum = lc.source_lum(ctx)
    gain = np.ones(len(P))
    A, B, Wt = [], [], []
    for b in range(int(ctx.p("bounces"))):
        # walls
        tx = np.where(D[:, 0] > 0, (x1 - P[:, 0]) / np.where(D[:, 0] == 0, 1e-9, D[:, 0]), (x0 - P[:, 0]) / np.where(D[:, 0] == 0, -1e-9, D[:, 0]))
        ty = np.where(D[:, 1] > 0, (y1 - P[:, 1]) / np.where(D[:, 1] == 0, 1e-9, D[:, 1]), (y0 - P[:, 1]) / np.where(D[:, 1] == 0, -1e-9, D[:, 1]))
        tw = np.minimum(tx, ty)
        nwall = np.where((tx < ty)[:, None], np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        t = tw.copy(); kind = np.full(len(P), -1)
        N = nwall.copy()
        for k, (c, r) in enumerate(zip(mir, rad)):
            pc = P - c
            bq = (D * pc).sum(1)
            cc = (pc ** 2).sum(1) - r * r
            disc = bq * bq - cc
            ok = disc > 0
            tc = np.where(ok, -bq - np.sqrt(np.maximum(disc, 0)), np.inf)
            hit = ok & (tc > 1e-4) & (tc < t)
            t = np.where(hit, tc, t); kind = np.where(hit, k, kind)
        Q = P + D * t[:, None]
        for k in range(len(mir)):
            m = kind == k
            if m.any():
                n = (Q[m] - mir[k]) / rad[k]
                N[m] = n
        A.append(P.copy()); B.append(Q.copy()); Wt.append(gain.copy())
        d_n = (D * N).sum(1)
        D = D - 2 * d_n[:, None] * N
        P = Q + D * 1e-3
        gain = gain * 0.88
    A = np.concatenate(A); B = np.concatenate(B); Wt = np.concatenate(Wt)
    if lum is not None:
        mid = ((A + B) / 2)
        ix = np.clip(mid[:, 0].astype(int), 0, W - 1); iy = np.clip(mid[:, 1].astype(int), 0, H - 1)
        Wt = Wt * (0.35 + 1.3 * lum[iy, ix])
    lc.splat_segments(buf, A, B, Wt * 0.2, step=0.8)
    # mirrors: faint rim
    rim = np.zeros((H, W), np.float32)
    for c, r in zip(mir, rad):
        a = np.linspace(0, 2 * math.pi, 240, endpoint=False)
        lc.splat(rim, c[0] + r * np.cos(a), c[1] + r * np.sin(a), 0.12 * 2 * math.pi * r / 240)
    I = buf + rim * 1.2
    return lc.develop(ctx, I, exposure=1.35 * ctx.p("exposure"), glow=0.9, glow_sigmas=(1.5, 5, 18), base=lc.ground(ctx))


@effect("caustic-pool", family="light", tags=["caustics", "refraction", "water", "all-over", "physics"],
        params={"relief": (1.0, "surface relief strength (stronger = more folds, denser filaments)"),
                "scale": (170, "ripple wavelength in px at 1500 wide"),
                "sharp": (0.07, "singularity softening (smaller = thinner, brighter filaments)"),
                "exposure": (1.0, "exposure")})
def caustic_pool(ctx):
    """Water-pool caustics from the ray-mapping Jacobian. A smooth random
    height field h(x, y) refracts parallel sunlight: a ray leaving the
    surface at x lands at x + d * grad h (small-angle Snell). Irradiance on
    the floor is the inverse area magnification, E = 1 / |det(I + d Hess h)|,
    which diverges on the fold lines (det = 0): the bright filament network
    of a pool. Evaluated in source coordinates (first-sheet approximation:
    noise-free, no ray splatting). source: luminance modulates the sunlight
    entering the surface; data unused."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    h = lc.smooth_noise(rng, (H, W), float(ctx.p("scale")) * sc * 0.5, 3, 0.45)
    gy, gx = np.gradient(h)
    gxy, gxx = np.gradient(gx)
    gyy, _ = np.gradient(gy)
    s = float(np.std(gxx)) + 1e-9
    a = float(ctx.p("relief")) * 0.85 / s
    det = (1 + a * gxx) * (1 + a * gyy) - (a * gxy) ** 2
    E = 1.0 / (np.abs(det) + float(ctx.p("sharp")))
    lum = lc.source_lum(ctx)
    if lum is not None:
        E = E * (0.3 + 1.4 * lum)
    E = cv2.GaussianBlur(E.astype(np.float32), (0, 0), 0.8 * sc)
    E = np.maximum(E - 0.9, 0)
    return lc.develop(ctx, E * 0.22, exposure=1.0 * ctx.p("exposure"), glow=0.5, glow_sigmas=(1.5, 5, 16), base=lc.ground(ctx))
