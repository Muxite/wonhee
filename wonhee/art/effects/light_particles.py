"""Particle light: curl-noise flow streaks, fireworks, bokeh."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..registry import effect
from . import light_common as lc


@effect("flow-streaks", family="light", tags=["flow", "curl-noise", "particles", "all-over", "trails"],
        params={"particles": (9000, "number of particle trails"),
                "steps": (64, "integration steps per trail"),
                "scale": (3.2, "flow-field feature size (cells across the width)"),
                "step": (3.0, "step length in px at 1500 wide"),
                "exposure": (1.0, "exposure")})
def flow_streaks(ctx):
    """Particle trails in a divergence-free (curl-noise) flow, v = (dpsi/dy,
    -dpsi/dx) of a smooth random stream function psi: incompressible, so
    streaks never pile up in sinks and the whole frame is covered evenly.
    Each particle is integrated (RK2) for `steps`, drawn as a comet whose head
    is brightest and tail fades. data: points become vortices (+-) added to
    psi; source: luminance is added to psi so flow follows image contours,
    and also sets particle density/brightness."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    gw, gh = 360, int(round(360 * H / W))
    psi = lc.lowres_noise(rng, (gh, gw), float(ctx.p("scale")), 3, 0.55)
    lum = lc.source_lum(ctx)
    if lum is not None:
        l = cv2.resize(lum, (gw, gh), interpolation=cv2.INTER_AREA)
        psi = psi + 3.0 * cv2.GaussianBlur(l, (0, 0), 3)
    dp = lc.data_points(ctx)
    if dp is not None:
        yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
        for i, (px, py) in enumerate(dp[0]):
            psi += (1 if i % 2 else -1) * 2.0 * np.exp(-(((xx - px / W * gw) / 18) ** 2 + ((yy - py / H * gh) / 18) ** 2))
    psi = cv2.GaussianBlur(psi, (0, 0), 1.5)
    gpy, gpx = np.gradient(psi)
    vx = gpy; vy = -gpx
    sp = np.hypot(vx, vy); vx /= sp.mean() + 1e-6; vy /= sp.mean() + 1e-6
    n = int(int(ctx.p("particles")) * max(0.1, sc * sc)); S = int(ctx.p("steps")); h = float(ctx.p("step")) * sc
    P = np.stack([rng.uniform(0, W, n), rng.uniform(0, H, n)], 1)
    if lum is not None:
        keep = rng.random(n) < (0.25 + 0.75 * lum[np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)])
        P = P[keep]; n = len(P)
    amp = rng.uniform(0.5, 1.5, n) ** 2

    def vel(Q):
        gx_ = np.clip(Q[:, 0] / W * (gw - 1), 0, gw - 1.001); gy_ = np.clip(Q[:, 1] / H * (gh - 1), 0, gh - 1.001)
        x0 = gx_.astype(int); y0 = gy_.astype(int); fx = gx_ - x0; fy = gy_ - y0
        def s(f):
            return (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy) + f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy)
        return np.stack([s(vx), s(vy)], 1)

    buf = np.zeros((H, W), np.float32)
    for s_ in range(S):
        v1 = vel(P)
        mid = P + v1 * h * 0.5
        v2 = vel(mid)
        Q = P + v2 * h
        age = s_ / S
        w = amp * (age ** 1.8) * 0.6 / max(h, 1e-6) * 0.9
        lc.splat_segments(buf, P, Q, w, step=0.9)
        P = Q
    return lc.develop(ctx, buf, exposure=0.95 * ctx.p("exposure"), glow=0.8, glow_sigmas=(1.5, 5, 16), base=lc.ground(ctx))


@effect("firework-bursts", family="light", tags=["fireworks", "particles", "ballistic", "all-over", "starburst"],
        params={"bursts": (15, "bursts on a jittered grid (spread over the frame)"),
                "sparks": (190, "sparks per burst"),
                "drag": (0.055, "air drag k (1/s): higher = tighter, earlier droop"),
                "gravity": (0.9, "gravity relative to burst speed"),
                "exposure": (1.0, "exposure")})
def firework_bursts(ctx):
    """Ballistic fireworks with drag. A spark launched at v0 follows
    v(t) = g/k + (v0 - g/k) e^(-k t) (closed form), a spherical shell
    projected to 2D (so rings fill in toward the middle). Each burst is
    caught at its own age: young = tight rays, old = drooping willows.
    Trail brightness ~ 1/speed (slow = more exposure) and decays with age;
    a bright glitter bead marks each spark's end. data: points are the burst
    centres (size from weight); source: luminance scales burst energy."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    dp = lc.data_points(ctx)
    cen = dp[0] if dp is not None else lc.jitter_grid(rng, int(ctx.p("bursts")), W, H, 0.85)
    lum = lc.source_lum(ctx)
    k = float(ctx.p("drag"))
    buf = np.zeros((H, W), np.float32)
    beads = np.zeros((H, W), np.float32)
    acc = lc.SegAcc()
    m = max(30, int(int(ctx.p("sparks")) * max(0.25, sc)))
    for bi, c in enumerate(cen):
        R = H * rng.uniform(0.09, 0.17)
        if dp is not None:
            R *= 0.8 + 0.4 * dp[2][bi] / dp[2].mean()
        if lum is not None:
            R *= 0.7 + 0.6 * float(lum[np.clip(int(c[1]), 0, H - 1), np.clip(int(c[0]), 0, W - 1)])
        T = rng.uniform(5.0, 17.0)           # age in drag times
        v0 = R * k * 1.05 / (1 - math.exp(-k * T)) * 1.6
        ph = rng.uniform(0, 2 * math.pi, m); cz = rng.uniform(-1, 1, m)
        sp = v0 * np.sqrt(1 - cz ** 2) * (1 + 0.04 * rng.standard_normal(m))
        vx0 = sp * np.cos(ph); vy0 = sp * np.sin(ph)
        g = float(ctx.p("gravity")) * v0 * k * 0.55 * rng.uniform(0.6, 1.2)
        steps = 70
        t = np.linspace(0, T, steps)[:, None]
        e = np.exp(-k * t)
        X = c[0] + vx0[None] * (1 - e) / k
        Y = c[1] + g * t / k + (vy0[None] - g / k) * (1 - e) / k
        spd = np.hypot(vx0[None] * e, g / k + (vy0[None] - g / k) * e) + 0.2 * v0
        fade = np.exp(-2.6 * t / T)
        wgt = fade * (0.9 * v0) / spd * rng.uniform(0.7, 1.2)
        acc.segments(np.stack([X[:-1], Y[:-1]], -1).reshape(-1, 2), np.stack([X[1:], Y[1:]], -1).reshape(-1, 2),
                     (wgt[:-1] * 0.12).reshape(-1), (wgt[1:] * 0.12).reshape(-1))
        lc.splat(beads, X[-1], Y[-1], rng.uniform(0.6, 1.6, m))
        lc.splat(beads, np.array([c[0]]), np.array([c[1]]), 0.0)
    acc.flush(buf, 1.0)
    beads = cv2.GaussianBlur(beads, (0, 0), 1.1 * sc)
    I = buf + beads * 2.4
    return lc.develop(ctx, I, exposure=2.0 * ctx.p("exposure"), glow=0.5, glow_sigmas=(2, 7, 22), base=lc.ground(ctx))


@effect("bokeh-field", family="light", tags=["bokeh", "defocus", "all-over", "lens"],
        params={"discs": (520, "total discs over all depth layers"),
                "layers": (3, "depth layers (far = small, near = big)"),
                "blades": (7, "aperture blades (0 = circle)"),
                "rim": (0.55, "bright rim strength (real bokeh is brighter at its edge)"),
                "exposure": (1.0, "exposure")})
def bokeh_field(ctx):
    """Defocused point lights. Every point light blurs into the aperture
    shape (a polygon with `blades` sides, or a circle), brighter on the rim
    from spherical aberration. Depth layers set the blur radius, positions
    come from best-candidate blue noise so discs spread over the whole
    frame. source: the image defocused: disc brightness is the image
    luminance at its centre (a photo seen through a very out-of-focus lens);
    data: points are emitters with extra bright, larger discs."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    n = int(ctx.p("discs")); L = int(ctx.p("layers"))
    lum = lc.source_lum(ctx)
    dp = lc.data_points(ctx)
    buf = np.zeros((H, W), np.float32)
    blades = int(ctx.p("blades"))
    sizes = np.geomspace(0.014, 0.06, L) * H
    counts = np.array([n * (0.55 ** i) for i in range(L)]); counts = (counts / counts.sum() * n).astype(int)
    for li in range(L):
        pts = lc.blue_noise(rng, max(counts[li], 3), W, H, -0.02, tries=6)
        base_r = sizes[li] * (-1 + 0 if False else 1)
        for j, (x, y) in enumerate(pts):
            r = base_r * rng.uniform(0.8, 1.25)
            amp = rng.uniform(0.3, 1.0) ** 1.5 * (0.5 + 0.5 * li / max(L - 1, 1))
            if lum is not None:
                amp = (0.15 + 1.4 * float(lum[np.clip(int(y), 0, H - 1), np.clip(int(x), 0, W - 1)])) * (0.7 + 0.3 * rng.random())
            x0, x1 = int(max(0, x - r - 2)), int(min(W, x + r + 3)); y0, y1 = int(max(0, y - r - 2)), int(min(H, y + r + 3))
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx, dy = xx - x, yy - y
            rho = np.hypot(dx, dy)
            if blades >= 3:
                th = np.arctan2(dy, dx) + (j % 5) * 0.0
                seg = 2 * math.pi / blades
                edge = r * math.cos(math.pi / blades) / np.cos(((th % seg) - seg / 2))
            else:
                edge = r
            u = rho / edge
            disc = np.clip((1.0 - u) * edge / 1.6, 0, 1)
            rimm = np.exp(-((u - 0.93) / 0.07) ** 2) * float(ctx.p("rim"))
            buf[y0:y1, x0:x1] += (amp * (disc * (0.55 + rimm))) * 0.9
    if dp is not None:
        for (x, y), wv in zip(dp[0], dp[2]):
            r = H * 0.05 * (0.8 + 0.4 * wv / dp[2].mean())
            yy, xx = np.mgrid[0:H, 0:W]
            rho = np.hypot(xx - x, yy - y) / r
            buf += (np.clip((1 - rho) * r / 1.6, 0, 1) * (0.9 + np.exp(-((rho - 0.93) / 0.07) ** 2))).astype(np.float32)
    buf = cv2.GaussianBlur(buf, (0, 0), 0.7 * sc)
    return lc.develop(ctx, buf, exposure=1.1 * ctx.p("exposure"), glow=0.25, glow_sigmas=(3, 10, 30), base=lc.ground(ctx), grain=0.014)
