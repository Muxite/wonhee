"""Lightning: dielectric breakdown (Lichtenberg figures) and fractal bolts."""
from __future__ import annotations

import math

import numpy as np

from ..registry import effect
from . import light_common as lc


def _descendants(parent, order):
    n = len(parent)
    d = np.ones(n)
    for i in order[::-1]:
        if parent[i] >= 0:
            d[parent[i]] += d[i]
    return d


@effect("lightning-dbm", family="light", tags=["lightning", "physics", "all-over", "simulation", "lichtenberg"],
        params={"seeds": (15, "discharge roots on a jittered grid (they tile the frame)"),
                "eta": (2.2, "DBM exponent: growth ~ potential^eta (1 = bushy, 3+ = sparse stringy)"),
                "cells": (170, "simulation grid width in cells"),
                "fill": (0.15, "fraction of grid cells the discharges may occupy"),
                "exposure": (1.0, "exposure")})
def lightning_dbm(ctx):
    """Dielectric breakdown model. Poisson's equation (uniform background charge, periodic domain so there is no
    frame edge to hug) is relaxed on a grid with every grown cell held at 0; a new
    cell joins a discharge with probability ~ phi^eta among the free cells
    touching it. Many roots compete (their fields screen each other), so the
    figures tile the frame like Lichtenberg dust figures. Branch brightness
    ~ sqrt(sub-tree size) (current carried). data: points become roots;
    source: luminance raises the local breakdown threshold (dark areas grow
    more), i.e. the image shows through as where the lightning is dense."""
    from scipy import ndimage
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    gw = int(ctx.p("cells")); gh = int(round(gw * H / W))
    eta = float(ctx.p("eta"))
    dp = lc.data_points(ctx)
    roots = (dp[0] / [W, H] * [gw, gh]) if dp is not None else lc.jitter_grid(rng, int(ctx.p("seeds")), gw, gh, 0.7)
    lum = lc.source_lum(ctx)
    gate = None
    if lum is not None:
        import cv2
        gate = 0.4 + 1.2 * (1 - cv2.resize(lum, (gw, gh), interpolation=cv2.INTER_AREA))
    occ = np.zeros((gh, gw), bool)
    parent = []; pos = []
    for x, y in roots:
        xi, yi = int(np.clip(x, 0, gw - 1)) % gw, int(np.clip(y, 0, gh - 1)) % gh
        if not occ[yi, xi]:
            occ[yi, xi] = True; parent.append(-1); pos.append((xi, yi))
    idmap = -np.ones((gh, gw), int)
    for i, (x, y) in enumerate(pos):
        idmap[y, x] = i
    phi = np.ones((gh, gw), np.float32)
    phi[occ] = 0
    border = np.zeros((gh, gw), bool); border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    keep_out = np.zeros((gh, gw), bool); m_ = 5; keep_out[:m_, :] = keep_out[-m_:, :] = keep_out[:, :m_] = keep_out[:, -m_:] = True
    target = int(ctx.p("fill") * gw * gh)
    k8 = np.ones((3, 3), bool); k8[1, 1] = False
    batch = max(6, target // 90)
    order_t = []
    while len(pos) < target:
        for _ in range(45):
            p = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) + np.roll(phi, 1, 1) + np.roll(phi, -1, 1)) * 0.25
            p += 0.25 * 0.02; p[occ] = 0.0
            phi = p
        cand = ndimage.binary_dilation(occ, border_value=0, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)) & ~occ
        cy, cx = np.nonzero(cand)
        if len(cy) == 0:
            break
        pr = np.maximum(phi[cy, cx], 1e-6) ** eta
        if gate is not None:
            pr = pr * gate[cy, cx]
        pr /= pr.sum()
        m = min(batch, len(cy))
        pick = rng.choice(len(cy), size=m, replace=False, p=pr)
        for j in pick:
            x, y = int(cx[j]), int(cy[j])
            if occ[y, x]:
                continue
            best, bi = 9.0, -1   # parent: neighbour in cluster (4-neigh), nearest-by-order random
            nb = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            ps = [idmap[b, a] for a, b in nb if 0 <= a < gw and 0 <= b < gh and idmap[b, a] >= 0]
            if not ps:
                continue
            idmap[y, x] = len(pos); parent.append(int(ps[rng.integers(len(ps))])); pos.append((x, y)); occ[y, x] = True
    parent = np.array(parent); pos = np.array(pos, np.float64)
    order = np.arange(len(parent))
    d = _descendants(parent, order)
    has = parent >= 0
    cell = np.array([W / gw, H / gh])
    P = (pos + 0.5) * cell
    # jag: small deterministic jitter breaks the grid look
    P = P + rng.normal(0, 0.22, P.shape) * cell
    a, b = P[has], P[parent[has]]
    near = np.hypot(*(a - b).T) < 3 * max(cell)
    has_idx = np.nonzero(has)[0][near]
    a, b = a[near], b[near]
    wt = np.sqrt(d[has_idx])
    buf = np.zeros((H, W), np.float32)
    # thin core pass plus a broader pass for trunks
    lc.splat_segments(buf, a, b, 0.5 * wt / wt.mean(), step=0.7)
    buf2 = np.zeros((H, W), np.float32)
    lc.splat_segments(buf2, a, b, (wt / wt.max()) ** 1.5 * 3.0, step=0.7)
    import cv2
    buf2 = cv2.GaussianBlur(buf2, (0, 0), 1.6 * sc)
    I = buf * 0.9 + buf2 * 1.4
    return lc.develop(ctx, I, exposure=1.7 * ctx.p("exposure"), glow=1.0, glow_sigmas=(3, 9, 26), base=lc.ground(ctx))


def _bolt(rng, p0, p1, rough, depth):
    pts = np.array([p0, p1], float)
    disp = np.hypot(*(p1 - p0)) * rough
    for _ in range(depth):
        mid = (pts[:-1] + pts[1:]) / 2
        d = pts[1:] - pts[:-1]
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
        mid = mid + nrm * (rng.normal(0, 1, len(mid)) * disp * 0.35)[:, None]
        out = np.empty((len(pts) + len(mid), 2)); out[0::2] = pts; out[1::2] = mid
        pts = out; disp *= 0.55
    return pts


@effect("lightning-bolts", family="light", tags=["lightning", "all-over", "bolts", "fast"],
        params={"bolts": (8, "main bolts across the frame width"),
                "forks": (5, "fork attempts per bolt (recursive)"),
                "rough": (0.32, "midpoint displacement roughness"),
                "exposure": (1.0, "exposure"),
                "angle": (0.0, "mean bolt direction, radians from vertical")})
def lightning_bolts(ctx):
    """Fractal bolts: midpoint-displacement main channels from top to bottom at
    stratified x positions (so none is central), each throwing recursive forks
    that thin out and dim. Hot white core, wide halo (several exposures
    overlaid, as a long exposure captures re-strikes). data: points are
    strike targets (bolts run from the frame top to each point's x,y);
    source: luminance gates fork brightness."""
    import cv2
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    n = int(ctx.p("bolts"))
    core = np.zeros((H, W), np.float32)
    lum = lc.source_lum(ctx)
    dp = lc.data_points(ctx)
    segs = []   # (pts, weight)

    def fork(pts, w, depth):
        if depth > 2 or len(pts) < 12:
            return
        k = int(ctx.p("forks")) // (depth + 1) + 1
        for _ in range(k):
            i = int(rng.integers(len(pts) // 8, len(pts) - 4))
            d = pts[min(i + 6, len(pts) - 1)] - pts[i - 1]
            ang = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.3, 0.9)
            ln = rng.uniform(0.08, 0.28) * H / (1 + depth)
            q = pts[i] + ln * np.array([math.cos(ang), math.sin(ang)])
            sub = _bolt(rng, pts[i], q, ctx.p("rough"), 6)
            segs.append((sub, w * 0.5))
            fork(sub, w * 0.5, depth + 1)

    xs = (np.arange(n) + rng.uniform(0.15, 0.85, n)) / n * W
    for i in range(n):
        if dp is not None and i < len(dp[0]):
            end = dp[0][i]
            start = np.array([end[0] + rng.normal(0, W * 0.04), -0.05 * H])
        else:
            x0 = xs[i]
            tilt = math.tan(ctx.p("angle") + rng.normal(0, 0.12))
            start = np.array([x0 + tilt * 0.5 * H, -0.05 * H]); end = np.array([x0 - tilt * 0.5 * H + rng.normal(0, W * 0.03), 1.05 * H])
        # stack of re-strikes along the same channel
        for r in range(2):
            e = end + rng.normal(0, 2.5 * sc, 2)
            main = _bolt(rng, start, e, ctx.p("rough") * (0.8 + 0.4 * r / 2), 8)
            w = rng.uniform(0.7, 1.2)
            segs.append((main, w))
            if r == 0:
                fork(main, w, 0)
    acc = lc.SegAcc()
    for pts, w in segs:
        wgt = w * (0.55 if w < 0.6 else 1.0)
        if lum is not None:
            wgt *= 0.5 + float(lum[np.clip(int(pts[0, 1]), 0, H - 1), np.clip(int(pts[0, 0]), 0, W - 1)])
        acc.polyline(pts, wgt * 0.5)
    acc.flush(core)
    core = cv2.GaussianBlur(core, (0, 0), 0.8 * sc)
    return lc.develop(ctx, core * 1.7, exposure=1.4 * ctx.p("exposure"), glow=1.4, glow_sigmas=(3, 10, 30), base=lc.ground(ctx))
