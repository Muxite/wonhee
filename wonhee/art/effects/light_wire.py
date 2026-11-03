"""Wire light: plasma arcs between nodes, neon circuit tubes, rain through a rolling shutter."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..registry import effect
from . import light_common as lc


def _jag(rng, a, b, rough, depth):
    pts = np.array([a, b], float)
    disp = np.hypot(*(b - a)) * rough
    for _ in range(depth):
        mid = (pts[:-1] + pts[1:]) / 2
        d = pts[1:] - pts[:-1]
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
        mid = mid + nrm * (rng.normal(0, 1, len(mid)) * disp * 0.5)[:, None]
        out = np.empty((len(pts) + len(mid), 2)); out[0::2] = pts; out[1::2] = mid
        pts = out; disp *= 0.55
    return pts


@effect("plasma-arcs", family="light", tags=["plasma", "arcs", "graph", "all-over", "electric", "data-driven"],
        params={"nodes": (46, "electrode count when no data is given (blue noise)"),
                "exposures": (5, "re-strikes of each arc (long exposure of a flickering arc)"),
                "rough": (0.16, "arc jaggedness relative to its length"),
                "keep": (0.8, "fraction of Delaunay edges kept (shortest first)"),
                "exposure": (1.0, "exposure")})
def plasma_arcs(ctx):
    """Electric arcs between electrodes. Nodes (blue-noise spread, or real
    data) are joined along a Delaunay triangulation (or the data's own
    edges); each edge is struck several times as a fractal midpoint-
    displaced channel, so the long exposure shows a fuzzy bright core with
    a shimmering envelope; shorter gaps carry more current (brighter).
    Electrodes glow; their size follows the node weight.
    data: {"nodes": [...], "edges": [...]} (dependency graph, import
    graph): nodes become electrodes, edges become arcs. source: luminance
    scales arc current."""
    from scipy.spatial import Delaunay
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    dp = lc.data_points(ctx)
    if dp is not None:
        P, E, wts = dp
    else:
        P = lc.blue_noise(rng, int(ctx.p("nodes")), W, H, -0.03, tries=14)
        wts = np.ones(len(P))
        E = np.zeros((0, 2), int)
    if len(E) == 0 and len(P) >= 3:
        tri = Delaunay(P)
        es = set()
        for s in tri.simplices:
            for i in range(3):
                a, b = int(s[i]), int(s[(i + 1) % 3])
                es.add((min(a, b), max(a, b)))
        E = np.array(sorted(es))
        ln = np.hypot(*(P[E[:, 0]] - P[E[:, 1]]).T)
        E = E[np.argsort(ln)[: max(1, int(len(E) * float(ctx.p("keep"))))]]
    lum = lc.source_lum(ctx)
    core = np.zeros((H, W), np.float32)
    acc = lc.SegAcc()
    meanlen = np.mean(np.hypot(*(P[E[:, 0]] - P[E[:, 1]]).T)) if len(E) else 1.0
    for a, b in E:
        L = float(np.hypot(*(P[a] - P[b])))
        cur = (meanlen / max(L, 1.0)) ** 0.8
        if lum is not None:
            m = (P[a] + P[b]) / 2
            cur *= 0.4 + 1.2 * float(lum[np.clip(int(m[1]), 0, H - 1), np.clip(int(m[0]), 0, W - 1)])
        for _ in range(int(ctx.p("exposures"))):
            pts = _jag(rng, P[a], P[b], float(ctx.p("rough")), 7)
            acc.polyline(pts, 0.16 * cur)
    acc.flush(core)
    # electrodes
    beads = np.zeros((H, W), np.float32)
    for (x, y), wv in zip(P, wts):
        lc.splat(beads, np.array([x]), np.array([y]), 6.0 * (0.8 + 0.4 * wv / wts.mean()))
    beads = cv2.GaussianBlur(beads, (0, 0), 3.0 * sc)
    core_s = cv2.GaussianBlur(core, (0, 0), 0.9 * sc)
    I = core_s * 1.6 + beads * 1.0
    return lc.develop(ctx, I, exposure=1.5 * ctx.p("exposure"), glow=1.2, glow_sigmas=(3, 9, 28), base=lc.ground(ctx))


def _route(a, b):
    """Orthogonal + 45 degree route between a and b: [a, p1, p2, b]."""
    d = b - a
    ax, ay = abs(d[0]), abs(d[1])
    sx, sy = np.sign(d[0]), np.sign(d[1])
    if ax >= ay:
        p1 = a + np.array([sx * (ax - ay) / 2, 0.0]); p2 = p1 + np.array([sx * ay, d[1]])
    else:
        p1 = a + np.array([0.0, sy * (ay - ax) / 2]); p2 = p1 + np.array([d[0], sy * ax])
    return np.array([a, p1, p2, b])


def _chaikin(pts, it=3):
    for _ in range(it):
        q = []
        for a, b in zip(pts[:-1], pts[1:]):
            q.append(0.75 * a + 0.25 * b); q.append(0.25 * a + 0.75 * b)
        pts = np.array([pts[0]] + q + [pts[-1]])
    return pts


@effect("neon-circuit", family="light", tags=["neon", "tubes", "circuit", "graph", "all-over", "data-driven"],
        params={"traces": (60, "number of tube traces (stratified over the frame)"),
                "pitch": (0.095, "routing lattice pitch as a fraction of frame height"),
                "length": (7, "mean trace length in lattice steps"),
                "lit": (0.8, "fraction of tubes that are powered (rest are dim glass)"),
                "exposure": (1.0, "exposure")})
def neon_circuit(ctx):
    """Neon tubing routed like PCB traces: random walks on a lattice that
    only turn by 45 degrees, corners rounded by Chaikin subdivision (glass
    can't bend sharply), a bright gas core inside a wider tube halo, ringed
    terminals at the ends. Starts are stratified over lattice cells so the
    frame is evenly strung; a few tubes are unpowered (dim glass).
    data: edges are routed between node positions with the same
    orthogonal+45 rule, nodes get ring terminals (a dependency graph as a
    neon wiring diagram). source: luminance scales tube brightness."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    pitch = float(ctx.p("pitch")) * H
    lum = lc.source_lum(ctx)
    dp = lc.data_points(ctx)
    paths = []
    terms = []
    if dp is not None and len(dp[1]):
        P, E, _ = dp
        for a, b in E:
            paths.append(_chaikin(_route(P[a], P[b]), 3))
        terms = list(P)
    else:
        gx, gy = int(W / pitch) + 1, int(H / pitch) + 1
        dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        n = int(ctx.p("traces"))
        starts = lc.jitter_grid(rng, n, W, H, 1.0)[:n]
        for s in starts:
            cx, cy = round(s[0] / pitch), round(s[1] / pitch)
            d = int(rng.integers(0, 8))
            pts = [(cx, cy)]
            for step in range(max(2, int(rng.poisson(float(ctx.p("length")))))):
                r = rng.random()
                if r < 0.35:
                    d = (d + rng.choice([-1, 1])) % 8
                dx, dy = dirs[d]
                cx += dx; cy += dy
                pts.append((cx, cy))
            q = np.array(pts, float) * pitch
            paths.append(_chaikin(q, 3))
            terms += [q[0], q[-1]]
    core = np.zeros((H, W), np.float32); dim = np.zeros((H, W), np.float32)
    for p in paths:
        on = rng.random() < float(ctx.p("lit"))
        amp = rng.uniform(0.7, 1.2) if on else 0.12
        if lum is not None and on:
            m = p[len(p) // 2]
            amp *= 0.4 + 1.3 * float(lum[np.clip(int(m[1]), 0, H - 1), np.clip(int(m[0]), 0, W - 1)])
        lc.splat_polyline(core if on else dim, p, amp * 0.9, step=0.7)
    ring = np.zeros((H, W), np.float32)
    rr = pitch * 0.12
    for t in terms:
        a = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        lc.splat(ring, t[0] + rr * np.cos(a), t[1] + rr * np.sin(a), 0.9 * 2 * math.pi * rr / 40)
    tube = cv2.GaussianBlur(core, (0, 0), 1.3 * sc) * 2.2 + cv2.GaussianBlur(core, (0, 0), 4.5 * sc) * 5.0
    I = tube + cv2.GaussianBlur(dim, (0, 0), 1.3 * sc) * 1.5 + cv2.GaussianBlur(ring, (0, 0), 1.0 * sc) * 2.0
    return lc.develop(ctx, I, exposure=0.8 * ctx.p("exposure"), glow=0.8, glow_sigmas=(3, 10, 30), base=lc.ground(ctx))


@effect("rain-shutter", family="light", tags=["rain", "streaks", "rolling-shutter", "all-over", "night"],
        params={"streaks": (2600, "number of light streaks over all depth layers"),
                "shear": (0.22, "rolling-shutter shear (horizontal px per row / row)"),
                "wobble": (14.0, "camera-shake wobble amplitude in px at 1500 wide"),
                "bands": (0.3, "scan-line flicker banding 0..1"),
                "exposure": (1.0, "exposure")})
def rain_shutter(ctx):
    """Night rain lit by lights, photographed with a rolling shutter. Falling
    drops at three depths leave streaks whose length ~ speed x exposure
    (near = long, thick, bright; far = short, fine). A rolling shutter reads
    rows one after another, so a moving scene is sheared: every streak
    gets x(y) = x0 + shear * y + wobble * sin(2 pi y / lambda) (shake), and
    flickering lights expose rows unequally (horizontal bands). A bead
    marks each streak's lower end. source: luminance scales streak
    brightness (rain lit by the image); data: points are lamps that boost
    nearby streaks."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    n = int(int(ctx.p("streaks")) * max(0.2, sc))
    lum = lc.source_lum(ctx)
    dp = lc.data_points(ctx)
    shear = float(ctx.p("shear"))
    wob = float(ctx.p("wobble")) * sc
    lam = rng.uniform(0.6, 1.1) * H; ph = rng.uniform(0, 6.28)
    buf_layers = [np.zeros((H, W), np.float32) for _ in range(3)]
    beads = np.zeros((H, W), np.float32)
    depth = rng.choice(3, n, p=[0.6, 0.28, 0.12])
    x0 = rng.uniform(-0.3 * W, 1.3 * W, n)
    y0 = rng.uniform(-0.15 * H, 1.1 * H, n)
    base_len = np.array([0.045, 0.11, 0.26]) * H
    ln = base_len[depth] * rng.uniform(0.6, 1.5, n)
    amp = np.array([0.5, 0.9, 1.6])[depth] * rng.uniform(0.4, 1.3, n) ** 2
    if dp is not None:
        for (px, py) in dp[0]:
            m = np.exp(-(((x0 - px) / (0.12 * W)) ** 2 + ((y0 - py) / (0.12 * H)) ** 2))
            amp = amp * (1 + 2.5 * m)
    if lum is not None:
        amp = amp * (0.2 + 1.6 * lum[np.clip(y0.astype(int), 0, H - 1), np.clip(x0.astype(int), 0, W - 1)])
    S = 14
    for d in range(3):
        idx = np.nonzero(depth == d)[0]
        if not len(idx):
            continue
        t = np.linspace(0, 1, S)[None, :]
        Y = y0[idx, None] + ln[idx, None] * t
        X = x0[idx, None] + shear * Y + 0.0 * t + wob * np.sin(2 * math.pi * Y / lam + ph) + (d - 1) * 0.0
        # slight drift per streak (wind)
        X = X + (rng.normal(0, 0.05, len(idx))[:, None] * ln[idx, None]) * t
        w = amp[idx, None] * (0.3 + 0.7 * t) / 3.0
        A = np.stack([X[:, :-1], Y[:, :-1]], -1).reshape(-1, 2); B = np.stack([X[:, 1:], Y[:, 1:]], -1).reshape(-1, 2)
        lc.splat_segments(buf_layers[d], A, B, w[:, :-1].reshape(-1), w[:, 1:].reshape(-1), step=0.8)
        lc.splat(beads, X[:, -1], Y[:, -1], amp[idx] * (0.6 + d * 0.5))
    I = np.zeros((H, W), np.float32)
    for d, sg in enumerate((0.7, 1.3, 2.4)):
        I += cv2.GaussianBlur(buf_layers[d], (0, 0), sg * sc) * (1.0, 1.6, 2.6)[d] * 1.8
    I += cv2.GaussianBlur(beads, (0, 0), 1.4 * sc) * 1.2
    # scan-line flicker banding
    b = cv2.GaussianBlur(rng.standard_normal(H).astype(np.float32)[None], (0, 0), max(1.5, 0.012 * H))[0]
    b = 1 + float(ctx.p("bands")) * b / (b.std() + 1e-6) * 0.5
    I = I * np.clip(b, 0.3, 1.9)[:, None]
    return lc.develop(ctx, I, exposure=1.3 * ctx.p("exposure"), glow=0.55, glow_sigmas=(2, 8, 24), base=lc.ground(ctx))
