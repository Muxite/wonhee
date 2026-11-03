"""AutoSpotCV: an embedding space. Every face image becomes a vector; vectors of one identity gather into a cluster.
Clusters are spread evenly over the whole frame (no hub): a soft cloud of points per identity, curved tethers pulling each
point toward its identity centre, a thin ring at the cluster edge and faint Voronoi boundaries between identities."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, get_points, lookup


@effect("proj-autospotcv-embedding", family="graph", tags=["embedding", "clusters", "all-over", "project"],
        params={"clusters": (30, "identity clusters (jittered grid, bleeds off the frame)"), "points": (60, "points per cluster"),
                "tether": (0.8, "fraction of points with a pull tether"), "spread": (0.23, "cluster radius as a fraction of the cell"),
                "voronoi": (0.5, "boundary line strength")})
def embedding(ctx):
    """data: optional (x, y[, w]) cluster centres in 0..1. Synthetic and seeded otherwise."""
    W, H = ctx.size
    rng = ctx.rng
    sc = W / 1500.0
    n = int(ctx.p("clusters"))
    xs, ys, _ = get_points(ctx, n, rng, signed=False, margin=-0.02)
    cx, cy = xs * W, ys * H
    k = len(cx)
    cell = np.sqrt(W * H / k)
    pts = np.zeros((H, W), np.uint8); tet = np.zeros_like(pts); ring = np.zeros_like(pts)
    sig_c = rng.uniform(0.75, 1.25, k) * float(ctx.p("spread")) * cell
    m = int(ctx.p("points")); ptether = float(ctx.p("tether"))
    for i in range(k):
        px = cx[i] + rng.normal(0, sig_c[i], m); py = cy[i] + rng.normal(0, sig_c[i], m)
        cv2.circle(ring, (int(cx[i]), int(cy[i])), int(2.3 * sig_c[i]), 255, max(1, int(2 * sc)), cv2.LINE_AA)
        for a, b in zip(px, py):
            if rng.random() < ptether:
                dx, dy = cx[i] - a, cy[i] - b
                L = np.hypot(dx, dy) + 1e-6
                bend = rng.uniform(-0.3, 0.3) * L
                nx, ny = -dy / L, dx / L
                mx, my = a + dx / 2 + nx * bend, b + dy / 2 + ny * bend
                t = np.linspace(0, 1, 9)[:, None]
                P = (1 - t) ** 2 * [a, b] + 2 * (1 - t) * t * [mx, my] + t ** 2 * [cx[i], cy[i]]
                cv2.polylines(tet, [np.round(P).astype(np.int32)], False, 255, max(1, int(1.2 * sc)), cv2.LINE_AA)
            r = max(2, int(rng.uniform(3.8, 6.2) * sc))
            cv2.circle(pts, (int(a), int(b)), r, 255, -1, cv2.LINE_AA)
    # Voronoi boundaries between identity centres
    hw, hh = W // 2, H // 2
    X, Y = np.meshgrid(np.arange(hw) * 2.0, np.arange(hh) * 2.0)
    d = np.stack([(X - x) ** 2 + (Y - y) ** 2 for x, y in zip(cx, cy)])
    idx = d.argmin(0)
    edge = ((idx != np.roll(idx, 1, 0)) | (idx != np.roll(idx, 1, 1))).astype(np.float32)
    edge = cv2.resize(cv2.GaussianBlur(edge, (0, 0), 0.8), (W, H))
    f = lambda a: a.astype(np.float32) / 255.0
    P, T, R = f(pts), f(tet), f(ring)
    glow = cv2.GaussianBlur(P, (0, 0), 7 * sc) * 0.8 + cv2.GaussianBlur(T, (0, 0), 4 * sc) * 0.5
    wash = cv2.GaussianBlur(P + 0.5 * T, (0, 0), 26 * sc)
    wash = np.clip(wash / (np.percentile(wash, 99) + 1e-6), 0, 1)
    t = 0.05 + 0.26 * wash + 0.22 * np.clip(glow, 0, 1) + 0.38 * T + 0.16 * R + float(ctx.p("voronoi")) * 0.28 * edge + 0.55 * P
    t = np.clip(t, 0, 1)
    return lookup(t, ctx.palette)
