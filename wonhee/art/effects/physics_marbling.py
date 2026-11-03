"""Paper marbling as a composition of exact area-preserving flows: ink drops
(c + (p-c) sqrt(1 + r^2/|p-c|^2)) and comb tine lines (p + a g^d u). Each
pixel is traced backwards through the inverse maps to the ink it came from."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, get_points, lookup, source_lum


@effect("physics-marbling", family="physics", tags=["fluid", "physics", "all-over", "marbling", "flow"],
        params={"drops": (48, "number of ink drops"), "combs": (2, "comb passes"), "tines": (16, "tine lines per comb pass"),
                "rings": (5.0, "ink rings per drop (colour bands)"), "scale": (1.0, "working-resolution factor (<1 faster, softer)")})
def marbling(ctx):
    """data: (x, y, w) in 0..1 = drop positions (w scales the drop radius).
    source: image is the ink itself, so the output is the marbled image."""
    W, H = ctx.size
    rng = ctx.rng
    sc = float(ctx.p("scale"))
    w, h = max(64, int(min(W, 1000) * sc)), max(48, int(min(W, 1000) * sc * H / W))
    A = W / H
    xs, ys = np.meshgrid((np.arange(w) + 0.5) / h, (np.arange(h) + 0.5) / h)  # units of frame height
    ops = []
    nd = int(ctx.p("drops"))
    if ctx.data is not None:
        dx, dy, dw = get_points(ctx, 0, rng)
        cx = list(dx * A); cy = list(dy); cr = [0.05 + 0.06 * abs(v) for v in dw]
    else:
        cx = list(rng.random(nd) * A); cy = list(rng.random(nd)); cr = list(0.04 + 0.11 * rng.random(nd) ** 1.5)
    for x, y, r in zip(cx, cy, cr):
        ops.append(("drop", x, y, r))
    for _ in range(int(ctx.p("combs"))):
        ang = rng.choice([0.0, np.pi / 2]) + rng.normal(0, 0.12)
        u = np.array([np.cos(ang), np.sin(ang)])
        nrm = np.array([-u[1], u[0]])
        off = rng.random() * 0.08
        n = int(ctx.p("tines"))
        spacing = (max(A, 1.0) * 1.3) / n
        for i in range(n):
            base = (np.array([A / 2, 0.5]) + nrm * ((i - n / 2) * spacing + off))
            ops.append(("tine", base[0], base[1], nrm[0], nrm[1], u[0], u[1], 0.13 * (1 if rng.random() < 0.5 else -1), 0.05 + 0.05 * rng.random()))
    px, py = xs.copy(), ys.copy()
    ink = np.full((h, w), -1, np.int32)
    alive = np.ones((h, w), bool)
    ringv = np.zeros((h, w), np.float32)
    for k in range(len(ops) - 1, -1, -1):
        op = ops[k]
        if op[0] == "drop":
            _, x, y, r = op
            ddx, ddy = px - x, py - y
            d2 = ddx * ddx + ddy * ddy
            inside = alive & (d2 < r * r)
            ink[inside] = k
            ringv[inside] = np.sqrt(d2[inside]) / r
            alive &= ~inside
            f = np.sqrt(np.maximum(1 - (r * r) / np.maximum(d2, 1e-12), 0))
            px = np.where(alive, x + ddx * f, px)
            py = np.where(alive, y + ddy * f, py)
        else:
            _, bx, by, nx, ny, ux, uy, amp, lam = op
            d = np.abs((px - bx) * nx + (py - by) * ny)
            shift = amp * np.exp(-(d / lam) ** 2)
            px = px - shift * ux
            py = py - shift * uy
    # base ink: concentric rings about the origin of each pixel's pre-image
    pal = ctx.palette
    rings = float(ctx.p("rings"))
    rr = np.hypot(px - A / 2, py - 0.5)
    base = 0.5 + 0.5 * np.cos(rr * rings * 2 * np.pi / 0.35)
    t = 0.10 + 0.38 * base
    vals = (0.18 + 0.72 * rng.random(len(ops))).astype(np.float32)
    idx = np.where(ink >= 0, ink, 0)
    dropt = vals[idx]
    # ink inside a drop keeps a ring structure by its own local radius
    ct = np.where(ink >= 0, dropt, t)
    # banding within the drop: distance from pre-image of its centre is lost, use ink value + stripes of base
    dring = 0.5 + 0.5 * np.cos(ringv * np.pi * 2 * (2 + (idx % 3)))
    ct = np.where(ink >= 0, 0.18 + 0.7 * dropt * (0.45 + 0.55 * dring), ct)
    out = lookup(ct, pal)
    if ctx.source is not None:
        sx = np.clip(px / A * (ctx.source.shape[1] - 1), 0, ctx.source.shape[1] - 1).astype(np.float32)
        sy = np.clip(py * (ctx.source.shape[0] - 1), 0, ctx.source.shape[0] - 1).astype(np.float32)
        out = cv2.remap(np.asarray(ctx.source, np.float32), sx, sy, cv2.INTER_LINEAR)
    return fit(out, W, H)
