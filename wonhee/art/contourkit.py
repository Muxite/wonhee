"""Helpers for the ``contour_*`` effects (no effect is registered here).

The contour family separates four controls, so a look can be changed one at a
time: FIELD (what is contoured: a photo's tone, a data array, or procedural
terrain; with `scale` for the coarse smoothing, `denoise` for speckle,
`mask` for where lines may appear), LINES (levels, spacing, orientation, weight
hierarchy, wobble), COLOUR (palette ramp, dark/light, tint) and SURFACE
(paper tooth, grain, haze). `min_px` guarantees every line at least that many
output pixels wide so the drawing survives a 288x192 thumbnail.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import features as F
from . import natkit as N

FIELD_PARAMS = {
    "scale": (1.0, "FIELD: coarse smoothing multiplier (bigger = broader landforms, fewer small islands)"),
    "denoise": (0.6, "FIELD: 0..1 noise suppression (median + despeckle of the field and tiny components)"),
    "use_source": (1.0, "FIELD: 0..1 blend of the source photo's tone with procedural terrain (0 = procedural only)"),
    "local": (0.5, "FIELD: 0..1 local-contrast equalisation of the source tone (spreads lines over the whole frame)"),
    "mask": ("none", "FIELD: where lines may appear: none | tone (mid-tones) | detail (busy areas) | saliency (what differs from the average)"),
    "mask_amount": (0.0, "FIELD: 0..1 how strongly the mask removes lines elsewhere"),
}
LINE_PARAMS = {
    "min_px": (1.2, "LINES: minimum line width in OUTPUT pixels (thumbnail guarantee; lines never thinner than this)"),
    "wobble": (0.0, "LINES: 0..1 hand-drawn wobble of the lines (smooth space-time noise, coordinate based)"),
}


def with_contour(params: dict, source=True) -> dict:
    out = N.with_common(params, source=False)
    for grp in (FIELD_PARAMS, LINE_PARAMS):
        for k, v in grp.items():
            out.setdefault(k, v)
    out.setdefault("crop", N.SOURCE_PARAMS["crop"])
    return out


def sc(size):
    return size[0] / 1500.0


# ---------------------------------------------------------------- field
def field(ctx, size, work=None):
    """Scalar field in 0..1 at ``size`` plus (img, theta, coh) of the source (or None).
    Source tone (local-equalised), ctx.data 2-D array, or procedural terrain."""
    W, H = size
    rng = np.random.default_rng(ctx.seed + 31)
    s = sc(size)
    img = theta = coh = None
    data = ctx.data
    f = None
    if isinstance(data, np.ndarray) and data.ndim == 2:
        f = cv2.resize(data.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        f = F.stretch(f, 1, 99)
    elif isinstance(data, (list, tuple)) and len(data) and np.ndim(data) == 2:
        f = F.stretch(cv2.resize(np.asarray(data, np.float32), (W, H), interpolation=cv2.INTER_CUBIC), 1, 99)
    src = ctx.source
    k = float(ctx.p("use_source", 1.0))
    if src is not None and f is None and k > 0:
        img = F.fit_cover(np.asarray(src, np.float32)[..., :3], (W, H), ctx.p("crop", "feature"))
        L = F.luminance(img)
        mix = float(ctx.p("local", 0.5))
        L = F.stretch(F.local_equalize(L, 2.2, mix=mix) if mix > 0 else L, 1, 99)
        f = L
    proc = None
    if f is None or k < 1:
        proc = terrain(rng, H, W)
        f = proc if f is None else f * k + proc * (1 - k)
    f = f.astype(np.float32)
    sg = float(ctx.p("scale", 1.0))
    dn = float(ctx.p("denoise", 0.6))
    if dn > 0:
        m = cv2.medianBlur((np.clip(f, 0, 1) * 255).astype(np.uint8), 5 if dn > 0.4 else 3).astype(np.float32) / 255
        f = f * (1 - dn) + m * dn
    f = F.blur(f, (6.0 if (src is not None or data is not None) else 2.5) * s * sg)
    f = F.stretch(f, 1, 99)
    if img is not None:
        theta, coh = F.structure_orientation(F.blur(f, 0.8 * s), sigma=1.5, rho=10.0, seed=ctx.seed)
    return f, img, theta, coh


def terrain(rng, H, W):
    """Procedural height field: broad massifs + ridges + fine relief; no focal point."""
    s = W / 1500.0
    f = N.fbm(rng, H, W, W * 0.28, 3, 0.5) + 0.55 * np.abs(N.fbm(rng, H, W, W * 0.12, 3, 0.55, aniso=1.6, angle=0.5)) \
        + 0.18 * N.fbm(rng, H, W, W * 0.03, 3, 0.6)
    return F.stretch(f, 1, 99)


def mask_field(ctx, f, img):
    """0..1 weight of where lines may appear (1 = everywhere)."""
    mode = str(ctx.p("mask", "none"))
    amt = float(ctx.p("mask_amount", 0.0))
    if mode == "none" or amt <= 0:
        return np.ones_like(f)
    if mode == "tone":
        m = 1 - np.abs(f - 0.5) * 2
    elif mode == "saliency" and img is not None:
        m = F.saliency(img)
        m = F.stretch(F.blur(m, 8 * sc(f.shape[::-1])), 5, 95)
    else:
        m = F.detail_energy(f, 18)
        m = F.stretch(m, 5, 95)
    return (1 - amt + amt * np.clip(m, 0, 1)).astype(np.float32)


# ---------------------------------------------------------------- lines
def line_alpha(dist_px, width_px):
    """Antialiased line coverage from the distance to the line centre (px)."""
    return np.clip(width_px * 0.5 - dist_px + 0.5, 0, 1)


def level_lines(f, n, width_px, grad_floor=0.0):
    """Distance (px) to the nearest of ``n`` evenly spaced iso-levels of f in 0..1
    and the local slope. Distance = |frac - 0.5 offset| / |grad|."""
    g = f * n
    fr = g - np.floor(g)
    d_lvl = np.minimum(fr, 1 - fr)
    gy, gx = np.gradient(g)
    gm = np.hypot(gx, gy) + 1e-6
    return d_lvl / gm, gm


def wobble_field(ctx, size, amount, period=0.07):
    """Smooth displacement field (dx, dy) in px: coordinate-based space-time noise."""
    W, H = size
    if amount <= 0:
        return None
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = float(ctx.p("time", 0.0))
    a = amount * 0.012 * W
    dx = a * N.tnoise(xx / W, yy / W, t, ctx.seed + 1, period)
    dy = a * N.tnoise(xx / W + 7.1, yy / W, t, ctx.seed + 2, period)
    return dx + a * 0.35 * N.tnoise(xx / W, yy / W, t, ctx.seed + 3, period * 0.3), dy + a * 0.35 * N.tnoise(xx / W, yy / W, t, ctx.seed + 4, period * 0.3)


def warp_by(a, disp):
    if disp is None:
        return a
    H, W = a.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(np.ascontiguousarray(a, np.float32), gx + disp[0], gy + disp[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def drop_small(mask, min_area):
    """Remove connected components smaller than min_area px from a binary mask."""
    if min_area <= 1:
        return mask
    m = (mask > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_area)
    return keep[lab].astype(np.float32)


# ---------------------------------------------------------------- colour
def paper(ctx, size, f=None):
    """Background colour (H, W, 3): canvas with a faint tooth (surface layer)."""
    W, H = size
    rng = np.random.default_rng(ctx.seed + 5)
    R = ink_ramp(ctx)
    base = N.paint(np.full((H, W), 0.0, np.float32), R)
    return base


def ink_ramp(ctx):
    return N.ramp(ctx.palette, 640, ctx.p("ramp", "mono"), float(ctx.p("tint", 0.0)), float(ctx.p("contrast", 1.0)), bool(ctx.p("invert", False)))
