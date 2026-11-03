"""Refraction caustics: light through a rippled surface (a pool floor, a
glass relief). Rays are traced through the slope of a random height field and
splatted where they land; the density of landings is the caustic network."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, lookup, smooth_noise, source_lum


@effect("physics-caustics", family="physics", tags=["optics", "physics", "all-over", "caustics", "refraction"],
        params={"scale": (0.07, "feature size of the surface (fraction of frame height)"), "depth": (0.9, "focusing depth (higher = sharper, thinner network)"),
                "octaves": (2, "number of surface noise octaves"), "gamma": (0.55, "tone curve")})
def caustics(ctx):
    """source: luminance is the height field of the refracting surface (a glass relief of the image)."""
    W, H = ctx.size
    rng = ctx.rng
    sw = min(W, 1100)
    sh = max(8, round(sw * H / W))
    h = np.zeros((sh, sw), np.float32)
    lum = source_lum(ctx, sw, sh)
    if lum is not None:
        h = cv2.GaussianBlur(lum, (0, 0), sh * 0.015) * sh * 0.12
    else:
        for o in range(int(ctx.p("octaves"))):
            h += smooth_noise(rng, sh, sw, sh * float(ctx.p("scale")) / (1.8 ** o)) / (1.7 ** o)
    h = cv2.GaussianBlur(h, (0, 0), 1.2)
    gy, gx = np.gradient(h)
    gx /= np.abs(gx).std() + 1e-9; gy /= np.abs(gy).std() + 1e-9
    k = float(ctx.p("depth")) * sh * float(ctx.p("scale")) * 0.9
    SS = 2
    ys, xs = np.mgrid[0:sh * SS, 0:sw * SS].astype(np.float32) / SS
    gxs = cv2.resize(gx, (sw * SS, sh * SS), interpolation=cv2.INTER_LINEAR)
    gys = cv2.resize(gy, (sw * SS, sh * SS), interpolation=cv2.INTER_LINEAR)
    lx = (xs + k * gxs) % sw
    ly = (ys + k * gys) % sh
    ix, iy = np.floor(lx).astype(np.int64), np.floor(ly).astype(np.int64)
    fx, fy = lx - ix, ly - iy
    D = np.zeros(sw * sh, np.float32)
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        D += np.bincount(((iy + dy) % sh).ravel() * sw + ((ix + dx) % sw).ravel(), weights=w.ravel(), minlength=sw * sh).astype(np.float32)
    D = D.reshape(sh, sw) / (SS * SS)
    D = cv2.GaussianBlur(D, (0, 0), 0.9)
    t = np.clip(D / (np.percentile(D, 99.3) + 1e-9), 0, 1) ** float(ctx.p("gamma"))
    t = 0.05 + 0.9 * t
    return fit(lookup(t, ctx.palette), W, H)
