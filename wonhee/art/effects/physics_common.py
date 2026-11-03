"""Shared helpers for the physics_* effects (not an effect itself)."""
from __future__ import annotations

import cv2
import numpy as np

_RAMPS: dict = {}


def fit(a: np.ndarray, W: int, H: int, cubic: bool = True) -> np.ndarray:
    return cv2.resize(np.ascontiguousarray(a, dtype=np.float32), (int(W), int(H)),
                      interpolation=cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR)


def sim_size(ctx, max_w: int):
    """(sw, sh) working resolution with the frame aspect, capped at max_w wide."""
    W, H = ctx.size
    sw = int(min(W, max_w))
    return sw, max(8, int(round(sw * H / W)))


def pnorm(f, lo=2.0, hi=98.0):
    a, b = np.percentile(f, [lo, hi])
    return np.clip((f - a) / (b - a + 1e-9), 0, 1)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def lookup(t, pal, stops=("canvas", "structure", "accent"), n=512):
    """Map t in 0..1 through the palette ramp (linear interpolation)."""
    key = (pal.name, tuple(stops), n, tuple(np.round(pal.canvas, 4)), tuple(np.round(pal.structure, 4)), tuple(np.round(pal.accent, 4)))
    ramp = _RAMPS.get(key)
    if ramp is None:
        ramp = pal.ramp(n, stops)
        _RAMPS[key] = ramp
    x = np.clip(t, 0, 1) * (n - 1)
    i = np.minimum(x.astype(np.int32), n - 2)
    f = (x - i)[..., None].astype(np.float32)
    return ramp[i] * (1 - f) + ramp[i + 1] * f


def get_points(ctx, n, rng, signed=False, margin=0.08):
    """Source points as fractions of the frame: (x, y, w) arrays. Uses ctx.data
    (list of (x, y[, w]) in 0..1) if given, else a jittered grid of n points."""
    d = ctx.data
    if d is not None:
        arr = np.asarray(d, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr):
            w = arr[:, 2] if arr.shape[1] > 2 else np.ones(len(arr))
            return arr[:, 0], arr[:, 1], w
    W, H = ctx.size
    gx = int(np.ceil(np.sqrt(n * W / H)))
    gy = int(np.ceil(n / gx))
    cells = [(i, j) for i in range(gx) for j in range(gy)]
    order = rng.permutation(len(cells))[:n]
    xs, ys = [], []
    for k in order:
        i, j = cells[k]
        xs.append((i + 0.2 + 0.6 * rng.random()) / gx)
        ys.append((j + 0.2 + 0.6 * rng.random()) / gy)
    w = rng.choice([-1.0, 1.0], len(xs)) * (0.7 + 0.6 * rng.random(len(xs))) if signed else 0.7 + 0.6 * rng.random(len(xs))
    return np.array(xs), np.array(ys), w


def source_lum(ctx, w, h):
    if ctx.source is None:
        return None
    s = np.asarray(ctx.source, np.float32)
    lum = s @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return fit(lum, w, h, cubic=False)


def smooth_noise(rng, h, w, sigma, wrap=True):
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    from scipy import ndimage
    n = ndimage.gaussian_filter(n, sigma, mode="wrap" if wrap else "reflect")
    return n / (n.std() + 1e-9)
