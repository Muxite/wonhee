"""Strange attractors rendered as log-density fields: Clifford and de Jong maps
iterated for millions of points, one attractor per tile of a poster grid.
Tiles whose parameters give a thin or degenerate orbit are re-rolled."""
from __future__ import annotations

import numpy as np

from ..registry import effect
from .physics_common import lookup


def _iter(kind, p, n, steps, burn, rng):
    x = rng.uniform(-0.1, 0.1, n); y = rng.uniform(-0.1, 0.1, n)
    a, b, c, d = p
    out = np.empty((steps, 2, n), np.float32)
    for i in range(burn + steps):
        if kind == "clifford":
            x, y = np.sin(a * y) + c * np.cos(a * x), np.sin(b * x) + d * np.cos(b * y)
        else:
            x, y = np.sin(a * y) - np.cos(b * x), np.sin(c * x) - np.cos(d * y)
        if i >= burn:
            out[i - burn, 0], out[i - burn, 1] = x, y
    return out[:, 0].ravel(), out[:, 1].ravel()


def _good(kind, p, rng):
    x, y = _iter(kind, p, 400, 60, 30, rng)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return False
    H, xe, ye = np.histogram2d(x, y, bins=24)
    return (H > 0).mean() > 0.45 and (x.std() > 0.6 and y.std() > 0.6)


@effect("physics-attractor", family="physics", tags=["chaos", "physics", "all-over", "attractor", "tiles"],
        params={"tiles": (6, "number of attractor tiles"), "kind": ("mixed", "clifford | dejong | mixed"), "points": (900000, "iterates per tile"),
                "gap": (0.0, "gap between tiles (fraction of tile)"), "gain": (1.0, "density gain")})
def attractor(ctx):
    """data: optional 1-D series; each value (scaled) seeds one tile's parameters so a project's series maps to a poster of attractors."""
    W, H = ctx.size
    rng = ctx.rng
    n = int(ctx.p("tiles"))
    series = None
    if ctx.data is not None and np.asarray(ctx.data).ndim == 1 and len(ctx.data) > 0:
        series = np.asarray(ctx.data, float)
        n = min(len(series), 24)
    cols = max(1, int(round(np.sqrt(n * W / H))))
    rows = int(np.ceil(n / cols))
    tw, th = W / cols, H / rows
    kind0 = str(ctx.p("kind"))
    img = np.zeros((H, W), np.float32)
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        sub = np.random.default_rng(ctx.seed * 7919 + i * 101 + 1 + (0 if series is None else int(abs(series[i % len(series)]) * 1e4) % 100003))
        kind = kind0 if kind0 in ("clifford", "dejong") else ("clifford" if i % 2 == 0 else "dejong")
        for _ in range(150):
            p = sub.uniform(-3.0, 3.0, 4) if kind == "dejong" else np.array([sub.uniform(-2.2, 2.2), sub.uniform(-2.2, 2.2), sub.uniform(-1.2, 1.2), sub.uniform(-1.2, 1.2)])
            if _good(kind, p, sub):
                break
        pts = int(ctx.p("points"))
        x, y = _iter(kind, p, 2000, pts // 2000, 40, sub)
        x0, x1 = np.percentile(x, [0.3, 99.7]); y0, y1 = np.percentile(y, [0.3, 99.7])
        s = min(tw / (x1 - x0 + 1e-6), th / (y1 - y0 + 1e-6)) * (1 - float(ctx.p("gap")))
        # fill the tile in the long direction, centre the short one
        ix = ((x - (x0 + x1) / 2) * s + tw / 2).astype(np.int64); iy = ((y - (y0 + y1) / 2) * s + th / 2).astype(np.int64)
        ok = (ix >= 0) & (ix < int(tw)) & (iy >= 0) & (iy < int(th))
        tile = np.bincount(iy[ok] * int(tw) + ix[ok], minlength=int(tw) * int(th)).reshape(int(th), int(tw)).astype(np.float32)
        v = np.log1p(tile)
        v = v / (np.percentile(v[v > 0], 97.0) + 1e-6) if (v > 0).any() else v
        y_off, x_off = int(r * th), int(c * tw)
        img[y_off:y_off + int(th), x_off:x_off + int(tw)] = np.clip(v * float(ctx.p("gain")), 0, 1.15)
    import cv2
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    t = 0.03 + 0.97 * np.clip(img, 0, 1) ** 1.15
    return lookup(t, ctx.palette).astype(np.float32)
