"""dotkit: a general dot-matrix system for wonhee.art.

Three layers:

1. FIELDS: helpers that make a scalar field (H x W float) from noise, wave
   interference, an image's luminance or a data matrix (``norm01``,
   ``smooth_noise``, ``wave_field``, ``luminance``).
2. LATTICES: where the dots go (``square``, ``hexagonal``, ``polar``,
   ``poisson``, ``stipple``, ``quadtree``, ``grid``). A lattice is a ``Dots``
   (x, y, cell size per dot, optional grid row/col).
3. RENDERING: ``render_dots(field, ...)`` draws any field on any lattice with
   round / square / diamond / line / plus / ring dots (analytic anti-aliasing),
   size, alpha and colour driven by one or two fields, plus an LED look (off
   dim dots, glow). ``halftone`` makes screen halftones (round, line, hex,
   Bayer ordered dither, error diffusion).

A text / bitmap rasteriser (``text_matrix``) turns strings into boolean dot
matrices (built-in 5x7 font, or any TTF through PIL), so a word is a mosaic.

Everything is numpy / scipy / PIL math driven by a seeded ``rng``. Coordinates
are pixels; pixel ``i`` has its centre at ``i + 0.5``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

SHAPES = ("round", "square", "diamond", "line", "plus", "ring")


# --------------------------------------------------------------------------- fields

def norm01(f, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Robustly rescale any array to 0..1 (percentile clip)."""
    f = np.asarray(f, np.float32)
    a, b = np.percentile(f, [lo, hi])
    if b - a < 1e-9:
        return np.full(f.shape, 0.5, np.float32)
    return np.clip((f - a) / (b - a), 0, 1).astype(np.float32)


def luminance(img) -> np.ndarray:
    """RGB (H,W,3) 0..1 -> luminance (H,W)."""
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        return img
    return img[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def smooth_noise(shape, scale: float, rng, octaves: int = 3, persistence: float = 0.5) -> np.ndarray:
    """Multi-octave value noise in 0..1; ``scale`` = feature size in pixels of
    the coarsest octave. Detail is statistically uniform over the frame."""
    H, W = shape
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(1.0, scale / (2 ** o))
        gh, gw = int(H / s) + 3, int(W / s) + 3
        g = rng.random((gh, gw)).astype(np.float32)
        z = ndimage.zoom(g, (s, s), order=3)
        z = z[:H, :W]
        if z.shape != (H, W):
            z = np.pad(z, ((0, H - z.shape[0]), (0, W - z.shape[1])), mode="edge")
        out += amp * z
        tot += amp
        amp *= persistence
    return norm01(out / tot)


def wave_field(shape, rng, n_sources: int = 7, wavelength: float = 60.0, decay: float = 0.0, jitter: float = 0.6) -> np.ndarray:
    """Analytic interference of circular waves from sources spread on a
    jittered grid (never one centre). Returns a signed field, roughly -1..1.
    ``decay`` > 0 attenuates with distance (1/(1+decay*r/wavelength))."""
    H, W = shape
    gx = int(math.ceil(math.sqrt(n_sources * W / H)))
    gy = int(math.ceil(n_sources / gx))
    cells = [(i, j) for i in range(gx) for j in range(gy)]
    rng.shuffle(cells)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = np.zeros((H, W), np.float32)
    for i, j in cells[:n_sources]:
        sx = (i + 0.5 + jitter * (rng.random() - 0.5)) / gx * W
        sy = (j + 0.5 + jitter * (rng.random() - 0.5)) / gy * H
        wl = wavelength * (0.75 + 0.5 * rng.random())
        r = np.hypot(xx - sx, yy - sy)
        att = 1.0 / (1.0 + decay * r / wl)
        f += att * np.sin(2 * math.pi * r / wl + rng.random() * 6.283)
    return f / max(1.0, math.sqrt(n_sources) * 0.7)


def resize_field(field, size) -> np.ndarray:
    """Resample a field to (H, W) = size[::-1] with bilinear interpolation."""
    W, H = size
    f = np.asarray(field, np.float32)
    if f.shape == (H, W):
        return f
    return ndimage.zoom(f, (H / f.shape[0], W / f.shape[1]), order=1)[:H, :W]


# --------------------------------------------------------------------------- lattices

@dataclass
class Dots:
    """A set of dot sites. ``cell`` is the local pitch (diameter available to
    the dot). ``row``/``col``/``grid_shape`` are set for grid lattices so a
    matrix can be read by index instead of interpolated."""
    x: np.ndarray
    y: np.ndarray
    cell: np.ndarray
    row: np.ndarray | None = None
    col: np.ndarray | None = None
    grid_shape: tuple[int, int] | None = None

    def __len__(self):
        return len(self.x)

    def subset(self, mask) -> "Dots":
        m = np.asarray(mask)
        return Dots(self.x[m], self.y[m], self.cell[m],
                    None if self.row is None else self.row[m], None if self.col is None else self.col[m], self.grid_shape)

    @staticmethod
    def concat(parts: list["Dots"]) -> "Dots":
        return Dots(np.concatenate([p.x for p in parts]), np.concatenate([p.y for p in parts]), np.concatenate([p.cell for p in parts]))


def square(size, pitch: float, angle: float = 0.0, offset=(0.0, 0.0), margin: float = 0.0) -> Dots:
    """Square lattice, optionally rotated by ``angle`` degrees (covers the frame)."""
    W, H = size
    R = math.hypot(W, H) / 2 + pitch
    n = int(R / pitch) + 2
    ii, jj = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    u = (ii.ravel() + offset[0]) * pitch
    v = (jj.ravel() + offset[1]) * pitch
    a = math.radians(angle)
    x = W / 2 + u * math.cos(a) - v * math.sin(a)
    y = H / 2 + u * math.sin(a) + v * math.cos(a)
    m = (x >= -margin) & (x <= W + margin) & (y >= -margin) & (y <= H + margin)
    return Dots(x[m].astype(np.float32), y[m].astype(np.float32), np.full(int(m.sum()), pitch, np.float32))


def hexagonal(size, pitch: float, angle: float = 0.0, margin: float = 0.0) -> Dots:
    """Hexagonal (triangular) lattice; nearest neighbours are ``pitch`` apart."""
    W, H = size
    R = math.hypot(W, H) / 2 + pitch
    h = pitch * math.sqrt(3) / 2
    nj = int(R / h) + 2
    ni = int(R / pitch) + 2
    ii, jj = np.meshgrid(np.arange(-ni, ni + 1), np.arange(-nj, nj + 1))
    u = (ii + 0.5 * (jj % 2)).ravel() * pitch
    v = jj.ravel() * h
    a = math.radians(angle)
    x = W / 2 + u * math.cos(a) - v * math.sin(a)
    y = H / 2 + u * math.sin(a) + v * math.cos(a)
    m = (x >= -margin) & (x <= W + margin) & (y >= -margin) & (y <= H + margin)
    return Dots(x[m].astype(np.float32), y[m].astype(np.float32), np.full(int(m.sum()), pitch, np.float32))


def grid(rows: int, cols: int, size, pitch: float | None = None, origin=None, fit: str = "contain") -> Dots:
    """rows x cols grid with row/col indices (for matrices, bitmaps, text).
    ``pitch`` defaults to fit the grid in the frame ('contain') or to cover it
    ('cover'); ``origin`` = top-left of the grid in pixels (default centred)."""
    W, H = size
    if pitch is None:
        pitch = min(W / cols, H / rows) if fit == "contain" else max(W / cols, H / rows)
    ox, oy = origin if origin is not None else ((W - cols * pitch) / 2, (H - rows * pitch) / 2)
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    return Dots((ox + (cc.ravel() + 0.5) * pitch).astype(np.float32), (oy + (rr.ravel() + 0.5) * pitch).astype(np.float32),
                np.full(rows * cols, pitch, np.float32), rr.ravel(), cc.ravel(), (rows, cols))


def polar(size, center, ring_pitch: float, dot_pitch: float | None = None, r_max: float | None = None, r_min: float = 0.0,
          phase: float = 0.0, twist: float = 0.0, rng=None) -> Dots:
    """Concentric rings of dots around ``center`` (pixels). Ring k has radius
    r_min + k*ring_pitch and as many dots as keep ~``dot_pitch`` arc spacing;
    ``twist`` rotates successive rings (radians per ring)."""
    W, H = size
    dot_pitch = dot_pitch or ring_pitch
    cx, cy = center
    r_max = r_max or math.hypot(max(cx, W - cx), max(cy, H - cy))
    xs, ys, cs = [], [], []
    k = 0 if r_min > 0 else 1
    if r_min <= 0:
        xs.append(np.array([cx])); ys.append(np.array([cy])); cs.append(np.array([ring_pitch]))
    while True:
        r = r_min + k * ring_pitch
        if r > r_max:
            break
        n = max(6, int(round(2 * math.pi * r / dot_pitch)))
        a = phase + k * twist + np.arange(n) * 2 * math.pi / n
        xs.append(cx + r * np.cos(a)); ys.append(cy + r * np.sin(a))
        cs.append(np.full(n, min(ring_pitch, 2 * math.pi * r / n)))
        k += 1
    x, y, c = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)
    m = (x >= -dot_pitch) & (x <= W + dot_pitch) & (y >= -dot_pitch) & (y <= H + dot_pitch)
    return Dots(x[m].astype(np.float32), y[m].astype(np.float32), c[m].astype(np.float32))


def poisson(size, min_dist: float, rng, k: int = 20, density=None) -> Dots:
    """Bridson Poisson-disk sampling (blue-noise stipple). ``density`` (field
    0..1, optional) shrinks the local radius where it is high (x0.35..1)."""
    W, H = size
    cs = min_dist / math.sqrt(2) * 0.35 if density is not None else min_dist / math.sqrt(2)
    gw, gh = int(W / cs) + 1, int(H / cs) + 1
    grid_ = -np.ones((gh, gw), np.int64)
    pts: list[tuple[float, float]] = []
    active: list[int] = []
    dens = None if density is None else resize_field(density, (W, H))

    def rad(x, y):
        if dens is None:
            return min_dist
        return min_dist * (1.0 - 0.65 * float(dens[min(int(y), H - 1), min(int(x), W - 1)]))

    def ok(x, y, r):
        gx, gy = int(x / cs), int(y / cs)
        rg = int(r / cs) + 1
        for j in range(max(0, gy - rg), min(gh, gy + rg + 1)):
            for i in range(max(0, gx - rg), min(gw, gx + rg + 1)):
                q = grid_[j, i]
                if q >= 0 and (pts[q][0] - x) ** 2 + (pts[q][1] - y) ** 2 < r * r:
                    return False
        return True

    def add(x, y):
        pts.append((x, y)); grid_[int(y / cs), int(x / cs)] = len(pts) - 1; active.append(len(pts) - 1)

    add(rng.random() * W, rng.random() * H)
    while active:
        ai = int(rng.integers(len(active)))
        px, py = pts[active[ai]]
        r0 = rad(px, py)
        found = False
        ang = rng.random(k) * 2 * math.pi
        rr = r0 * (1 + rng.random(k))
        for a, r in zip(ang, rr):
            x, y = px + r * math.cos(a), py + r * math.sin(a)
            if 0 <= x < W and 0 <= y < H and ok(x, y, rad(x, y)):
                add(x, y); found = True
                break
        if not found:
            active[ai] = active[-1]; active.pop()
    p = np.array(pts, np.float32)
    return Dots(p[:, 0], p[:, 1], np.full(len(p), min_dist, np.float32))


def stipple(density, size, n: int, rng, iters: int = 5, work: int = 260) -> Dots:
    """Weighted Voronoi stipple: ``n`` points whose local density follows
    ``density`` (field, higher = more dots), relaxed with Lloyd iterations on a
    coarse grid. ``cell`` = local spacing (so sizes scale with it)."""
    W, H = size
    sc = work / max(W, H)
    gw, gh = max(8, int(W * sc)), max(8, int(H * sc))
    d = resize_field(norm01(density), (gw, gh)) + 0.02
    p = (d / d.sum()).ravel()
    idx = rng.choice(gw * gh, size=n, p=p)
    ys, xs = np.divmod(idx, gw)
    pts = np.stack([xs + rng.random(n), ys + rng.random(n)], 1)
    gy, gx = np.mgrid[0:gh, 0:gw]
    gxy = np.stack([gx.ravel() + 0.5, gy.ravel() + 0.5], 1)
    wgt = d.ravel()
    for _ in range(iters):
        _, nn = cKDTree(pts).query(gxy)
        sw = np.bincount(nn, wgt, minlength=n)
        cx = np.bincount(nn, wgt * gxy[:, 0], minlength=n)
        cy = np.bincount(nn, wgt * gxy[:, 1], minlength=n)
        ok_ = sw > 0
        pts[ok_, 0] = cx[ok_] / sw[ok_]
        pts[ok_, 1] = cy[ok_] / sw[ok_]
    x = np.clip(pts[:, 0] / sc, 0, W - 1e-3)
    y = np.clip(pts[:, 1] / sc, 0, H - 1e-3)
    dist, _ = cKDTree(np.stack([x, y], 1)).query(np.stack([x, y], 1), k=2)
    cell = np.clip(dist[:, 1], 2, None)
    return Dots(x.astype(np.float32), y.astype(np.float32), cell.astype(np.float32))


def quadtree(field, size, max_cell: float, min_cell: float, rng=None, thresholds=(0.22, 0.38, 0.52, 0.66, 0.78), jitter: float = 0.2) -> Dots:
    """Adaptive squares: start from ``max_cell`` squares and quarter a square
    while the field at its centre exceeds the threshold for its depth (busy
    area = small dense squares, quiet area = big sparse ones). Returns centres
    with ``cell`` = side length. ``field`` is read in 0..1 (call norm01 first)."""
    W, H = size
    f = resize_field(norm01(field), (W, H))
    cur = [(gx * max_cell, gy * max_cell, max_cell) for gx in range(int(W / max_cell) + 1) for gy in range(int(H / max_cell) + 1)]
    out = []
    depth = 0
    while cur:
        a = np.array(cur, np.float32)
        cx = np.clip(a[:, 0] + a[:, 2] / 2, 0, W - 1).astype(int)
        cy = np.clip(a[:, 1] + a[:, 2] / 2, 0, H - 1).astype(int)
        v = f[cy, cx]
        th = thresholds[min(depth, len(thresholds) - 1)]
        if rng is not None:
            th = th * (1 + jitter * (rng.random(len(a)) - 0.5) * 2)
        split = (v > th) & (a[:, 2] / 2 >= min_cell)
        for s in a[~split]:
            out.append(s)
        nxt = []
        for x0, y0, s in a[split]:
            h = s / 2
            nxt += [(x0, y0, h), (x0 + h, y0, h), (x0, y0 + h, h), (x0 + h, y0 + h, h)]
        cur = nxt
        depth += 1
    o = np.array(out, np.float32)
    m = (o[:, 0] < W) & (o[:, 1] < H)
    o = o[m]
    return Dots(o[:, 0] + o[:, 2] / 2, o[:, 1] + o[:, 2] / 2, o[:, 2].copy())


def make_lattice(kind: str, size, pitch: float, rng=None, field=None, angle: float = 0.0) -> Dots:
    """String front door: square | hex | rotated | poisson | stipple | quadtree."""
    if kind == "square":
        return square(size, pitch, angle)
    if kind == "rotated":
        return square(size, pitch, angle or 30.0)
    if kind == "hex":
        return hexagonal(size, pitch, angle)
    if kind == "poisson":
        return poisson(size, pitch, rng)
    if kind == "stipple":
        n = int(size[0] * size[1] / (pitch * pitch) * 1.0)
        return stipple(field if field is not None else np.ones((8, 8), np.float32), size, n, rng)
    if kind == "quadtree":
        return quadtree(field, size, pitch * 4, pitch, rng)
    raise ValueError(f"unknown lattice {kind!r}")


def bsp_panels(size, rng, n: int = 12, min_frac: float = 0.16, gutter: int = 0) -> list[tuple[int, int, int, int]]:
    """Split the frame into ~n rectangles (x, y, w, h) by random binary
    subdivision of the largest panel; panels cover the frame exactly."""
    W, H = size
    panels = [(0, 0, W, H)]
    mw, mh = int(W * min_frac), int(H * min_frac)
    guard = 0
    while len(panels) < n and guard < 400:
        guard += 1
        panels.sort(key=lambda p: -p[2] * p[3] * (0.7 + 0.6 * rng.random()))
        x, y, w, h = panels.pop(0)
        vert = w / W * 1.5 > h / H if abs(w / W * 1.5 - h / H) > 0.2 else rng.random() < 0.5
        if vert and w >= 2 * mw:
            c = int(w * (0.3 + 0.4 * rng.random())); c = min(max(c, mw), w - mw)
            panels += [(x, y, c, h), (x + c, y, w - c, h)]
        elif not vert and h >= 2 * mh:
            c = int(h * (0.3 + 0.4 * rng.random())); c = min(max(c, mh), h - mh)
            panels += [(x, y, w, c), (x, y + c, w, h - c)]
        else:
            panels.append((x, y, w, h))
            if all(p[2] < 2 * mw and p[3] < 2 * mh for p in panels):
                break
    return panels


def palette_ramp(P, n: int = 256, stops=("edge", "structure", "accent")) -> np.ndarray:
    """Dot colour ramp from a wonhee.art Palette: dim structure -> structure ->
    accent (never reaches the canvas, so low values stay visible)."""
    return P.ramp(n, stops)


def source_field(ctx, blur: float = 1.0):
    """Luminance of ctx.source resized to the frame, or None."""
    if ctx.source is None:
        return None
    f = resize_field(luminance(ctx.source), ctx.size)
    return ndimage.gaussian_filter(f, blur) if blur else f


# --------------------------------------------------------------------------- sampling

def sample(field, dots: Dots, size) -> np.ndarray:
    """Field value at each dot: by index for a matching grid lattice, else bilinear."""
    f = np.asarray(field, np.float32)
    if f.ndim == 1 and len(f) == len(dots):      # per-dot values
        return f
    if dots.grid_shape is not None and f.shape == dots.grid_shape:
        return f[dots.row, dots.col]
    W, H = size
    yy = dots.y / H * (f.shape[0] - 1)
    xx = dots.x / W * (f.shape[1] - 1)
    return ndimage.map_coordinates(f, [np.clip(yy, 0, f.shape[0] - 1), np.clip(xx, 0, f.shape[1] - 1)], order=1, mode="nearest").astype(np.float32)


def ramp_lookup(ramp, v) -> np.ndarray:
    """ramp: (n,3); v in 0..1 -> (len(v),3)."""
    ramp = np.asarray(ramp, np.float32)
    return ramp[np.clip((np.asarray(v) * (len(ramp) - 1) + 0.5).astype(int), 0, len(ramp) - 1)]


# --------------------------------------------------------------------------- drawing

def new_canvas(size, color) -> np.ndarray:
    W, H = size
    c = np.empty((H, W, 3), np.float32)
    c[:] = np.asarray(color, np.float32)
    return c


def _sdf(shape: str, dx, dy, r, ang, thick):
    """Signed distance (pixels) of a dot of 'radius' r at offsets dx, dy."""
    if shape == "round":
        return np.hypot(dx, dy) - r
    if ang is not None:
        c, s = np.cos(ang), np.sin(ang)
        dx, dy = dx * c + dy * s, -dx * s + dy * c
    if shape == "square":
        cr = 0.22 * r
        qx, qy = np.abs(dx) - (r - cr), np.abs(dy) - (r - cr)
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - cr
    if shape == "diamond":
        return (np.abs(dx) + np.abs(dy)) * 0.70710678 - r * 1.05
    if shape == "line":
        L = r * 1.7
        px = np.clip(dx, -L, L)
        return np.hypot(dx - px, dy) - np.maximum(r * thick, 0.6)
    if shape == "plus":
        t = np.maximum(r * 0.34 * thick, 0.6)
        a = np.maximum(np.abs(dx) - r * 1.1, np.abs(dy) - t)
        b = np.maximum(np.abs(dy) - r * 1.1, np.abs(dx) - t)
        return np.minimum(a, b)
    if shape == "ring":
        return np.abs(np.hypot(dx, dy) - r * 0.78) - np.maximum(r * 0.2, 0.6)
    raise ValueError(f"unknown shape {shape!r}")


def draw_dots(canvas, x, y, r, rgb, alpha=1.0, shape: str = "round", angle=None, thick: float = 0.45, chunk: int = 3_000_000) -> None:
    """Paint dots onto ``canvas`` (H,W,3) in place with analytic anti-aliasing.
    x, y, r: per-dot arrays (r = radius in px); rgb: (3,) or (N,3); alpha scalar or (N,);
    angle: radians per dot (for square/line/plus/diamond), scalar or (N,)."""
    H, W, _ = canvas.shape
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    n = len(x)
    if n == 0:
        return
    r = np.broadcast_to(np.asarray(r, np.float32), (n,))
    rgb = np.broadcast_to(np.asarray(rgb, np.float32), (n, 3))
    alpha = np.broadcast_to(np.asarray(alpha, np.float32), (n,))
    ang = None if angle is None else np.broadcast_to(np.asarray(angle, np.float32), (n,))
    ext = {"round": 1.0, "square": 1.5, "diamond": 1.5, "line": 1.8, "plus": 1.2, "ring": 1.0}[shape]
    hb = np.ceil(r * ext).astype(int) + 2
    keep = (r > 0.05) & (alpha > 0.003) & (x > -hb) & (x < W + hb) & (y > -hb) & (y < H + hb)
    flat = canvas.reshape(-1, 3)
    for h in np.unique(hb[keep]):
        sel = np.nonzero(keep & (hb == h))[0]
        o = np.arange(-h, h + 1)
        per = (2 * h + 1) ** 2
        step = max(1, chunk // per)
        for s0 in range(0, len(sel), step):
            ids = sel[s0:s0 + step]
            cx = np.floor(x[ids]).astype(int); cy = np.floor(y[ids]).astype(int)
            X = cx[:, None, None] + o[None, None, :]
            Y = cy[:, None, None] + o[None, :, None]
            dx = (X + 0.5) - x[ids][:, None, None]
            dy = (Y + 0.5) - y[ids][:, None, None]
            d = _sdf(shape, dx, dy, r[ids][:, None, None], None if ang is None else ang[ids][:, None, None], thick)
            a = np.clip(0.5 - d, 0, 1) * alpha[ids][:, None, None]
            valid = (X >= 0) & (X < W) & (Y >= 0) & (Y < H) & (a > 0.002)
            lin = (Y * W + X)[valid]
            av = a[valid][:, None]
            cols = np.broadcast_to(rgb[ids][:, None, None, :], (len(ids), 2 * h + 1, 2 * h + 1, 3))[valid]
            flat[lin] = flat[lin] * (1 - av) + cols * av


def glow(canvas, layer, sigma: float, strength: float = 1.0) -> np.ndarray:
    """Screen a blurred copy of ``layer`` (H,W,3) over ``canvas`` (bloom)."""
    g = ndimage.gaussian_filter(layer, (sigma, sigma, 0)) * strength
    g += ndimage.gaussian_filter(layer, (sigma * 2.6, sigma * 2.6, 0)) * strength * 0.5
    return 1 - (1 - canvas) * (1 - np.clip(g, 0, 1))


def render_dots(field, size, lattice: Dots | str = "hex", *, pitch: float = 14.0, rng=None, bg=(0.1, 0.1, 0.1),
                ramp=None, color=(0.8, 0.8, 0.8), shape: str = "round", size_by=None, color_by=None, alpha_by=None,
                angle_by=None, r_min: float = 0.12, r_max: float = 0.5, gamma: float = 1.0, size_const: bool = False,
                alpha_min: float = 1.0, normalize: bool = True, off=None, off_r: float = 0.14, glow_sigma: float = 0.0,
                glow_strength: float = 0.8, core: float = 0.0, thick: float = 0.45, canvas=None, threshold: float | None = None) -> np.ndarray:
    """Render ANY field as a dot matrix.

    field      H x W (or rows x cols for a ``grid`` lattice) array; the default
               driver of size, colour and alpha.
    lattice    a ``Dots`` or a kind string (see ``make_lattice``).
    size_by / color_by / alpha_by / angle_by
               optional second fields (any shape; resampled): size by one
               field, colour by another. None = use ``field``. ``size_const``
               = constant size (brightness/colour only, LED style).
    r_min,r_max  radius range as a fraction of the dot's cell (0.5 = touching).
    ramp       (n,3) colour ramp for ``color_by`` values; else ``color``.
    alpha_min  alpha at field 0 (1 = opaque everywhere; <1 fades quiet dots).
    off        colour of the unlit dim dot drawn at every site first (LED look),
               radius ``off_r`` of the cell. ``glow_sigma``>0 adds bloom of the
               lit dots (px) and ``core``>0 a lighter small core in each lit dot.
    threshold  if given, dots with value < threshold are not lit (only 'off').
    """
    W, H = size
    if isinstance(lattice, str):
        lattice = make_lattice(lattice, size, pitch, rng, field=field)
    cv = new_canvas(size, bg) if canvas is None else canvas
    L = lattice
    v = sample(norm01(field) if normalize else field, L, size)
    sv = v if size_by is None else sample(norm01(size_by) if normalize else size_by, L, size)
    cv_ = v if color_by is None else sample(norm01(color_by) if normalize else color_by, L, size)
    av = None if alpha_by is None else sample(norm01(alpha_by) if normalize else alpha_by, L, size)
    ang = None if angle_by is None else sample(angle_by, L, size)
    if off is not None:
        draw_dots(cv, L.x, L.y, L.cell * off_r, off, 1.0, "round")
    sel = np.ones(len(L), bool) if threshold is None else (v >= threshold)
    sub = L.subset(sel)
    s = sv[sel]
    rad = sub.cell * (r_max if size_const else (r_min + (r_max - r_min) * np.power(np.clip(s, 0, 1), gamma)))
    if ramp is not None:
        cols = ramp_lookup(ramp, cv_[sel])
    else:
        cols = np.asarray(color, np.float32)
    al = np.ones(len(sub), np.float32) if av is None else (alpha_min + (1 - alpha_min) * np.clip(av[sel], 0, 1))
    if glow_sigma > 0:
        layer = np.zeros_like(cv)
        draw_dots(layer, sub.x, sub.y, rad, cols, al, shape, None if ang is None else ang[sel], thick)
        draw_dots(cv, sub.x, sub.y, rad, cols, al, shape, None if ang is None else ang[sel], thick)
        cv[...] = glow(cv, layer, glow_sigma, glow_strength)
    else:
        draw_dots(cv, sub.x, sub.y, rad, cols, al, shape, None if ang is None else ang[sel], thick)
    if core > 0:
        light = np.clip(np.asarray(cols, np.float32) * 0.45 + 0.55, 0, 1)
        draw_dots(cv, sub.x, sub.y, rad * 0.45, light, al * core, "round")
    return cv


# --------------------------------------------------------------------------- halftones

def _bayer(n: int) -> np.ndarray:
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (n * n)


def error_diffuse(v, kernel: str = "fs", serpentine: bool = True, level: float = 0.5) -> np.ndarray:
    """Error-diffusion binarisation of a 0..1 array (Floyd-Steinberg or
    Atkinson). Python loop: keep v small (<= ~150k cells)."""
    h, w = v.shape
    a = np.asarray(v, np.float64).copy()
    out = np.zeros((h, w), bool)
    if kernel == "atkinson":
        taps = [(1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)]
        wts = [1 / 8] * 6
    else:
        taps = [(1, 0), (-1, 1), (0, 1), (1, 1)]
        wts = [7 / 16, 3 / 16, 5 / 16, 1 / 16]
    for y in range(h):
        flip = serpentine and (y % 2 == 1)
        xs = range(w - 1, -1, -1) if flip else range(w)
        for x in xs:
            old = a[y, x]
            new = 1.0 if old >= level else 0.0
            out[y, x] = new > 0
            err = old - new
            for (dx, dy), wt in zip(taps, wts):
                xx, yy = (x - dx if flip else x + dx), y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    a[yy, xx] += err * wt
    return out


def halftone_mask(field, size, mode: str = "round", cell: float = 12.0, angle: float = 45.0, gamma: float = 1.0,
                  thickness: float = 1.0, bayer_n: int = 8, normalize: bool = True, warp=None) -> np.ndarray:
    """Halftone screen -> coverage mask (H,W) 0..1.

    mode: 'round' (AM dot screen at ``angle``), 'line' (line screen, width by
    field), 'hex' (hexagonally packed round dots), 'bayer' (ordered dither with
    ``bayer_n`` matrix and ``cell`` px per dither pixel), 'fs' / 'atkinson'
    (error diffusion on a ``cell`` px grid)."""
    W, H = size
    f = resize_field(field, (W, H)) if np.asarray(field).shape != (H, W) else np.asarray(field, np.float32)
    f = norm01(f) if normalize else f
    f = np.power(np.clip(f, 0, 1), gamma)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx += 0.5; yy += 0.5
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    if mode in ("round", "hex"):
        u = (xx - W / 2) * ca + (yy - H / 2) * sa
        v = -(xx - W / 2) * sa + (yy - H / 2) * ca
        if mode == "round":
            cu, cv_ = np.round(u / cell) * cell, np.round(v / cell) * cell
        else:
            hh = cell * math.sqrt(3) / 2
            j0 = np.round(v / hh); u0 = np.round(u / cell - 0.5 * (j0 % 2)) * cell + 0.5 * (j0 % 2) * cell
            v0 = j0 * hh
            j1 = j0 + np.where(v / hh > j0, 1, -1)
            u1 = np.round(u / cell - 0.5 * (j1 % 2)) * cell + 0.5 * (j1 % 2) * cell
            v1 = j1 * hh
            near0 = (u - u0) ** 2 + (v - v0) ** 2 <= (u - u1) ** 2 + (v - v1) ** 2
            cu, cv_ = np.where(near0, u0, u1), np.where(near0, v0, v1)
        px = W / 2 + cu * ca - cv_ * sa
        py = H / 2 + cu * sa + cv_ * ca
        val = ndimage.map_coordinates(f, [np.clip(py - 0.5, 0, H - 1), np.clip(px - 0.5, 0, W - 1)], order=1, mode="nearest")
        rad = cell * np.sqrt(np.clip(val, 0, 1) / math.pi) * (1.0 if mode == "round" else 1.0) * (1.08 if mode == "hex" else 1.0)
        d = np.hypot(u - cu, v - cv_)
        return np.clip(0.5 - (d - rad), 0, 1).astype(np.float32)
    if mode == "line":
        v = -(xx - W / 2) * sa + (yy - H / 2) * ca
        if warp is not None:                      # px displacement of the lines (curved screen)
            v = v + resize_field(warp, (W, H))
        cv_ = np.round(v / cell) * cell
        d = np.abs(v - cv_)
        half = 0.5 * cell * np.clip(f, 0, 1) * thickness
        return np.clip(0.5 - (d - half), 0, 1).astype(np.float32)
    k = max(1, int(round(cell)))
    ch, cw = (H + k - 1) // k, (W + k - 1) // k
    small = resize_field(f, (cw, ch))
    if mode == "bayer":
        T = _bayer(bayer_n)
        tt = np.tile(T, (ch // bayer_n + 1, cw // bayer_n + 1))[:ch, :cw]
        b = (small > tt)
    elif mode in ("fs", "atkinson"):
        b = error_diffuse(small, mode)
    else:
        raise ValueError(f"unknown halftone mode {mode!r}")
    return np.kron(b.astype(np.float32), np.ones((k, k), np.float32))[:H, :W]


def mix_mask(mask, bg, fg) -> np.ndarray:
    """Colour a coverage mask: bg where 0, fg where 1 (fg may be an (H,W,3) image)."""
    m = np.asarray(mask, np.float32)[..., None]
    return np.asarray(bg, np.float32) * (1 - m) + np.asarray(fg, np.float32) * m


# --------------------------------------------------------------------------- text / bitmap

_F57 = {
    "A": ".###.|#...#|#...#|#####|#...#|#...#|#...#", "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".###.|#...#|#....|#....|#....|#...#|.###.", "D": "####.|#...#|#...#|#...#|#...#|#...#|####.",
    "E": "#####|#....|#....|####.|#....|#....|#####", "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".###.|#...#|#....|#.###|#...#|#...#|.###.", "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": ".###.|..#..|..#..|..#..|..#..|..#..|.###.", "J": "..###|...#.|...#.|...#.|...#.|#..#.|.##..",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#", "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#", "N": "#...#|##..#|#.#.#|#..##|#...#|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.", "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "Q": ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#", "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".####|#....|#....|.###.|....#|....#|####.", "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.", "V": "#...#|#...#|#...#|#...#|#...#|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#.#.#|#.#.#|##.##|#...#", "X": "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..", "Z": "#####|....#|...#.|..#..|.#...|#....|#####",
    "0": ".###.|#...#|#..##|#.#.#|##..#|#...#|.###.", "1": "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
    "2": ".###.|#...#|....#|...#.|..#..|.#...|#####", "3": "####.|....#|....#|.###.|....#|....#|####.",
    "4": "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.", "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": ".###.|#....|#....|####.|#...#|#...#|.###.", "7": "#####|....#|...#.|..#..|.#...|.#...|.#...",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.", "9": ".###.|#...#|#...#|.####|....#|....#|.###.",
    " ": ".....|.....|.....|.....|.....|.....|.....", ".": ".....|.....|.....|.....|.....|.##..|.##..",
    ",": ".....|.....|.....|.....|.##..|..#..|.#...", ":": ".....|.##..|.##..|.....|.##..|.##..|.....",
    "-": ".....|.....|.....|#####|.....|.....|.....", "+": ".....|..#..|..#..|#####|..#..|..#..|.....",
    "/": "....#|....#|...#.|..#..|.#...|#....|#....", "%": "##..#|##.#.|...#.|..#..|.#...|#.#.#|#..##",
    "!": "..#..|..#..|..#..|..#..|..#..|.....|..#..", "?": ".###.|#...#|....#|...#.|..#..|.....|..#..",
    "*": ".....|#.#.#|.###.|#####|.###.|#.#.#|.....", "#": ".#.#.|.#.#.|#####|.#.#.|#####|.#.#.|.#.#.",
    "=": ".....|.....|#####|.....|#####|.....|.....", "<": "...#.|..#..|.#...|#....|.#...|..#..|...#.",
    ">": ".#...|..#..|...#.|....#|...#.|..#..|.#...", "_": ".....|.....|.....|.....|.....|.....|#####",
    "x": ".....|.....|#...#|.#.#.|..#..|.#.#.|#...#", "^": "..#..|.#.#.|#...#|.....|.....|.....|.....",
    "@": ".###.|#...#|#.###|#.#.#|#.###|#....|.####", "'": "..#..|..#..|.#...|.....|.....|.....|.....",
    "(": "...#.|..#..|.#...|.#...|.#...|..#..|...#.", ")": ".#...|..#..|...#.|...#.|...#.|..#..|.#...",
    "[": ".###.|.#...|.#...|.#...|.#...|.#...|.###.", "]": ".###.|...#.|...#.|...#.|...#.|...#.|.###.",
    "■": "#####|#####|#####|#####|#####|#####|#####", "●": ".###.|#####|#####|#####|#####|#####|.###.",
    "→": "..#..|...#.|#####|...#.|..#..|.....|.....", "↑": "..#..|.###.|#.#.#|..#..|..#..|..#..|..#..",
    "♥": ".#.#.|#####|#####|#####|.###.|..#..|.....",
}
FONT_5X7: dict[str, np.ndarray] = {k: np.array([[c == "#" for c in row] for row in v.split("|")], bool) for k, v in _F57.items()}


def text_matrix(text: str, *, font: str = "5x7", spacing: int = 1, rows: int | None = None, ttf: str | None = None,
                threshold: float = 0.5, lines_gap: int = 2) -> np.ndarray:
    """Rasterise ``text`` to a boolean (rows x cols) dot matrix.

    font='5x7' uses the built-in bitmap glyphs (A-Z 0-9 punctuation, a few
    symbols; lower case maps to upper; unknown chars become blanks). Newlines
    stack lines. With ``ttf`` (path to any TrueType font) or font='ttf' the
    string is drawn with PIL and area-averaged down to ``rows`` dot rows
    (default 9), then thresholded: good for names with unusual shapes."""
    if ttf or font == "ttf":
        return _ttf_matrix(text, ttf, rows or 9, threshold, spacing)
    lines = []
    for ln in text.split("\n"):
        cols = []
        for ch in ln:
            g = FONT_5X7.get(ch) if ch in FONT_5X7 else FONT_5X7.get(ch.upper(), FONT_5X7[" "])
            cols.append(g)
            cols.append(np.zeros((7, spacing), bool))
        lines.append(np.concatenate(cols[:-1], 1) if cols else np.zeros((7, 1), bool))
    w = max(l.shape[1] for l in lines)
    parts = []
    for i, l in enumerate(lines):
        parts.append(np.pad(l, ((0, 0), (0, w - l.shape[1]))))
        if i < len(lines) - 1:
            parts.append(np.zeros((lines_gap, w), bool))
    return np.concatenate(parts, 0)


def _ttf_matrix(text, ttf, rows, threshold, spacing):
    from PIL import Image, ImageDraw, ImageFont
    import os
    path = ttf
    if path is None:
        for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"):
            if os.path.exists(cand):
                path = cand
                break
    S = 8
    big = rows * S * 2
    font = ImageFont.truetype(path, big) if path else ImageFont.load_default()
    l, t, r, b = font.getbbox(text)
    im = Image.new("L", (r - l + 4, b - t + 4), 0)
    ImageDraw.Draw(im).text((2 - l, 2 - t), text, fill=255, font=font)
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.1)
    if len(ys) == 0:
        return np.zeros((rows, 1), bool)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cols = max(1, int(round(a.shape[1] * rows / a.shape[0])))
    small = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((cols, rows), Image.BOX), np.float32) / 255
    return small > threshold


def tile_text(text: str, rows_total: int, cols_total: int, *, row0: int = 0, col0: int = 0, gap: int = 4, **kw) -> np.ndarray:
    """Repeat a single-line text matrix with ``gap`` blank columns to fill a
    cols_total-wide band (a marquee), starting at column offset col0, with the
    glyph rows starting at row0 inside a rows_total-tall band."""
    m = text_matrix(text, **kw)
    unit = np.pad(m, ((0, 0), (0, gap)))
    reps = cols_total // unit.shape[1] + 2
    band = np.tile(unit, (1, reps))
    band = np.roll(band, -col0 % unit.shape[1], axis=1)[:, :cols_total]
    out = np.zeros((rows_total, cols_total), bool)
    h = min(m.shape[0], rows_total - row0)
    if h > 0:
        out[row0:row0 + h] = band[:h]
    return out
