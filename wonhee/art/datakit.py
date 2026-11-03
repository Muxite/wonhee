"""datakit: shared helpers for the data_* (matrix / information texture) and multichan_* effects.

Not an effect module. Everything is numpy/scipy/cv2/PIL, deterministic in (seed, params, data).

Contents
  colour    tested sequential / diverging / categorical scales built in OKLab (``lut``), lightness-monotone
            checks (``check_scale``), palette-keyed default scale, light and dark variants
  data      ``prepare_matrix`` (ctx.data -> matrix + labels + provenance), synthetic generators (always
            flagged ``synthetic``), ``normalize`` with explicit ranges, ``cluster_order`` (hierarchical order
            with optimal leaf ordering), ``aggregate`` (area-average to fewer cells)
  geometry  ``edges`` (cell boundaries with seeded proportion variation / growth / bleed offset)
  render    ``render_cells`` (vectorised cell field with gaps, surfaces), ``Overlay`` (text, legend, brackets)
  fields    ``standin_fields`` (physics / noise scalar fields standing in for real channels), ``norm_range``
  provenance ``provenance`` (metadata dict: synthetic / standin / aesthetic flags)
"""
from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.cluster import hierarchy

from . import palette as pal
from .features import from_oklab, to_oklab

# ----------------------------------------------------------------------------------------------------------------
# colour scales
# ----------------------------------------------------------------------------------------------------------------
# sequential: hue waypoints (degrees, unwrapped), chroma peak, lightness gamma
SEQ = {
    "magma":   dict(hues=[300, 350, 385, 445], c=0.19, g=1.0),
    "ember":   dict(hues=[25, 45, 70, 90], c=0.15, g=1.0),
    "ocean":   dict(hues=[265, 235, 205, 175], c=0.13, g=1.0),
    "kelp":    dict(hues=[300, 255, 195, 125], c=0.14, g=1.0),
    "ice":     dict(hues=[255, 235, 215], c=0.085, g=1.0),
    "rose":    dict(hues=[345, 5, 25, 55], c=0.16, g=1.0),
    "cream-red": dict(hues=[88, 60, 33, 27], c=0.20, g=1.05),
    "ink":     dict(hues=[270, 265, 258], c=0.12, g=1.0),
    "mono":    dict(hues=[60, 60], c=0.012, g=1.0),
}
DIV = {
    "blue-red":    (250, 25),
    "teal-rose":   (195, 355),
    "violet-gold": (300, 85),
    "green-plum":  (150, 330),
}
SEQ_L = {"dark": (0.21, 0.95), "light": (0.925, 0.34)}
DIV_L = {"dark": (0.20, 0.82), "light": (0.965, 0.40)}      # centre, ends


def _rgb(L, C, h):
    return pal.oklch_to_rgb(L, C, h % 360)


@functools.lru_cache(maxsize=256)
def _seq_lut(name: str, polarity: str, n: int) -> np.ndarray:
    s = SEQ[name]
    l0, l1 = SEQ_L[polarity]
    t = np.linspace(0, 1, n)
    hue = np.interp(t, np.linspace(0, 1, len(s["hues"])), s["hues"])
    # chroma: small at the ends, peaked in the middle third
    f1 = 0.78 if polarity == "light" else 0.30          # dark end of a light scale keeps its hue (cream -> red, not brown)
    ch = s["c"] * np.maximum(np.sin(np.pi * np.clip(t, 0, 1)) ** 0.8, 0.16 * (1 - t) + f1 * t)
    if name in ("mono",):
        ch = np.full(n, s["c"])
    L = l0 + (l1 - l0) * t ** s["g"]
    return np.stack([_rgb(L[i], ch[i], hue[i]) for i in range(n)]).astype(np.float32)


@functools.lru_cache(maxsize=128)
def _div_lut(name: str, polarity: str, n: int, cmax: float = 0.17) -> np.ndarray:
    hA, hB = DIV[name]
    c0, e0 = DIV_L[polarity]
    t = np.linspace(-1, 1, n)
    out = []
    for v in t:
        a = abs(v)
        L = c0 + (e0 - c0) * a ** 0.9
        C = 0.012 + cmax * np.sin(np.clip(a, 0, 1) * np.pi * 0.62) ** 0.9
        out.append(_rgb(L, C, hA if v < 0 else hB))
    return np.stack(out).astype(np.float32)


def seq_from_hue(h: float, polarity: str = "dark", n: int = 256, spread: float = 75.0, c: float = 0.16) -> np.ndarray:
    """A multi-hue sequential scale keyed on one hue: sweeps ``spread`` degrees across the scale."""
    l0, l1 = SEQ_L[polarity]
    t = np.linspace(0, 1, n)
    hue = h - spread * 0.5 + spread * t
    ch = c * (0.18 + 0.82 * np.sin(np.pi * t) ** 0.8)
    L = l0 + (l1 - l0) * t
    return np.stack([_rgb(L[i], ch[i], hue[i]) for i in range(n)]).astype(np.float32)


def lut_from_stops(stops, n: int = 256) -> np.ndarray:
    """Interpolate RGB stops in OKLab (perceptually even)."""
    lab = to_oklab(np.asarray(stops, np.float32)[None])[0]
    t = np.linspace(0, len(lab) - 1, n)
    i = np.minimum(t.astype(int), len(lab) - 2)
    f = (t - i)[:, None]
    mixed = lab[i] * (1 - f) + lab[i + 1] * f
    return np.clip(from_oklab(mixed[None])[0], 0, 1).astype(np.float32)


def lut(spec, palette=None, n: int = 256, polarity: str | None = None) -> np.ndarray:
    """Resolve a scale spec to an n x 3 table. spec: name in SEQ / DIV, 'auto' (keyed on the palette's
    structure hue), 'palette' (canvas-near -> structure -> accent), an n x 3 array, or a list of hex/RGB stops."""
    pol = polarity or (palette.polarity if palette is not None else "dark")
    if isinstance(spec, np.ndarray) and spec.ndim == 2:
        return spec.astype(np.float32)
    if isinstance(spec, (list, tuple)) and spec and not isinstance(spec, str):
        stops = [pal.hex_to_rgb(s) if isinstance(s, str) else np.asarray(s, np.float32) for s in spec]
        return lut_from_stops(stops, n)
    spec = str(spec)
    if spec in SEQ:
        return _seq_lut(spec, pol, n)
    if spec in DIV:
        return _div_lut(spec, pol, n)
    if spec == "palette" and palette is not None:
        lo = palette.canvas + (palette.structure - palette.canvas) * 0.35
        return lut_from_stops([lo, palette.structure, palette.accent], n)
    # auto: keyed on the palette
    h = pal.rgb_to_oklch(palette.structure)[2] if palette is not None else 40.0
    return seq_from_hue(h, pol, n)


def categorical(n: int, polarity: str = "dark", h0: float = 25.0, c: float = 0.14) -> np.ndarray:
    """n categorical colours: hues spread evenly (offset so neighbours differ a lot), lightness alternating
    in two bands so they also separate in greyscale."""
    order = [(i * 5) % n if math.gcd(5, n) == 1 else i for i in range(n)]
    Ls = (0.64, 0.78) if polarity == "dark" else (0.50, 0.62)
    cols = [None] * n
    for k, i in enumerate(order):
        cols[i] = _rgb(Ls[k % 2], c, h0 + 360.0 * k / n)
    return np.stack(cols).astype(np.float32)


def lightness(rgb) -> np.ndarray:
    return to_oklab(np.asarray(rgb, np.float32).reshape(1, -1, 3))[0, :, 0]


def check_scale(tab: np.ndarray, kind: str = "sequential") -> dict:
    """Quality of a colour table: sequential -> lightness monotone (violations), diverging -> symmetric lightness
    about the centre; always reports the min lightness step ('flat') and gamut (all in 0..1)."""
    L = lightness(tab)
    d = np.diff(L)
    out = {"L_min": float(L.min()), "L_max": float(L.max()), "range": float(np.ptp(L)),
           "in_gamut": bool(tab.min() >= -1e-6 and tab.max() <= 1 + 1e-6)}
    if kind == "sequential":
        s = np.sign(L[-1] - L[0])
        out["monotone"] = bool((d * s >= -2e-3).all())
    else:
        h = len(L) // 2
        out["symmetric"] = bool(np.abs(L[:h] - L[::-1][:h]).max() < 0.06)
        out["monotone"] = True
    return out


def apply_lut(v: np.ndarray, tab: np.ndarray) -> np.ndarray:
    n = len(tab)
    idx = np.clip(np.nan_to_num(v, nan=0.0) * (n - 1), 0, n - 1)
    i = idx.astype(np.int32)
    j = np.minimum(i + 1, n - 1)
    f = (idx - i)[..., None]
    return tab[i] * (1 - f) + tab[j] * f


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Prep:
    M: np.ndarray                      # R x C float, NaN = missing
    rows: list | None = None
    cols: list | None = None
    synthetic: bool = False
    kind: str = "scores"               # scores | corr | field
    name: str = ""
    group_rows: np.ndarray | None = None
    group_cols: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def provenance(name: str, data=None, **kw) -> dict:
    """Metadata a caller can attach next to the rendered image. ``synthetic`` is True whenever the effect had to
    invent its own values (ctx.data None): such an image is decoration, never a result."""
    d = {"effect": name, "synthetic": data is None}
    d["note"] = ("decorative synthetic data; not measurements or results" if data is None
                 else "values come from the supplied ctx.data")
    d.update(kw)
    return d


def synth_bench(rng, R=38, C=9) -> Prep:
    """Benchmark-like scores: column skill x row difficulty with block structure, pass fractions in 1/k steps
    (clearly not results: the labels are generic)."""
    skill = np.sort(rng.normal(0, 1.1, C))[::-1]
    diff = rng.normal(0, 1.0, R)
    k = rng.integers(2, 5, C)
    fam = rng.integers(0, 3, R)
    inter = rng.normal(0, 0.8, (3, C))
    z = 0.8 * skill[None, :] - diff[:, None] + inter[fam] - 0.15
    p = 1 / (1 + np.exp(-1.8 * z))
    n = rng.choice([3, 4, 6, 12], size=(R, C))
    M = rng.binomial(n, p) / n
    perm = rng.permutation(R)
    return Prep(M[perm].astype(np.float32), [f"r{i + 1:02d}" for i in range(R)], [f"c{j + 1}" for j in range(C)],
                True, "scores", "synthetic-bench", fam[perm])


def synth_corr(rng, n=24, k=4) -> Prep:
    """Correlation matrix from k latent factors, rows shuffled (so ordering/clustering has work to do)."""
    load = rng.normal(0, 1, (n, k)) * (rng.random((n, k)) < 0.35)
    owner = rng.integers(0, k, n)
    load[np.arange(n), owner] += rng.uniform(1.0, 2.0, n) * rng.choice([-1, 1], n, p=[0.2, 0.8])
    X = rng.normal(0, 1, (k, 400))
    S = load @ X + rng.normal(0, 1.4, (n, 400))
    M = np.corrcoef(S)
    perm = rng.permutation(n)
    M = M[np.ix_(perm, perm)]
    lab = [f"v{i + 1:02d}" for i in range(n)]
    return Prep(M.astype(np.float32), lab, lab, True, "corr", "synthetic-corr", owner[perm], owner[perm])


def synth_field(rng, R=30, C=40) -> Prep:
    """Smooth structured field with row and column effects (for pixel fields)."""
    base = ndimage.gaussian_filter(rng.normal(0, 1, (R, C)), (R / 9, C / 11), mode="wrap")
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    rowfx = ndimage.gaussian_filter(rng.normal(0, 1, R), 1.2)[:, None] * 0.18
    colfx = ndimage.gaussian_filter(rng.normal(0, 1, C), 1.2)[None, :] * 0.18
    M = np.clip(base + rowfx + colfx + rng.normal(0, 0.05, (R, C)), 0, 1)
    return Prep(M.astype(np.float32), None, None, True, "field", "synthetic-field")


def prepare_matrix(ctx, synth: str = "bench", shape=(38, 9)) -> Prep:
    """ctx.data -> Prep. Accepted: 2D array; {"matrix"|"scores"|"values": 2D (3D scores are averaged over the last
    axis), "rows": [...], "cols": [...], "mask": bool}; a dict of equal-length 1D series (rows = series).
    None -> a seeded synthetic matrix of kind ``synth`` (bench | corr | field), flagged ``synthetic``."""
    d = ctx.data
    rng = ctx.rng
    if d is None:
        if synth == "corr":
            return synth_corr(rng, shape[0], 4)
        if synth == "field":
            return synth_field(rng, shape[0], shape[1])
        return synth_bench(rng, shape[0], shape[1])
    rows = cols = None
    mask = None
    if isinstance(d, dict):
        key = next((k for k in ("matrix", "scores", "values", "data") if k in d), None)
        if key is None:
            ks = list(d.keys())
            arr = np.array([np.asarray(d[k], np.float32) for k in ks])
            rows = ks
            M = arr
        else:
            M = np.asarray(d[key], np.float32)
            rows, cols, mask = d.get("rows"), d.get("cols"), d.get("mask")
            if rows is not None and len(rows) and isinstance(rows[0], dict):
                rows = [str(r.get("id", r)) for r in rows]
            if cols is not None and len(cols) and isinstance(cols[0], dict):
                cols = [str(c.get("variant", c.get("model", c))) for c in cols]
    else:
        M = np.asarray(d, np.float32)
    if M.ndim == 3:
        M = np.nanmean(M, axis=-1)
    if M.ndim == 1:
        M = M[None]
    if mask is not None:
        M = np.where(np.asarray(mask, bool), M, np.nan)
    return Prep(M.astype(np.float32), rows, cols, False, "scores", "data")


def normalize(M: np.ndarray, lo=None, hi=None, mode: str = "robust", center: float | None = None):
    """-> (0..1 array, (lo, hi) used). Explicit lo/hi win and are returned (so legends tell the truth).
    mode: robust (1..99 pct) | minmax | rank. ``center`` makes the range symmetric about it (diverging)."""
    fin = np.isfinite(M)
    if lo is None or hi is None:
        if not fin.any():
            lo, hi = 0.0, 1.0
        elif mode == "minmax":
            lo, hi = float(M[fin].min()), float(M[fin].max())
        elif mode == "rank":
            r = np.full(M.shape, np.nan, np.float32)
            order = np.argsort(np.argsort(M[fin]))
            r[fin] = order / max(1, len(order) - 1)
            return r, (0.0, 1.0)
        else:
            lo, hi = [float(x) for x in np.percentile(M[fin], [1, 99])]
    if center is not None:
        a = max(abs(lo - center), abs(hi - center))
        lo, hi = center - a, center + a
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return np.clip((M - lo) / (hi - lo), 0, 1).astype(np.float32), (float(lo), float(hi))


def cluster_order(M: np.ndarray, axis: int = 0, method: str = "average", metric: str = "euclidean"):
    """Hierarchical ordering of rows (axis 0) or columns (axis 1) with optimal leaf ordering. NaN -> column mean.
    Returns (order, linkage Z)."""
    A = np.asarray(M, np.float64)
    if axis == 1:
        A = A.T
    if len(A) < 3:
        return np.arange(len(A)), None
    A = np.where(np.isfinite(A), A, np.nanmean(A) if np.isfinite(A).any() else 0.0)
    Z = hierarchy.linkage(A, method=method, metric=metric)
    Z = hierarchy.optimal_leaf_ordering(Z, A)
    return np.asarray(hierarchy.leaves_list(Z)), Z


def order_matrix(P: Prep, how: str) -> Prep:
    """how: data (unchanged) | mean (rows by mean desc, cols by mean desc) | cluster (both axes) | rows | cols."""
    M = P.M
    ro = np.arange(M.shape[0]); co = np.arange(M.shape[1])
    if how == "mean":
        ro = np.argsort(-np.nanmean(M, 1)); co = np.argsort(-np.nanmean(M, 0))
    elif how in ("cluster", "rows", "cols"):
        if how in ("cluster", "rows"):
            ro, _ = cluster_order(M, 0)
        if how in ("cluster", "cols"):
            if P.kind == "corr" and M.shape[0] == M.shape[1]:
                co = ro
            else:
                co, _ = cluster_order(M, 1)
    Q = Prep(M[np.ix_(ro, co)], [P.rows[i] for i in ro] if P.rows else None, [P.cols[i] for i in co] if P.cols else None,
             P.synthetic, P.kind, P.name, P.group_rows[ro] if P.group_rows is not None else None,
             P.group_cols[co] if P.group_cols is not None else None, dict(P.extra))
    Q.extra["row_order"], Q.extra["col_order"] = ro, co
    return Q


def aggregate(M: np.ndarray, R: int, C: int) -> np.ndarray:
    """Area-average M to at most R x C cells (NaN aware). Never upsamples."""
    R, C = max(1, min(R, M.shape[0])), max(1, min(C, M.shape[1]))
    if (R, C) == M.shape:
        return M
    fin = np.isfinite(M).astype(np.float32)
    v = np.where(np.isfinite(M), M, 0).astype(np.float32)
    a = cv2.resize(v, (C, R), interpolation=cv2.INTER_AREA)
    w = cv2.resize(fin, (C, R), interpolation=cv2.INTER_AREA)
    return np.where(w > 0.3, a / np.maximum(w, 1e-6), np.nan).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------------------------
def edges(n: int, total: float, rng, var: float = 0.0, growth: float = 1.0, offset: float = 0.0,
          weights: np.ndarray | None = None, bleed: bool = True) -> np.ndarray:
    """Boundaries (n+1 values, pixels) of n cells that together span ``total`` px. ``var`` = lognormal spread of
    cell sizes (varied proportions), ``growth`` = geometric ratio between neighbours (cells swell along the
    axis), ``weights`` multiply sizes (data-driven proportions). With ``bleed`` the span is shifted by ``offset``
    (fraction of the first cell, negative start) so the outer cells are cut by the frame."""
    w = np.exp(rng.normal(0, var, n)) if var > 0 else np.ones(n)
    if weights is not None:
        w = w * np.asarray(weights, float)
    if abs(growth - 1) > 1e-6:
        w = w * growth ** np.arange(n)
    w = w / w.sum()
    e = np.concatenate([[0], np.cumsum(w)]) * total
    if bleed and offset:
        e = e - offset * (e[1] - e[0])
    return e


def render_cells(V: np.ndarray, row_e: np.ndarray, col_e: np.ndarray, size, tab: np.ndarray, canvas,
                 gap: float = 2.0, gap_color=None, surface: str = "flat", surf_amt: float = 0.5, rng=None,
                 nan_color=None, cell_jitter: float = 0.0, vmap=None) -> tuple[np.ndarray, dict]:
    """Paint a value matrix V (0..1, NaN = missing) into the frame using explicit cell boundaries. Pixels outside
    the boundaries get the canvas colour. Returns (img, info) with per-pixel cell indices for overlays."""
    W, H = size
    R, C = V.shape
    ys = np.arange(H) + 0.5
    xs = np.arange(W) + 0.5
    ri = np.searchsorted(row_e, ys) - 1
    ci = np.searchsorted(col_e, xs) - 1
    vr = (ri >= 0) & (ri < R)
    vc = (ci >= 0) & (ci < C)
    rr = np.clip(ri, 0, R - 1); cc = np.clip(ci, 0, C - 1)
    Vv = V
    if cell_jitter > 0 and rng is not None:
        Vv = np.clip(V + rng.normal(0, cell_jitter, V.shape), 0, 1)
    val = Vv[rr][:, cc]
    img = apply_lut(val, tab)
    nanm = ~np.isfinite(val)
    if nanm.any():
        nc = nan_color if nan_color is not None else np.asarray(canvas) * 0.85 + img.mean((0, 1)) * 0.15
        img[nanm] = nc
    # distances to cell borders (per axis, pixels), for gaps and surfaces
    dy = np.minimum(ys - row_e[rr], row_e[rr + 1] - ys)
    dx = np.minimum(xs - col_e[cc], col_e[cc + 1] - xs)
    h_c = (row_e[rr + 1] - row_e[rr])[:, None]
    w_c = (col_e[cc + 1] - col_e[cc])[None, :]
    inside = np.minimum(dy[:, None], dx[None, :])
    if surface in ("glass", "paper", "bevel"):
        u = np.clip(dy[:, None] / np.maximum(h_c * 0.5, 1), 0, 1)
        vv = ((ys - row_e[rr]) / np.maximum(row_e[rr + 1] - row_e[rr], 1))[:, None]
        if surface == "glass":
            shade = 1 + surf_amt * 0.18 * (0.5 - vv) + surf_amt * 0.10 * np.exp(-((vv - 0.08) / 0.07) ** 2)
            img = img * shade[..., None]
        elif surface == "bevel":
            edge = np.clip(1 - inside / max(1.5, 0.08 * min(h_c.mean(), w_c.mean())), 0, 1)
            img = img * (1 - surf_amt * 0.22 * edge)[..., None]
        else:
            if rng is not None:
                g = rng.normal(0, 1, (H, W)).astype(np.float32)
                g = cv2.GaussianBlur(g, (0, 0), 0.8)
                img = img * (1 + surf_amt * 0.06 * g)[..., None] + surf_amt * 0.012 * 0
    gc = np.asarray(gap_color if gap_color is not None else canvas, np.float32)
    if gap > 0:
        cov = np.clip(inside - gap / 2 + 0.5, 0, 1)[..., None]
        img = img * cov + gc * (1 - cov)
    out = ~(vr[:, None] & vc[None, :])
    if out.any():
        img[out] = np.asarray(canvas, np.float32)
    return np.clip(img, 0, 1).astype(np.float32), dict(ri=ri, ci=ci)


# ----------------------------------------------------------------------------------------------------------------
# text / overlay
# ----------------------------------------------------------------------------------------------------------------
_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"]


@functools.lru_cache(maxsize=32)
def font(px: int):
    for f in _FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, max(6, int(px)))
    return ImageFont.load_default(size=max(6, int(px)))


def ink_for(rgb, dark_ink=(0.08, 0.08, 0.09), light_ink=(0.95, 0.94, 0.92)):
    return dark_ink if float(lightness(rgb)[0]) > 0.62 else light_ink


class Overlay:
    """Draw text / lines / legends on a float image (PIL under the hood). Call ``.done()`` for the array."""

    def __init__(self, img: np.ndarray):
        self.im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).convert("RGBA")
        self.lay = Image.new("RGBA", self.im.size, (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.lay)
        self.W, self.H = self.im.size

    @staticmethod
    def _c(rgb, a=1.0):
        return tuple(int(round(float(np.clip(v, 0, 1)) * 255)) for v in rgb) + (int(round(a * 255)),)

    def text(self, xy, s, px, rgb, a=1.0, anchor="la"):
        self.d.text(xy, s, font=font(px), fill=self._c(rgb, a), anchor=anchor)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, w=1, a=1.0, radius=0):
        # Tiny output sizes can give reversed extents; PIL requires x1>=x0, y1>=y0.
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        kw = dict(fill=self._c(fill, a) if fill is not None else None, outline=self._c(outline, a) if outline is not None else None, width=w)
        if radius:
            self.d.rounded_rectangle([x0, y0, x1, y1], radius=radius, **kw)
        else:
            self.d.rectangle([x0, y0, x1, y1], **kw)

    def line(self, pts, rgb, w=1, a=1.0):
        self.d.line(pts, fill=self._c(rgb, a), width=w)

    def colorbar(self, x0, y0, x1, y1, tab, lo, hi, ink, fmt="{:.2f}", px=11, ticks=3, label=None, vertical=False):
        n = 128
        ramp = apply_lut(np.linspace(0, 1, n)[None, :], tab)[0]
        for i in range(n):
            if vertical:
                ya = y1 - (y1 - y0) * (i + 1) / n; yb = y1 - (y1 - y0) * i / n
                self.d.rectangle([min(x0, x1), min(ya, yb), max(x0, x1), max(ya, yb)], fill=self._c(ramp[i]))
            else:
                xa = x0 + (x1 - x0) * i / n; xb = x0 + (x1 - x0) * (i + 1) / n
                self.d.rectangle([min(xa, xb), min(y0, y1), max(xa, xb), max(y0, y1)], fill=self._c(ramp[i]))
        self.rect(x0, y0, x1, y1, outline=ink, w=1, a=0.55)
        for k in range(ticks):
            f = k / max(1, ticks - 1)
            v = lo + (hi - lo) * f
            if vertical:
                y = y1 - (y1 - y0) * f
                self.line([(x1, y), (x1 + 4, y)], ink, 1, 0.8)
                self.text((x1 + 7, y), fmt.format(v), px, ink, 0.9, anchor="lm")
            else:
                x = x0 + (x1 - x0) * f
                self.line([(x, y1), (x, y1 + 4)], ink, 1, 0.8)
                self.text((x, y1 + 6), fmt.format(v), px, ink, 0.9, anchor="ma")
        if label:
            self.text((x0, y0 - 5), label, px, ink, 0.9, anchor="ld")

    def done(self) -> np.ndarray:
        out = Image.alpha_composite(self.im, self.lay).convert("RGB")
        return np.asarray(out, np.float32) / 255


def scale_px(W: int, base: float = 1500.0) -> float:
    """Size factor for pixel-defined details (gaps, fonts) so the composition is the same at any render size."""
    return W / base


# ----------------------------------------------------------------------------------------------------------------
# scalar fields for multichannel work
# ----------------------------------------------------------------------------------------------------------------
STANDIN = ("physics-vortex", "physics-fieldlines", "physics-wavemedium", "physics-caustics", "physics-marbling", "noise:70",
           "physics-chladni", "physics-interference")


def standin_fields(n: int, size, seed: int, which=None, max_w: int = 640, smooth: float = 0.0) -> tuple[dict, list]:
    """n scalar fields (0..1, float32 at <= max_w wide, frame aspect) taken from the luminance of library
    physics effects (names 'physics-*'), or smooth noise ('noise:SCALE' with SCALE in px at the working size).
    ``smooth`` (px at the working size) low-passes the field so it reads like a gridded measurement.
    These stand in for real measured channels in demos; the result is flagged ``standin``."""
    from .compose import render_effect
    from . import dotkit
    W, H = size
    sw = min(W, max_w); sh = max(8, int(round(sw * H / W)))
    names = list(which) if which else [STANDIN[(seed + 2 * i) % len(STANDIN)] for i in range(n)]
    mono = pal.make(40, harmony="mono")
    ch = {}
    for i, nm in enumerate(names[:n]):
        if nm.startswith("noise"):
            sc = float(nm.split(":")[1]) if ":" in nm else 60.0
            f = dotkit.smooth_noise((sh, sw), sc, np.random.default_rng(seed * 101 + i), 4, 0.5).astype(np.float32)
        else:
            img = render_effect(nm, size=(sw, sh), seed=seed * 7 + i, palette=mono)
            f = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        sm = smooth[i % len(smooth)] if isinstance(smooth, (list, tuple)) else smooth
        if sm > 0:
            f = ndimage.gaussian_filter(f, sm)
        lo, hi = np.percentile(f, [1, 99])
        ch[f"{nm.split('-', 1)[-1].replace(':', '')}"] = np.clip((f - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
    return ch, names[:n]


def norm_range(a: np.ndarray, rng_spec=None, pct=(1, 99)):
    """Normalise one channel with an explicit [lo, hi] (given) or percentile range (reported). -> (0..1, (lo, hi))."""
    a = np.asarray(a, np.float32)
    if rng_spec is not None:
        lo, hi = float(rng_spec[0]), float(rng_spec[1])
    else:
        fin = a[np.isfinite(a)]
        lo, hi = (float(np.percentile(fin, pct[0])), float(np.percentile(fin, pct[1]))) if fin.size else (0.0, 1.0)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32), (lo, hi)


def fit_field(a: np.ndarray, size) -> np.ndarray:
    """Resize a 2D field to the frame (cover, centred crop to the frame aspect, never stretch)."""
    W, H = size
    h, w = a.shape[:2]
    ta = W / H
    if w / h > ta:
        nw = int(round(h * ta)); x0 = (w - nw) // 2; a = a[:, x0:x0 + nw]
    else:
        nh = int(round(w / ta)); y0 = (h - nh) // 2; a = a[y0:y0 + nh]
    return cv2.resize(np.ascontiguousarray(a, np.float32), (W, H), interpolation=cv2.INTER_CUBIC if a.shape[1] < W else cv2.INTER_AREA)
