"""Region and material kit for the region_/layers_/collage_ effect families.

A photo or a real material is turned into a *designed* composition by
(1) partitioning the frame into a few simplified regions, (2) restyling each
region (lightness band, hue family, chroma, contrast) while (3) keeping the
physical texture through frequency bands (big shading / mid structure / fine
grain are retained or flattened separately).

Builds on ``features`` (superpixels, label_means, oklab, kmeans, fit_cover)
rather than duplicating it. Everything is deterministic numpy / scipy / cv2 on
float32 RGB (H, W, 3) 0..1.

Public pieces
    frame_source(ctx)        source fitted to the frame + the crop box used (so masks follow the crop)
    segment(img, ...)        RegionSet by colour | hue | luma | detail | superpixel
    regions_from_ctx(ctx, img, box, **kw)   user mask / polygon list from ctx.data, else segment()
    from_mask / from_polygons                hand-correctable masks
    RegionSet                labels (H,W), n, weights(soft) and helpers
    RegionStyle, assign_styles, restyle      per-region colour / contrast / texture-by-scale
    lowpass, guided, edge_smooth, band_split frequency tools
    boundary_lines, drop_shadow, paper_grain surface helpers

User masks (``ctx.data``): a mask image path / array (grey levels or flat
colours = one region each), a JSON path, or a dict ``{"mask": ..., "polygons":
[{"region": 1, "points": [[x, y], ...]}, ...], "n": 3}``. Mask and polygon
coordinates are in SOURCE image space (polygons 0..1 fractions or pixels);
they follow the same crop as the source, so hand corrections stay aligned.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import features as F
from . import palette as pal

# ------------------------------------------------------------------ common params

SOURCE_PARAMS = {
    "crop": ("feature", "SOURCE how a source of another aspect is cropped: feature | center"),
    "zoom": (0.0, "SOURCE 0 = auto; 0.3-0.9 forces a zoomed crop of the densest region"),
    "hue_from_source": (False, "SOURCE re-key the palette to the source's dominant hue"),
}
STRUCTURE_PARAMS = {
    "regions": (5, "STRUCTURE maximum number of regions (2-8); similar ones are merged (see merge); user masks override"),
    "merge": (0.4, "STRUCTURE merge regions whose colours are closer than this (0 = never; 0.4 merges a lit gradient, keeps water vs rock)"),
    "by": ("colour", "STRUCTURE how regions are found: colour | hue | luma | detail | superpixel"),
    "smooth": (0.5, "STRUCTURE boundary simplification 0..1 (higher = rounder, fewer wiggles)"),
    "min_area": (0.012, "STRUCTURE drop islands smaller than this share of the frame"),
    "snap": (0.6, "STRUCTURE 0..1 pull simplified boundaries back onto real image edges (guided filter)"),
}


def merge(*dicts) -> dict:
    out: dict = {}
    for d in dicts:
        out.update(d)
    return out


# ------------------------------------------------------------------ framing

def _boxes(img, size, crop, zoom):
    """Fractional crop box (x0, y0, x1, y1) matching what features.fit_cover would take."""
    W, H = size
    h, w = img.shape[:2]
    if abs(w / h - W / H) <= 0.01 and not (zoom and zoom < 0.99):
        return (0.0, 0.0, 1.0, 1.0)
    if crop == "center":
        cw = min(w, int(h * W / H)); ch = min(h, int(w * H / W))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return (x0 / w, y0 / h, (x0 + cw) / w, (y0 + ch) / h)
    small = cv2.resize(img, (min(w, 600), int(h * min(w, 600) / w)), interpolation=cv2.INTER_AREA) if w > 600 else img
    sh, sw = small.shape[:2]
    b = F.feature_crop(small, W / H, min_frac=zoom, max_frac=zoom, steps=1) if zoom else F.feature_crop(small, W / H, min_frac=0.6)
    return (b[0] / sw, b[1] / sh, b[2] / sw, b[3] / sh)


def frame_source(ctx, local: float = 0.0) -> SimpleNamespace:
    """Source fitted to the frame. Returns img, box (fractions of the source), src (original)."""
    W, H = ctx.size
    src = ctx.source if ctx.source is not None else F.synthetic_source((W, H), ctx.seed)
    src = np.asarray(src, np.float32)[..., :3]
    box = _boxes(src, (W, H), ctx.p("crop", "feature"), float(ctx.p("zoom", 0.0)))
    img = F.fit_cover(src, (W, H), box)
    if local > 0:
        lab = F.to_oklab(img)
        L = F.local_equalize(np.clip(lab[..., 0], 0, 1), 2.0, mix=local)
        lab[..., 0] = L
        img = F.from_oklab(lab)
    p = ctx.palette
    if ctx.p("hue_from_source", False):
        p = pal.make(F.dominant_hue(img, default=pal.rgb_to_oklch(p.structure)[2]), polarity=p.polarity, harmony="analogous",
                     three=p.structure2 is not None)
    return SimpleNamespace(img=img, box=box, src=src, pal=p, s=F.scale_of((H, W)), W=W, H=H, dark=p.polarity == "dark")


def _crop_to_frame(arr: np.ndarray, box, size, interp=cv2.INTER_NEAREST) -> np.ndarray:
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    a = np.ascontiguousarray(arr[int(round(y0 * h)):int(round(y1 * h)), int(round(x0 * w)):int(round(x1 * w))])
    return cv2.resize(a, tuple(size), interpolation=interp)


# ------------------------------------------------------------------ frequency tools

def lowpass(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass that stays fast for big sigma (works at reduced size)."""
    if sigma <= 0.3:
        return x
    f = max(1, int(sigma / 3.0))
    if f == 1:
        return ndi.gaussian_filter(x, (sigma, sigma) + (0,) * (x.ndim - 2), mode="reflect").astype(np.float32)
    h, w = x.shape[:2]
    sm = cv2.resize(x, (max(2, w // f), max(2, h // f)), interpolation=cv2.INTER_AREA)
    sm = ndi.gaussian_filter(sm, (sigma / f, sigma / f) + (0,) * (x.ndim - 2), mode="reflect")
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def band_split(L: np.ndarray, s_low: float, s_tex: float):
    """(low, mid, high): low = shading (sigma s_low), mid = structure between the scales, high = fine texture."""
    a = lowpass(L, s_low)
    b = lowpass(L, s_tex)
    return a, b - a, L - b


def guided(p: np.ndarray, guide: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Self-contained guided filter (He et al.) for a 2-D map p with a 2-D guide."""
    r = max(1, int(r))
    k = (2 * r + 1, 2 * r + 1)
    bx = lambda a: cv2.boxFilter(a, -1, k, borderType=cv2.BORDER_REFLECT)
    mI, mp = bx(guide), bx(p)
    cov = bx(guide * p) - mI * mp
    var = bx(guide * guide) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return bx(a) * guide + bx(b)


def edge_smooth(img: np.ndarray, strength: float, s: float = 1.0, iters: int = 2) -> np.ndarray:
    """Edge-aware smoothing (self-guided filter on luminance guide, per channel): flattens texture,
    keeps edges. strength 0..1 (0 = untouched)."""
    if strength <= 0:
        return img
    g = F.luminance(img)
    r = max(2, int((3 + 14 * strength) * s))
    eps = (0.004 + 0.05 * strength) ** 2 * 0.5
    out = img
    for _ in range(iters):
        out = np.stack([guided(out[..., c], g, r, eps) for c in range(3)], -1)
    return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ RegionSet

@dataclass
class RegionSet:
    labels: np.ndarray                      # (H, W) int32, region 0 = darkest (for segmented sets)
    n: int
    soft_px: float = 1.0
    _w: np.ndarray | None = field(default=None, repr=False)

    @property
    def weights(self) -> np.ndarray:
        """(n, H, W) float32 soft weights summing to 1 (hard labels blurred by soft_px for anti-aliasing / softness)."""
        if self._w is None:
            w = np.stack([(self.labels == i).astype(np.float32) for i in range(self.n)])
            if self.soft_px > 0.3:
                w = np.stack([ndi.gaussian_filter(x, self.soft_px) for x in w])
                w /= w.sum(0, keepdims=True) + 1e-9
            self._w = w
        return self._w

    def area(self) -> np.ndarray:
        return np.bincount(self.labels.ravel(), minlength=self.n).astype(np.float32) / self.labels.size

    def mask(self, i: int, soft: float | None = None) -> np.ndarray:
        m = (self.labels == i).astype(np.float32)
        s = self.soft_px if soft is None else soft
        return ndi.gaussian_filter(m, s) if s > 0.3 else m

    def edges(self, width: float = 1.5) -> np.ndarray:
        """Anti-aliased boundary lines between regions, 0..1."""
        e = F.label_edges(self.labels, 1)
        e = cv2.dilate(e, np.ones((3, 3), np.uint8)) if width > 1.5 else e
        sg = max(0.5, width * 0.45)
        e = ndi.gaussian_filter(e, sg)
        return np.clip(e * (1.6 + width * 0.3), 0, 1).astype(np.float32)


def _drop_small(lab: np.ndarray, n: int, min_area: float) -> np.ndarray:
    """Relabel connected components below min_area to the nearest bigger region."""
    thr = max(2, int(min_area * lab.size))
    out = lab.copy()
    bad = np.zeros(lab.shape, bool)
    for i in range(n):
        cc, k = ndi.label(lab == i)
        if k == 0:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        for j in np.where(sizes < thr)[0]:
            bad |= cc == (j + 1)
    if bad.any() and not bad.all():
        idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
        out = lab[idx[0], idx[1]]
    return out


def _order_by_luma(lab: np.ndarray, n: int, L: np.ndarray) -> np.ndarray:
    means = np.array([L[lab == i].mean() if (lab == i).any() else 9 for i in range(n)])
    rank = np.empty(n, int)
    rank[np.argsort(means)] = np.arange(n)
    return rank[lab].astype(np.int32)


def _compact(lab: np.ndarray) -> tuple[np.ndarray, int]:
    u = np.unique(lab)
    m = np.zeros(int(u.max()) + 1, np.int32)
    m[u] = np.arange(len(u))
    return m[lab].astype(np.int32), len(u)


def _refine(lab_small: np.ndarray, n: int, size, guide: np.ndarray, smooth: float, snap: float, s: float) -> np.ndarray:
    """Simplify boundaries (smooth one-hot at low res), upsample, optionally snap to the image
    edges with a guided filter, take argmax. Returns full-size int32 labels."""
    W, H = size
    h, w = lab_small.shape
    sg = smooth * 0.022 * max(h, w) + 0.01
    probs = np.stack([ndi.gaussian_filter((lab_small == i).astype(np.float32), sg) for i in range(n)])
    up = np.stack([cv2.resize(p, (W, H), interpolation=cv2.INTER_CUBIC) for p in probs])
    if snap > 0:
        g = lowpass(guide, 1.0 * s)
        r = int((4 + 8 * snap) * s)
        up = np.stack([guided(p, g, r, 0.0015 / (0.2 + snap)) for p in up])
    return np.argmax(up, 0).astype(np.int32)


def _merge_close(cent: np.ndarray, tol: float) -> np.ndarray:
    """Group centroids closer than tol (feature units); returns group id per centroid."""
    grp = np.arange(len(cent))
    c = cent.copy()
    cnt = np.ones(len(cent))
    while True:
        ids = np.unique(grp)
        if len(ids) <= 2:
            break
        best, pair = tol, None
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                d = np.linalg.norm(c[ids[a]] - c[ids[b]])
                if d < best:
                    best, pair = d, (ids[a], ids[b])
        if pair is None:
            break
        a, b = pair
        c[a] = (c[a] * cnt[a] + c[b] * cnt[b]) / (cnt[a] + cnt[b]); cnt[a] += cnt[b]
        grp[grp == b] = a
    return grp


def _open_labels(lab: np.ndarray, n: int, r: int) -> np.ndarray:
    """Remove thin structures (rings, hairlines) of every region by opening; freed pixels go to the nearest region."""
    if r < 1:
        return lab
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    keep = np.zeros(lab.shape, bool)
    for i in range(n):
        keep |= cv2.morphologyEx((lab == i).astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    if keep.all() or not keep.any():
        return lab
    idx = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
    return lab[idx[0], idx[1]]


def segment(img: np.ndarray, n: int = 4, by: str = "colour", seed: int = 0, smooth: float = 0.5, min_area: float = 0.012,
            snap: float = 0.6, soft_px: float | None = None, merge: float = 0.4) -> RegionSet:
    """Partition `img` into <= n simplified regions, ordered dark -> light (region 0 darkest)."""
    H, W = img.shape[:2]
    s = F.scale_of((H, W))
    n = int(max(2, min(10, n)))
    f = min(1.0, 300.0 / max(H, W))
    sm = cv2.resize(img, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA) if f < 1 else img
    h, w = sm.shape[:2]
    lab = F.to_oklab(F.blur(sm, 1.2))
    L = lab[..., 0]
    rng = np.random.default_rng(seed)
    if by == "luma":
        x = (L * 3.0).reshape(-1, 1)
    elif by == "hue":
        c = np.hypot(lab[..., 1], lab[..., 2])
        wgt = np.clip(c / 0.08, 0, 1)
        x = np.stack([lab[..., 1] / (c + 1e-4) * wgt, lab[..., 2] / (c + 1e-4) * wgt, L * 0.8], -1).reshape(-1, 3) * 3.0
    elif by == "detail":
        d = F.blur(F.detail_energy(L, 10), 3)
        x = (d / (d.max() + 1e-9) * 3.0).reshape(-1, 1)
    elif by == "superpixel":
        sp = F.superpixels(sm, n=260, compactness=14)
        m = F.label_means(lab, sp)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        feat = np.concatenate([m * np.array([3.0, 6.0, 6.0], np.float32), np.stack([F.label_means(yy / h, sp), F.label_means(xx / w, sp)], 1) * 0.6], 1)
        cent, grp = F._kmeans(feat, n, rng)
        mg = _merge_close(cent, merge)
        lab_small = mg[grp][sp].astype(np.int32)
        x = None
    else:   # colour
        x = np.concatenate([lab[..., :1] * 1.5, lab[..., 1:] * 7.0], -1).reshape(-1, 3)
    if x is not None:
        sub = x[rng.choice(len(x), min(len(x), 6000), replace=False)]
        cent, _ = F._kmeans(sub, n, rng, iters=12)
        mg = _merge_close(cent, merge)
        d = ((x[:, None, :] - cent[None]) ** 2).sum(-1)
        lab_small = mg[np.argmin(d, 1)].reshape(h, w).astype(np.int32)
    lab_small = _open_labels(lab_small, n, 1 + int(smooth * 1.5))
    lab_small = _drop_small(lab_small, n, min_area)
    lab_small, n2 = _compact(lab_small)
    full = _refine(lab_small, n2, (W, H), F.luminance(img), smooth, snap, s)
    full = _drop_small(full, n2, min_area * 0.5)
    full, n3 = _compact(full)
    full = _order_by_luma(full, n3, F.luminance(img))
    return RegionSet(full, n3, soft_px if soft_px is not None else 0.9 * s)


# ------------------------------------------------------------------ hand-correctable masks

def _read_any(v) -> Any:
    if isinstance(v, (str, os.PathLike)):
        p = str(v)
        if p.lower().endswith(".json"):
            with open(p) as fh:
                return json.load(fh)
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(p)
        if im.ndim == 3 and im.shape[2] == 4:
            im = np.dstack([im[..., :3], im[..., 3]])
        return im[..., ::-1].copy() if im.ndim == 3 else im
    return v


def from_mask(mask, size, box=(0, 0, 1, 1), n: int | None = None, soft_px: float = 1.0, luma: np.ndarray | None = None) -> RegionSet:
    """Mask image -> RegionSet. A grey / colour mask with <= 8 distinct values gives one region per value
    (ordered by value); otherwise grey levels are quantised into `n` equal bands. A bool/0-1 mask is 2 regions."""
    m = np.asarray(_read_any(mask))
    if m.ndim == 3:
        m = m[..., :3]
        flat = m.reshape(-1, m.shape[-1])
        if m.dtype != np.uint8:
            flat = (np.clip(flat, 0, 1) * 255).astype(np.uint8) if m.max() <= 1.0 else flat.astype(np.uint8)
        keys = flat[:, 0].astype(np.int32) * 65536 + flat[:, 1].astype(np.int32) * 256 + flat[:, 2]
        ordv = keys.reshape(m.shape[:2])
        gray = (flat @ np.array([0.2126, 0.7152, 0.0722])).reshape(m.shape[:2])
    else:
        ordv = gray = m.astype(np.float32)
        if m.dtype == bool or (m.max() <= 1.0 and len(np.unique(m)) <= 2):
            ordv = gray = (m > 0.5).astype(np.float32)
    u = np.unique(ordv)
    if len(u) <= 8:
        # order regions by their grey value
        means = np.array([gray[ordv == v].mean() for v in u])
        rank = {v: i for i, v in enumerate(u[np.argsort(means)])}
        idx = np.zeros(ordv.shape, np.int32)
        for v in u:
            idx[ordv == v] = rank[v]
        k = len(u)
    else:
        k = int(n or 3)
        g = (gray - gray.min()) / (np.ptp(gray) + 1e-9)
        idx = np.minimum((g * k).astype(np.int32), k - 1)
    lab = _crop_to_frame(idx.astype(np.uint8), box, size, cv2.INTER_NEAREST).astype(np.int32)
    return RegionSet(lab, k, soft_px)


def from_polygons(spec, size, box=(0, 0, 1, 1), src_size=None, soft_px: float = 1.0) -> RegionSet:
    """Polygon list -> RegionSet. spec = [{"region": 1, "points": [[x, y], ...]}, ...] or a list of point
    lists (regions 1, 2, ...), later polygons paint over earlier ones; everything else is region 0.
    Points are 0..1 fractions of the SOURCE (or pixels of the source if src_size is given)."""
    W, H = size
    polys = spec.get("polygons", spec) if isinstance(spec, dict) else spec
    items = []
    for i, p in enumerate(polys):
        if isinstance(p, dict):
            items.append((int(p.get("region", i + 1)), np.asarray(p["points"], np.float32)))
        else:
            items.append((i + 1, np.asarray(p, np.float32)))
    x0, y0, x1, y1 = box
    lab = np.zeros((H, W), np.uint8)
    for reg, pts in items:
        q = pts.copy()
        if src_size is not None and q.max() > 1.5:
            q[:, 0] /= src_size[0]; q[:, 1] /= src_size[1]
        q[:, 0] = (q[:, 0] - x0) / max(x1 - x0, 1e-6) * W
        q[:, 1] = (q[:, 1] - y0) / max(y1 - y0, 1e-6) * H
        cv2.fillPoly(lab, [np.round(q).astype(np.int32)], int(reg))
    k = int(lab.max()) + 1
    lab, k = _compact(lab.astype(np.int32))
    return RegionSet(lab, k, soft_px)


def regions_from_ctx(ctx, fr, **kw) -> RegionSet:
    """User mask / polygons from ctx.data when present, else segment(fr.img). kw -> segment()."""
    s = fr.s
    soft = kw.pop("soft_px", None)
    data = ctx.data
    if isinstance(data, (str, os.PathLike)) and not str(data).lower().endswith(".json"):
        data = {"mask": data}
    elif isinstance(data, (str, os.PathLike)):
        data = _read_any(data)
    elif isinstance(data, np.ndarray):
        data = {"mask": data}
    if isinstance(data, list):
        data = {"polygons": data}
    rs = None
    if isinstance(data, dict):
        if data.get("mask") is not None:
            rs = from_mask(data["mask"], (fr.W, fr.H), fr.box, data.get("n"), 0.9 * s)
        elif data.get("polygons"):
            rs = from_polygons(data, (fr.W, fr.H), fr.box, fr.src.shape[1::-1], 0.9 * s)
        if rs is not None and kw.get("smooth", 0) > 0:
            # polish hand masks lightly: edges snap to the picture (set snap=0 to keep them exactly)
            rs = _polish(rs, fr, kw["smooth"] * 0.4, kw.get("snap", 0.0) * 0.8)
    if rs is None:
        rs = segment(fr.img, seed=ctx.seed, soft_px=soft, **kw)
    return rs


def _polish(rs: RegionSet, fr, smooth: float, snap: float) -> RegionSet:
    if smooth <= 0.01 and snap <= 0.01:
        return rs
    s = fr.s
    probs = np.stack([ndi.gaussian_filter((rs.labels == i).astype(np.float32), 2 + smooth * 14 * s) for i in range(rs.n)])
    if snap > 0:
        g = lowpass(F.luminance(fr.img), 1.0 * s)
        probs = np.stack([guided(p, g, int((3 + 6 * snap) * s), 0.002) for p in probs])
    return RegionSet(np.argmax(probs, 0).astype(np.int32), rs.n, rs.soft_px)


# ------------------------------------------------------------------ per-region treatment

@dataclass
class RegionStyle:
    hue: float | None = None        # target hue (deg); None = keep the source hue
    hue_keep: float = 0.35          # share of the source's own hue variation kept around the target
    chroma: float = 1.0             # multiplier on the source chroma
    c_abs: float = 0.0              # absolute chroma floor/target blended in when hue is set
    L_mean: float | None = None     # move the region's mean lightness here (None keep)
    L_range: float = 1.0            # scale of the region's lightness spread (contrast around its mean)
    flat: float = 0.0               # 0..1 flatten the low-frequency shading toward the region mean
    low: float = 1.0                # gain on shading (low band)
    mid: float = 1.0                # gain on structure (mid band)
    high: float = 1.0               # gain on fine texture (high band)
    L_mul: float = 1.0              # final lightness multiplier (atmospheric dimming)
    steps: int = 0                  # >0 posterises the shading into this many tone steps (print look); texture stays


def _lbounds(dark: bool):
    return (0.17, 0.86) if dark else (0.30, 0.95)


def assign_styles(rs: RegionSet, img: np.ndarray, p, *, colour: str = "families", hues: str = "", spread: float = 0.5,
                  chroma: float = 1.0, contrast: float = 1.0, hue_keep: float = 0.35, base_hue: float | None = None,
                  c_abs: float = 0.06) -> list[RegionStyle]:
    """Colour arrangement for the regions (regions are ordered dark -> light).

    colour: source   keep each region's own hue (only lightness / chroma shaping)
            palette  hues from palette roles (structure, accent, structure2) in lightness order
            families distinct hue families spread from the palette hue (analogous + complement accents)
            hues     explicit list from `hues` ("190,30,260", cycled)
    spread: 0 keep each region's mean lightness, 1 spread region means evenly across the allowed range.
    """
    lo, hi = _lbounds(p.polarity == "dark")
    L = F.luminance(img)
    n = rs.n
    means = np.array([F.to_oklab(img[rs.labels == i].reshape(-1, 1, 3))[..., 0].mean() if (rs.labels == i).any() else 0.5 for i in range(n)])
    h0 = pal.rgb_to_oklch(p.structure)[2] if base_hue is None else base_hue
    ha = pal.rgb_to_oklch(p.accent)[2]
    if colour == "palette":
        roles = [h0, ha, pal.rgb_to_oklch(p.structure2)[2] if p.structure2 is not None else (h0 + 40) % 360]
        hs = [roles[i % 3] for i in range(n)]
    elif colour == "hues" and hues:
        lst = [float(x) for x in str(hues).split(",") if x.strip()]
        hs = [lst[i % len(lst)] for i in range(n)]
    elif colour == "families":
        # a deliberate set of hue families around the palette hue, handed to the regions by minimal hue
        # distance from each region's own hue (water stays in the blue-ish family, rock in the warm one)
        sets = {2: [0, 180], 3: [0, 175, 35]}
        offs = sets.get(n, [0, 175, 35, -40, 140, 70, 215, -75, 105, 250][:n])
        fam = [(h0 + o) % 360 for o in offs]
        lab = F.to_oklab(img[::4, ::4])
        lb = rs.labels[::4, ::4]
        reg_h, reg_c = [], []
        for i in range(n):
            m = lb == i
            if m.sum() < 4:
                reg_h.append(h0); reg_c.append(0.0); continue
            a_, b_ = lab[..., 1][m], lab[..., 2][m]
            reg_h.append(math.degrees(math.atan2(b_.mean(), a_.mean())) % 360); reg_c.append(float(math.hypot(a_.mean(), b_.mean())))
        rh = np.array(reg_h); rc = np.array(reg_c)
        # grey regions (no hue of their own) fall back on lightness order against families sorted by lightness-friendly warm-ness
        d = np.abs((rh[:, None] - np.array(fam)[None] + 180) % 360 - 180) / 180.0
        w_ = np.clip(rc / 0.05, 0, 1)[:, None]
        cost = d * w_ + (1 - w_) * (np.arange(n)[:, None] != np.argsort(np.argsort(means))[:, None]) * 0.0
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
        hs = [None] * n
        for r_, c_ in zip(ri, ci):
            hs[r_] = fam[c_]
    else:
        hs = [None] * n
    out = []
    order = np.argsort(means)
    for rank, i in enumerate(order):
        target = lo + (hi - lo) * (0.08 + 0.84 * rank / max(1, n - 1))
        Lm = means[i] * (1 - spread) + target * spread
        Lm = float(np.clip(Lm, lo + 0.02, hi - 0.02))
        out.append((i, RegionStyle(hue=hs[i], hue_keep=hue_keep, chroma=chroma, c_abs=c_abs if hs[i] is not None else 0.0,
                                   L_mean=Lm, L_range=contrast)))
    out.sort(key=lambda t: t[0])
    return [s for _, s in out]


def restyle(img: np.ndarray, rs: RegionSet, styles: list[RegionStyle], *, s_low: float, s_tex: float, dark: bool = True,
            chroma_cap: float = 0.2) -> np.ndarray:
    """Restyle every region with its own RegionStyle, retaining texture by frequency band.

    Lightness is split inside each region (normalised convolution, so bands do not bleed across
    boundaries) into shading / structure / fine texture; `flat`, `low`, `mid`, `high` rebalance them.
    Hue is replaced / blended per style, chroma bounded; one gamut-safe OKLCH conversion at the end."""
    lab = F.to_oklab(img)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    W = rs.weights
    lo, hi = _lbounds(dark)
    Lo = np.zeros_like(L); Ao = np.zeros_like(L); Bo = np.zeros_like(L)
    for i, st in enumerate(styles[:rs.n]):
        w = W[i]
        if w.sum() < 1:
            continue
        mean = float((L * w).sum() / w.sum())
        neutral = (st.flat == 0 and st.low == 1 and st.mid == 1 and st.high == 1 and st.steps == 0 and st.L_mean is None and st.L_range == 1)
        if neutral:
            Ln = L
        else:
            den = lowpass(w, s_low) + 1e-4
            den_t = lowpass(w, s_tex) + 1e-4
            base = lowpass(L * w, s_low) / den
            base_t = lowpass(L * w, s_tex) / den_t
            low = base - mean
            mid = base_t - base
            high = L - base_t
            Lm = mean if st.L_mean is None else st.L_mean
            shade = st.low * (1 - st.flat) * low + st.mid * mid
            if st.steps > 0:
                q = 0.16 / st.steps
                x = shade / q
                fl = np.floor(x)
                shade = (fl + F.smoothstep(0.35, 0.65, x - fl)) * q
            Ln = Lm + st.L_range * (shade + st.high * high)
        if st.L_mul != 1.0:
            Ln = Ln * st.L_mul
        # source chroma / hue
        C = np.hypot(a, b)
        h = np.arctan2(b, a)
        if st.hue is None:
            Cn = C * st.chroma
            hn = np.degrees(h)
        else:
            ht = np.radians(st.hue)
            # keep a share of the source hue variation around the target hue, wrapped to +-60 deg
            dh = np.arctan2(np.sin(h - ht), np.cos(h - ht))
            dh = np.clip(dh, -1.0, 1.0)
            hn = np.degrees(ht + dh * st.hue_keep)
            Cn = (C * (0.5 * st.chroma) + st.c_abs) * st.chroma
        # lightness-shaped chroma: calm darks and lights, richest in the mids
        shape = F.smoothstep(0.12, 0.45, Ln) * (1 - 0.45 * F.smoothstep(0.8, 0.97, Ln))
        Cn = np.minimum(Cn * (0.35 + 0.65 * shape), chroma_cap)
        hr = np.radians(hn)
        Lo += w * Ln
        Ao += w * (Cn * np.cos(hr))
        Bo += w * (Cn * np.sin(hr))
    Lo = np.clip(Lo, lo - 0.1, hi + 0.08)
    Cf = np.hypot(Ao, Bo)
    hf = np.degrees(np.arctan2(Bo, Ao))
    return F.oklch_field(Lo, Cf, hf)


# ------------------------------------------------------------------ surface helpers

def boundary_lines(rs: RegionSet, width_px: float, color, base: np.ndarray, strength: float = 1.0) -> np.ndarray:
    if width_px <= 0 or strength <= 0:
        return base
    e = rs.edges(width_px)[..., None] * strength
    return base * (1 - e) + np.asarray(color, np.float32) * e


def drop_shadow(alpha: np.ndarray, offset: tuple[float, float], blur: float) -> np.ndarray:
    """Soft shadow mask (0..1) of a (H, W) alpha, shifted by offset (dx, dy) px."""
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    sh = cv2.warpAffine(alpha.astype(np.float32), M, (alpha.shape[1], alpha.shape[0]), borderMode=cv2.BORDER_CONSTANT)
    return ndi.gaussian_filter(sh, blur) if blur > 0.3 else sh


def paper_grain(shape, seed: int, s: float, strength: float = 1.0) -> np.ndarray:
    """Zero-mean paper fibre / tooth texture field (H, W) for multiplying into colours (about +-0.03 * strength)."""
    H, W = shape[:2]
    rng = np.random.default_rng(seed)
    a = ndi.gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), 0.8 * s)
    b = ndi.gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), (0.6 * s, 5.0 * s))
    c = ndi.gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), (5.0 * s, 0.6 * s))
    f = a / (a.std() + 1e-9) * 0.6 + b / (b.std() + 1e-9) * 0.4 + c / (c.std() + 1e-9) * 0.4
    return (f / (f.std() + 1e-9) * 0.03 * strength).astype(np.float32)


def finish(out: np.ndarray, seed: int = 0, grain: float = 0.0, s: float = 1.0) -> np.ndarray:
    if grain > 0:
        g = ndi.gaussian_filter(np.random.default_rng(seed + 77).normal(0, 1, out.shape[:2]).astype(np.float32), 0.6 * max(s, 0.5))
        out = out + (g / (g.std() + 1e-9) * grain * 0.035)[..., None]
    return np.clip(out, 0.02, 0.985).astype(np.float32)
