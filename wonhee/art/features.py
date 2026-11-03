"""Feature extractors for source-driven art effects (owner direction 2026-11-03:
a project image is an edit of a real FEATURE of the original).

Everything here is deterministic numpy / scipy / cv2 on float32 RGB (H, W, 3)
in 0..1 (the ``ctx.source`` convention; NOT BGR uint8). Extractors return
fields at the resolution of the input unless noted. No AI.

Extractors
    luminance, to_oklab / from_oklab       colour helpers
    local_equalize                         CLAHE-style tone, spreads detail evenly
    detail_energy, saliency                where the information is
    edges_multiscale                       multi-scale gradient edges, 0..1
    structure_orientation                  orientation + coherence field (structure tensor)
    ridge_valley                           Hessian ridge / valley strength
    dominant_palette, dominant_hue         OKLCH k-means palette of the source
    superpixels                            SLIC-like labels
    detect_grid                            repeated-pattern period (x, y)
    text_mask                              text-like (small high-contrast stroke) regions
    feature_crop, detail_crops, tile_grid  most informative window(s) at an aspect
    fit_cover                              resize to a frame, cropping by detail not centre
    region_weights, composite_regions      per-region effects with soft or hard seams
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
from scipy import ndimage as ndi

_LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ---------------------------------------------------------------- colour

def luminance(img: np.ndarray) -> np.ndarray:
    return (img @ _LUM).astype(np.float32)


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_oklab(img: np.ndarray) -> np.ndarray:
    """sRGB 0..1 (H,W,3) -> OKLab (H,W,3) = (L, a, b)."""
    c = _lin(np.clip(np.asarray(img, np.float32), 0, 1))
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1).astype(np.float32)


def from_oklab(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1)
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)


def oklch_field(L, C, h_deg) -> np.ndarray:
    """OKLCH arrays -> sRGB, reducing chroma per pixel until in gamut."""
    L, C, h = np.broadcast_arrays(np.asarray(L, np.float32), np.asarray(C, np.float32), np.asarray(h_deg, np.float32))
    a0, b0 = np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h))
    k = np.ones_like(C)
    out = None
    for _ in range(6):
        lab = np.stack([L, C * k * a0, C * k * b0], -1)
        l = (lab[..., 0] + 0.3963377774 * lab[..., 1] + 0.2158037573 * lab[..., 2]) ** 3
        m = (lab[..., 0] - 0.1055613458 * lab[..., 1] - 0.0638541728 * lab[..., 2]) ** 3
        s = (lab[..., 0] - 0.0894841775 * lab[..., 1] - 1.2914855480 * lab[..., 2]) ** 3
        lin = np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1)
        bad = (lin.min(-1) < -1e-3) | (lin.max(-1) > 1 + 1e-3)
        out = lin
        if not bad.any():
            break
        k = np.where(bad, k * 0.8, k)
    lin = np.clip(out, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def stretch(g: np.ndarray, lo_p: float = 2, hi_p: float = 98) -> np.ndarray:
    lo, hi = np.percentile(g, (lo_p, hi_p))
    return np.clip((g - lo) / max(hi - lo, 1e-4), 0, 1).astype(np.float32)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    if a.ndim == 3:
        return ndi.gaussian_filter(a, (sigma, sigma, 0), mode="reflect")
    return ndi.gaussian_filter(a, sigma, mode="reflect")


def scale_of(shape) -> float:
    """Detail scale relative to a 1500 px card master."""
    return max(shape[0], shape[1]) / 1500.0


# ---------------------------------------------------------------- tone / detail

def local_equalize(L: np.ndarray, clip: float = 2.5, tiles: int = 6, mix: float = 1.0) -> np.ndarray:
    """CLAHE tone: every region uses the full tonal range, so no area is left
    flat (this is the main lever for 'no focal point'). Returns 0..1."""
    u8 = (np.clip(L, 0, 1) * 255 + 0.5).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(u8).astype(np.float32) / 255
    cl = blur(cl, 1.0)
    return (L * (1 - mix) + cl * mix).astype(np.float32)


def edges_multiscale(L: np.ndarray, scales: Sequence[float] = (1.0, 2.0, 4.0)) -> np.ndarray:
    """Gradient-magnitude edges over several scales, each normalised, combined
    by max (fine detail and big structure both survive). 0..1."""
    s = scale_of(L.shape)
    acc = np.zeros_like(L)
    for sg in scales:
        g = blur(L, sg * s)
        gx, gy = np.gradient(g)
        m = np.hypot(gx, gy)
        acc = np.maximum(acc, m / (np.percentile(m, 99) + 1e-6))
    return np.clip(acc, 0, 1).astype(np.float32)


def detail_energy(L: np.ndarray, sigma: float = 24.0) -> np.ndarray:
    """Smoothed edge energy (where the information is). 0..1, per pixel."""
    e = edges_multiscale(L)
    d = blur(e, sigma * scale_of(L.shape))
    return (d / (d.max() + 1e-9)).astype(np.float32)


def saliency(img: np.ndarray) -> np.ndarray:
    """Frequency-tuned saliency: OKLab distance from the global mean colour,
    after a light blur. 0..1. (Where it is *different*, not where it is busy.)"""
    lab = to_oklab(blur(img, 1.5 * scale_of(img.shape)))
    d = np.linalg.norm(lab - lab.reshape(-1, 3).mean(0), axis=-1)
    return (d / (d.max() + 1e-9)).astype(np.float32)


def structure_orientation(L: np.ndarray, sigma: float = 1.5, rho: float = 10.0, floor: float = 0.04, seed: int = 0):
    """Structure tensor orientation. Returns (theta, coherence): theta is the
    direction ALONG the local structure (perpendicular to the gradient), in
    radians; coherence 0..1 is how one-directional it is. In flat regions the
    tensor is filled with a smooth seeded direction field (strokes stay
    organic instead of random). """
    s = scale_of(L.shape)
    g = blur(L, sigma * s)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = gx * gx, gy * gy, gx * gy
    scl = float(np.percentile(jxx + jyy, 90)) + 1e-9
    rng = np.random.default_rng(seed)
    phi = blur(rng.normal(0, 1, L.shape).astype(np.float32), 60 * s) * 14.0   # smooth angle field
    phi = phi * 2.0
    fl = floor * scl
    # a flat-region tensor whose gradient direction is phi (so structure runs along phi+90)
    jxx = jxx + fl * np.cos(phi) ** 2
    jyy = jyy + fl * np.sin(phi) ** 2
    jxy = jxy + fl * np.cos(phi) * np.sin(phi)
    r = rho * s
    jxx, jyy, jxy = blur(jxx, r), blur(jyy, r), blur(jxy, r)
    ang_grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    tr = jxx + jyy + 1e-12
    root = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = root / tr
    return (ang_grad + np.pi / 2).astype(np.float32), np.clip(coh, 0, 1).astype(np.float32)


def ridge_valley(L: np.ndarray, sigmas: Sequence[float] = (1.5, 3.0)) -> tuple[np.ndarray, np.ndarray]:
    """Hessian ridge (bright line) and valley (dark line) strength, 0..1 each."""
    s = scale_of(L.shape)
    rid = np.zeros_like(L)
    val = np.zeros_like(L)
    for sg in sigmas:
        g = blur(L, sg * s)
        gyy, gyx = np.gradient(np.gradient(g, axis=0), axis=(0, 1))
        gxx = np.gradient(np.gradient(g, axis=1), axis=1)
        tr, det = gxx + gyy, gxx * gyy - gyx * gyx
        disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc          # l2 most negative -> ridge
        k = (sg * s) ** 2
        rid = np.maximum(rid, np.maximum(-l2, 0) * k)
        val = np.maximum(val, np.maximum(l1, 0) * k)
    rid /= np.percentile(rid, 99.5) + 1e-9
    val /= np.percentile(val, 99.5) + 1e-9
    return np.clip(rid, 0, 1).astype(np.float32), np.clip(val, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- palette

def _kmeans(x: np.ndarray, k: int, rng, iters: int = 14):
    # k-means++ init
    cent = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d = np.min([((x - c) ** 2).sum(1) for c in cent], axis=0)
        p = d / (d.sum() + 1e-12)
        cent.append(x[rng.choice(len(x), p=p)])
    cent = np.array(cent)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                cent[j] = x[m].mean(0)
    lab = np.argmin(((x[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    return cent, lab


def dominant_palette(img: np.ndarray, k: int = 5, seed: int = 0, n_samples: int = 5000) -> list[dict]:
    """OKLab k-means. Returns [{rgb, L, C, h, weight}] sorted by weight desc."""
    rng = np.random.default_rng(seed)
    flat = to_oklab(img).reshape(-1, 3)
    pts = flat[rng.choice(len(flat), min(n_samples, len(flat)), replace=False)]
    cent, lab = _kmeans(pts, k, rng)
    out = []
    for j in range(k):
        w = float((lab == j).mean())
        L, a, b = cent[j]
        out.append({"rgb": from_oklab(cent[j][None])[0], "L": float(L), "C": float(math.hypot(a, b)),
                    "h": float(math.degrees(math.atan2(b, a)) % 360), "weight": w})
    return sorted(out, key=lambda d: -d["weight"])


def dominant_hue(img: np.ndarray, min_chroma: float = 0.04, default: float = 40.0) -> float:
    """OKLCH hue (deg) that carries the most chroma-weighted area; `default` if the image is grey."""
    small = img[:: max(1, img.shape[0] // 160), :: max(1, img.shape[1] // 240)]
    lab = to_oklab(small).reshape(-1, 3)
    c = np.hypot(lab[:, 1], lab[:, 2])
    m = c > min_chroma
    if m.sum() < 0.01 * len(c):
        return default
    h = np.degrees(np.arctan2(lab[m, 2], lab[m, 1])) % 360
    hist, _ = np.histogram(h, bins=36, range=(0, 360), weights=c[m])
    hist = ndi.gaussian_filter1d(hist, 1.0, mode="wrap")
    return float((np.argmax(hist) + 0.5) * 10)


# ---------------------------------------------------------------- regions

def superpixels(img: np.ndarray, n: int = 400, compactness: float = 12.0, iters: int = 6, max_side: int = 520) -> np.ndarray:
    """SLIC-like superpixels (OKLab + xy), computed at reduced size and
    returned as an int32 label map at full size."""
    H, W = img.shape[:2]
    f = min(1.0, max_side / max(H, W))
    sm = cv2.resize(img, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA) if f < 1 else img
    h, w = sm.shape[:2]
    lab = to_oklab(blur(sm, 0.7)) * np.array([1.0, 1.0, 1.0], np.float32) * 100.0
    S = math.sqrt(h * w / n)
    gy = np.arange(S / 2, h, S); gx = np.arange(S / 2, w, S)
    cy, cx = np.meshgrid(gy, gx, indexing="ij")
    cy, cx = cy.ravel(), cx.ravel()
    cc = np.stack([lab[np.clip(cy.astype(int), 0, h - 1), np.clip(cx.astype(int), 0, w - 1)][:, i] for i in range(3)], 1)
    labels = -np.ones((h, w), np.int32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    m = compactness
    for _ in range(iters):
        dist = np.full((h, w), np.inf, np.float32)
        r = int(S * 1.6)
        for i in range(len(cy)):
            y0, y1 = max(0, int(cy[i]) - r), min(h, int(cy[i]) + r + 1)
            x0, x1 = max(0, int(cx[i]) - r), min(w, int(cx[i]) + r + 1)
            dl = ((lab[y0:y1, x0:x1] - cc[i]) ** 2).sum(-1)
            ds = (yy[y0:y1, x0:x1] - cy[i]) ** 2 + (xx[y0:y1, x0:x1] - cx[i]) ** 2
            d = dl + (m / S) ** 2 * ds
            sub = dist[y0:y1, x0:x1]
            better = d < sub
            sub[better] = d[better]
            labels[y0:y1, x0:x1][better] = i
        for i in range(len(cy)):
            msk = labels == i
            if msk.any():
                cy[i], cx[i] = yy[msk].mean(), xx[msk].mean()
                cc[i] = lab[msk].mean(0)
    labels[labels < 0] = 0
    if f < 1:
        labels = cv2.resize(labels.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return labels


def label_means(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean colour of every label -> (n_labels, 3)."""
    n = int(labels.max()) + 1
    flat = labels.ravel()
    cnt = np.bincount(flat, minlength=n).astype(np.float32) + 1e-9
    return np.stack([np.bincount(flat, weights=img[..., c].ravel(), minlength=n) / cnt for c in range(img.shape[2])], 1).astype(np.float32) \
        if img.ndim == 3 else (np.bincount(flat, weights=img.ravel(), minlength=n) / cnt).astype(np.float32)


def label_edges(labels: np.ndarray, width: int = 1) -> np.ndarray:
    """Boundary mask between labels (float 0..1)."""
    e = np.zeros(labels.shape, bool)
    e[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    e[1:, :] |= labels[1:, :] != labels[:-1, :]
    e = e.astype(np.float32)
    if width > 1:
        e = cv2.dilate(e, np.ones((width, width), np.uint8))
    return e


def detect_grid(L: np.ndarray, max_period: int | None = None) -> dict:
    """Repeated-pattern / grid detection from the autocorrelation of the edge
    profile along each axis. Returns {period_x, period_y, strength_x,
    strength_y}; period 0 means none. Useful for dashboards, tables, heatmaps,
    keyboards: align effects to the real grid."""
    H, W = L.shape
    e = edges_multiscale(L, (1.0, 2.0))
    out = {}
    for name, prof, n in (("x", e.mean(0), W), ("y", e.mean(1), H)):
        p = prof - ndi.uniform_filter1d(prof, max(5, n // 8))
        p = p - p.mean()
        ac = np.correlate(p, p, "full")[len(p) - 1:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = 6, (max_period or n // 3)
        seg = ac[lo:hi]
        if len(seg) < 4:
            out[f"period_{name}"], out[f"strength_{name}"] = 0, 0.0
            continue
        k = int(np.argmax(seg))
        s = float(seg[k])
        out[f"period_{name}"] = int(k + lo) if s > 0.25 else 0
        out[f"strength_{name}"] = s
    return out


def text_mask(img: np.ndarray) -> np.ndarray:
    """Text-like regions: many small, high-contrast strokes in short horizontal
    runs. Returns float 0..1 (blocks around text lines). Heuristic; good on
    screenshots, slides and posters."""
    H, W = img.shape[:2]
    s = scale_of(img.shape)
    L = (np.clip(luminance(img), 0, 1) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(L, cv2.MORPH_GRADIENT, k)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kw = max(5, int(14 * s)) | 1
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(3, int(4 * s)))))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    out = np.zeros((H, W), np.float32)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 4 or h > max(40, 0.08 * H) or w < 2.2 * h:
            continue
        dens = bw[y:y + h, x:x + w].mean() / 255
        if 0.12 < dens < 0.8:
            out[y:y + h, x:x + w] = 1.0
    return out


# ---------------------------------------------------------------- crops

def _integral(a):
    return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _box_mean(ii, y0, x0, y1, x1):
    return (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / ((y1 - y0) * (x1 - x0))


def feature_crop(img: np.ndarray, aspect: float = 1.5, min_frac: float = 0.45, max_frac: float = 1.0,
                 steps: int = 6, stride_frac: float = 0.05, evenness: float = 0.5) -> tuple[int, int, int, int]:
    """Most informative window of the given aspect (w/h), chosen by detail
    energy, NOT by centre. Scores every window by its mean edge energy,
    multiplied by an evenness term (share of its 4x4 cells that carry detail)
    so a lone hot-spot does not win. Returns (x0, y0, x1, y1)."""
    H, W = img.shape[:2]
    e = edges_multiscale(luminance(img))
    ii = _integral(e)
    cell = (e > 0.25 * (e.mean() + 1e-9)).astype(np.float32)
    ic = _integral(cell)
    best, box = -1.0, (0, 0, W, H)
    full_w = min(W, H * aspect)
    for f in np.linspace(max_frac, min_frac, steps):
        cw = int(full_w * f); ch = int(cw / aspect)
        if cw < 8 or ch < 8 or cw > W or ch > H:
            continue
        sx = max(1, int(W * stride_frac)); sy = max(1, int(H * stride_frac))
        for y0 in list(range(0, H - ch + 1, sy)) + [H - ch]:
            for x0 in list(range(0, W - cw + 1, sx)) + [W - cw]:
                m = _box_mean(ii, y0, x0, y0 + ch, x0 + cw)
                # evenness: min over 2x2 quadrants of cell-coverage
                qs = [_box_mean(ic, y0 + a * ch // 2, x0 + b * cw // 2, y0 + (a + 1) * ch // 2, x0 + (b + 1) * cw // 2) for a in (0, 1) for b in (0, 1)]
                score = m * ((1 - evenness) + evenness * min(qs) / (max(qs) + 1e-9)) * (f ** 0.15)
                if score > best:
                    best, box = score, (x0, y0, x0 + cw, y0 + ch)
    return box


def fit_cover(img: np.ndarray, size: tuple[int, int], crop: str | tuple | None = "feature", zoom: float = 0.0) -> np.ndarray:
    """Resize `img` to size=(W, H). If the aspect differs, crop first: 'feature'
    (most informative window), 'center', or an explicit (x0,y0,x1,y1) fraction box.
    zoom > 0 forces a 'feature' crop of that fraction of the largest fitting
    window even when the aspect already matches (zoom into the densest part)."""
    W, H = size
    h, w = img.shape[:2]
    if isinstance(crop, tuple):
        x0, y0, x1, y1 = crop
        img = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    elif abs(w / h - W / H) > 0.01 or (zoom and zoom < 0.99):
        if crop == "center":
            cw = min(w, int(h * W / H)); ch = min(h, int(w * H / W))
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            img = img[y0:y0 + ch, x0:x0 + cw]
        else:
            small = cv2.resize(img, (min(w, 600), int(h * min(w, 600) / w)), interpolation=cv2.INTER_AREA) if w > 600 else img
            f = w / small.shape[1]
            bx0, by0, bx1, by1 = feature_crop(small, W / H, min_frac=zoom, max_frac=zoom, steps=1) if zoom else feature_crop(small, W / H, min_frac=0.6)
            img = img[int(by0 * f):int(by1 * f), int(bx0 * f):int(bx1 * f)]
    interp = cv2.INTER_AREA if img.shape[1] > W else cv2.INTER_CUBIC
    return np.clip(cv2.resize(np.ascontiguousarray(img), (W, H), interpolation=interp), 0, 1).astype(np.float32)


def detail_crops(img: np.ndarray, n: int, aspect: float = 1.5, frac: float = 0.3, seed: int = 0) -> list[tuple[int, int, int, int]]:
    """n non-overlapping windows (greedy by mean detail energy) of width
    `frac` of the source, giving a spread of the most informative parts."""
    H, W = img.shape[:2]
    e = edges_multiscale(luminance(img))
    cw = int(min(W, H * aspect) * frac); ch = int(cw / aspect)
    cw, ch = max(cw, 8), max(ch, 8)
    ii = _integral(e)
    ys = list(range(0, H - ch + 1, max(1, ch // 4))); xs = list(range(0, W - cw + 1, max(1, cw // 4)))
    cand = sorted(((_box_mean(ii, y, x, y + ch, x + cw), x, y) for y in ys for x in xs), reverse=True)
    taken: list[tuple[int, int, int, int]] = []
    for sc, x, y in cand:
        if all(abs(x - a) >= cw * 0.8 or abs(y - b) >= ch * 0.8 for a, b, _, _ in taken):
            taken.append((x, y, x + cw, y + ch))
        if len(taken) == n:
            break
    while len(taken) < n:      # tiny sources: pad with shifted copies
        k = len(taken)
        taken.append((int((k * 37) % max(1, W - cw)), int((k * 53) % max(1, H - ch)), 0, 0))
        x, y = taken[-1][:2]
        taken[-1] = (x, y, x + cw, y + ch)
    return taken


def tile_grid(size: tuple[int, int], cols: int, rows: int, gap: int = 0) -> list[tuple[int, int, int, int]]:
    """Even grid of (x0, y0, x1, y1) boxes covering a frame."""
    W, H = size
    tw = (W - gap * (cols + 1)) / cols
    th = (H - gap * (rows + 1)) / rows
    return [(int(gap + c * (tw + gap)), int(gap + r * (th + gap)), int(gap + c * (tw + gap) + tw), int(gap + r * (th + gap) + th))
            for r in range(rows) for c in range(cols)]


# ---------------------------------------------------------------- source helpers

def synthetic_source(size: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Procedural stand-in when an effect is run without a source: layered
    fractal blobs, rings and bars (so tests and previews have structure)."""
    W, H = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    L = np.zeros((H, W), np.float32)
    for sg, amp in ((H / 3, 1.0), (H / 8, 0.6), (H / 24, 0.3)):
        n = ndi.gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), sg)
        L += amp * n / (n.std() + 1e-9)
    for _ in range(5):
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.08, 0.2) * H
        d = np.hypot(xx - cx, yy - cy)
        L += 1.4 * np.exp(-((d - r) / (0.015 * H)) ** 2)
    L = (L - L.min()) / (np.ptp(L) + 1e-9)
    hue = rng.uniform(0, 360)
    return oklch_field(0.25 + 0.6 * L, 0.05 + 0.08 * L, hue + 40 * L)


# ---------------------------------------------------------------- region masks (C)

def region_weights(img: np.ndarray, n: int = 2, mode: str = "detail", soft: float = 0.02, seed: int = 0) -> np.ndarray:
    """Partition the frame into n regions; returns weights (n, H, W) summing to 1.

    mode: 'detail'  quantiles of local detail energy (busy vs calm areas)
          'tone'    quantiles of tone (shadows / mids / lights)
          'cells'   jittered Voronoi cells (even spread of all effects)
          'superpixel' superpixels merged into n groups by colour k-means
          'diagonal' / 'columns' / 'rows'  hard geometric splits
    soft: seam softness as a fraction of the frame width (0 = hard seam).
    """
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed)
    L = luminance(img)
    if mode in ("detail", "tone"):
        f = blur(detail_energy(L, 30), 8 * scale_of(img.shape)) if mode == "detail" else blur(local_equalize(L, 2.0), 10 * scale_of(img.shape))
        qs = np.percentile(f, np.linspace(0, 100, n + 1)[1:-1])
        idx = np.digitize(f, qs)
        # soft via smoothing the one-hot
    elif mode == "cells":
        k = n * 6
        pts = rng.uniform(0, 1, (k, 2)) * [W, H]
        yy, xx = np.mgrid[0:H:4, 0:W:4]
        d = ((xx[..., None] - pts[:, 0]) ** 2 + (yy[..., None] - pts[:, 1]) ** 2)
        own = np.argmin(d, -1)
        group = rng.permutation(k) % n
        idx = cv2.resize(group[own].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(int)
    elif mode == "superpixel":
        lab = superpixels(img, n=300)
        means = to_oklab(label_means(img, lab)[None])[0]
        cent, grp = _kmeans(means, n, rng)
        order = np.argsort(cent[:, 0])
        rank = np.empty(n, int); rank[order] = np.arange(n)
        idx = rank[grp][lab]
    elif mode == "diagonal":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        t = (xx / W + 0.6 * yy / H) / 1.6
        idx = np.minimum((t * n).astype(int), n - 1)
    elif mode == "rows":
        idx = np.minimum((np.mgrid[0:H, 0:W][0] / H * n).astype(int), n - 1)
    else:  # columns
        idx = np.minimum((np.mgrid[0:H, 0:W][1] / W * n).astype(int), n - 1)
    wts = np.stack([(idx == i).astype(np.float32) for i in range(n)])
    if soft > 0:
        sg = soft * W
        wts = np.stack([blur(w, sg) for w in wts])
        wts /= wts.sum(0, keepdims=True) + 1e-9
    return wts


def composite_regions(images: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Blend n same-size renders with (n, H, W) weights."""
    out = np.zeros_like(images[0])
    for im, w in zip(images, weights):
        out += im * w[..., None]
    return np.clip(out, 0, 1).astype(np.float32)
