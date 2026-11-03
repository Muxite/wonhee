"""Dither, quantisation and pixel systems (families ``dither-*``).

What dotkit already has (binary Bayer / Floyd-Steinberg / Atkinson halftone masks, stipple) works on a
scalar field and gives a mask. This kit adds the missing parts:

* PHOTO IN, ANY PALETTE   colour images quantised to an arbitrary palette (library palette ramps,
                          k-means from the image, hex lists, retro presets) in OKLab.
* ALGORITHMS              none (posterise), Bayer 2/4/8/16, blue-noise threshold, white noise,
                          error diffusion: Floyd-Steinberg, Atkinson, Sierra, Sierra-2, Sierra-lite,
                          Jarvis-Judice-Ninke, Stucki, Burkes (serpentine), and colour stipple.
* PIXEL SCALE             the logical pixel grid is defined as a fraction of the frame width
                          (``cols`` logical pixels across), so the SAME picture appears at every output
                          size; below ``MIN_BLOCK`` output px per logical pixel the grid coarsens so
                          pixels still read at 288x192.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import features as F
from . import palette as pal
from . import printkit as K

MIN_BLOCK = 2.6          # output px per logical pixel at the smallest, so a dither still reads
MAX_CELLS = 260_000      # python error-diffusion budget

PRESETS = {
    "gameboy": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    "cga": ["#000000", "#55ffff", "#ff55ff", "#ffffff"],
    "mac": ["#1b1b1f", "#efeee6"],
    "sepia": ["#2b1d14", "#6b4a32", "#b08560", "#e9d8b8"],
    "riso-cobalt": ["#f2ead8", "#8fb4e8", "#1f4fa8", "#16203d"],
    "riso-pink": ["#f6ecdc", "#f29bb3", "#d6336c", "#35213a"],
    "pico": ["#1d2b53", "#7e2553", "#008751", "#ab5236", "#5f574f", "#c2c3c7", "#fff1e8", "#ff004d", "#ffa300", "#ffec27", "#00e436", "#29adff", "#83769c", "#ff77a8", "#ffccaa"],
    "ocean": ["#0b1a2b", "#154a6b", "#2f88a8", "#8fd0d6", "#f1ecd6"],
}


# ---------------------------------------------------------------- palettes

def ramp_from_palette(P, n: int, hues: int = 1) -> np.ndarray:
    """n colours spanning the library palette's lightness range in one hue family
    (hues=2: shadows in the structure hue, highlights drift to the accent hue)."""
    h = pal.rgb_to_oklch(P.structure)[2]
    ah = pal.rgb_to_oklch(P.accent)[2]
    dark = P.polarity == "dark"
    ls = np.linspace(0.16, 0.90, n) if dark else np.linspace(0.20, 0.96, n)
    out = []
    for i, l in enumerate(ls):
        t = i / max(n - 1, 1)
        bell = math.sin(math.pi * (0.12 + 0.8 * t))
        hh = h + (((ah - h + 180) % 360 - 180) * (t ** 1.5 if dark else (1 - t) ** 1.5) * (1 if hues > 1 else 0))
        out.append(pal.oklch_to_rgb(float(l), float(0.02 + 0.16 * bell), float(hh)))
    return np.array(out, np.float32)


def resolve_palette(spec: str, ctx, img: np.ndarray | None, n: int = 4, hues: int = 1) -> np.ndarray:
    """'palette' | 'palette:N' | 'auto:N' | preset name | 'duo:a,b' | '#hex,#hex,...' | names -> (k,3) rgb, dark -> light."""
    s = str(spec or "palette").strip()
    key, _, arg = s.partition(":")
    if key in ("palette", "ramp"):
        return ramp_from_palette(ctx.palette, int(arg) if arg else n, hues)
    if key == "auto":
        k = int(arg) if arg else n
        dp = F.dominant_palette(img, k, ctx.seed)
        lab = F.to_oklab(np.array([d["rgb"] for d in dp], np.float32)[None])[0]
        lab[:, 1:] *= 1.5                                 # k-means centres are averages: restore chroma
        Lmin, Lmax = lab[:, 0].min(), lab[:, 0].max()
        lab[:, 0] = 0.14 + (lab[:, 0] - Lmin) / max(Lmax - Lmin, 1e-3) * 0.78
        cols = F.from_oklab(lab[None])[0]
        Ls = lab[:, 0]
        return cols[np.argsort(Ls)]
    if s in PRESETS:
        cols = np.array([K.pal.hex_to_rgb(h) for h in PRESETS[s]], np.float32)
    else:
        cols = np.array(K.colors(s.replace("duo:", ""), ctx.palette), np.float32)
    Ls = F.to_oklab(cols[None])[0, :, 0]
    return cols[np.argsort(Ls)]


# ---------------------------------------------------------------- nearest colour LUT

class Nearest:
    """Nearest palette colour in OKLab through a 3-D lookup grid (fast scalar and array queries)."""

    def __init__(self, lab_pal: np.ndarray, n: int = 40):
        self.pal = lab_pal.astype(np.float64)
        self.n = n
        self.lo = np.array([0.0, -0.40, -0.40])
        self.hi = np.array([1.0, 0.40, 0.40])
        g = [np.linspace(self.lo[i], self.hi[i], n) for i in range(3)]
        mesh = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        d = ((mesh[:, None, :] - self.pal[None]) ** 2 * np.array([1.0, 0.8, 0.8])).sum(-1)
        self.tab = np.argmin(d, 1).astype(np.int32).reshape(n, n, n)
        self.scale = (n - 1) / (self.hi - self.lo)

    def query(self, lab: np.ndarray) -> np.ndarray:
        i = np.clip(np.rint((lab - self.lo) * self.scale).astype(int), 0, self.n - 1)
        return self.tab[i[..., 0], i[..., 1], i[..., 2]]


# ---------------------------------------------------------------- ordered / noise thresholds

def bayer(n: int) -> np.ndarray:
    m = np.array([[0, 2], [3, 1]], np.float64)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (n * n)


def blue_noise(n: int, rng) -> np.ndarray:
    """Approximate blue-noise tile: white noise minus its low-pass, rank-transformed to uniform."""
    w = rng.normal(0, 1, (n, n))
    from scipy import ndimage
    hp = w - ndimage.gaussian_filter(w, 1.6, mode="wrap")
    hp = hp - ndimage.gaussian_filter(hp, 0.6, mode="wrap") * 0
    r = np.argsort(np.argsort(hp.ravel())).reshape(n, n)
    return (r + 0.5) / (n * n)


def threshold_map(kind: str, shape, rng) -> np.ndarray:
    h, w = shape
    if kind.startswith("bayer"):
        n = int(kind[5:] or 4)
        t = bayer(n)
    elif kind == "blue":
        t = blue_noise(64, rng)
    elif kind == "white":
        return rng.random((h, w))
    else:
        raise ValueError(kind)
    return np.tile(t, (h // t.shape[0] + 1, w // t.shape[1] + 1))[:h, :w]


def ordered(lab: np.ndarray, near: Nearest, kind: str, spread: float, rng) -> np.ndarray:
    """Ordered / noise dither of an OKLab grid to palette indices: offset = (t - 0.5) * spread."""
    h, w, _ = lab.shape
    t = threshold_map(kind, (h, w), rng) - 0.5
    ls = np.sort(near.pal[:, 0])
    gap = float(np.median(np.diff(ls))) if len(ls) > 1 else 0.5      # typical lightness step between palette colours
    off = np.stack([t * spread * gap * 2.0, t * spread * gap * 0.7, t * spread * gap * 0.7], -1)
    return near.query(lab + off)


# ---------------------------------------------------------------- error diffusion

KERNELS = {      # (dx, dy, weight) / divisor
    "fs": ([(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16),
    "atkinson": ([(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8),
    "sierra": ([(1, 0, 5), (2, 0, 3), (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2), (-1, 2, 2), (0, 2, 3), (1, 2, 2)], 32),
    "sierra2": ([(1, 0, 4), (2, 0, 3), (-2, 1, 1), (-1, 1, 2), (0, 1, 3), (1, 1, 2), (2, 1, 1)], 16),
    "sierra-lite": ([(1, 0, 2), (-1, 1, 1), (0, 1, 1)], 4),
    "jjn": ([(1, 0, 7), (2, 0, 5), (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3), (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)], 48),
    "stucki": ([(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2), (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)], 42),
    "burkes": ([(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)], 32),
}


def diffuse(lab: np.ndarray, near: Nearest, kernel: str = "fs", serpentine: bool = True, strength: float = 1.0) -> np.ndarray:
    """Error diffusion of an OKLab grid to palette indices (pure python, <= MAX_CELLS)."""
    taps, div = KERNELS[kernel]
    h, w, _ = lab.shape
    L = lab[..., 0].astype(np.float64).ravel().tolist()
    A = lab[..., 1].astype(np.float64).ravel().tolist()
    B = lab[..., 2].astype(np.float64).ravel().tolist()
    pl = near.pal
    tab, n, lo, sc = near.tab, near.n, near.lo, near.scale
    tabl = tab.ravel().tolist()
    n2 = n * n
    out = [0] * (h * w)
    s = strength / div
    plL, plA, plB = pl[:, 0].tolist(), pl[:, 1].tolist(), pl[:, 2].tolist()
    lo0, lo1, lo2, s0, s1, s2 = lo[0], lo[1], lo[2], sc[0], sc[1], sc[2]
    nm = n - 1
    for y in range(h):
        flip = serpentine and (y & 1)
        xs = range(w - 1, -1, -1) if flip else range(w)
        row = y * w
        for x in xs:
            i = row + x
            l, a, b = L[i], A[i], B[i]
            i0 = int((l - lo0) * s0 + 0.5); i1 = int((a - lo1) * s1 + 0.5); i2 = int((b - lo2) * s2 + 0.5)
            i0 = 0 if i0 < 0 else (nm if i0 > nm else i0)
            i1 = 0 if i1 < 0 else (nm if i1 > nm else i1)
            i2 = 0 if i2 < 0 else (nm if i2 > nm else i2)
            k = tabl[i0 * n2 + i1 * n + i2]
            out[i] = k
            el, ea, eb = (l - plL[k]) * s, (a - plA[k]) * s, (b - plB[k]) * s
            for dx, dy, wt in taps:
                xx = x - dx if flip else x + dx
                yy = y + dy
                if 0 <= xx < w and yy < h:
                    j = yy * w + xx
                    L[j] += el * wt; A[j] += ea * wt; B[j] += eb * wt
    return np.array(out, np.int32).reshape(h, w)


ALGOS = ["none", "bayer2", "bayer4", "bayer8", "bayer16", "blue", "white"] + list(KERNELS)


def quantise(lab: np.ndarray, pal_rgb: np.ndarray, algo: str, rng, spread: float = 1.0, serpentine: bool = True, strength: float = 1.0) -> np.ndarray:
    """OKLab grid -> palette index grid."""
    near = Nearest(F.to_oklab(pal_rgb[None])[0])
    if algo == "none":
        return near.query(lab)
    if algo in KERNELS:
        return diffuse(lab, near, algo, serpentine, strength)
    return ordered(lab, near, algo, spread, rng)


# ---------------------------------------------------------------- pixel grid

def grid_size(size, cols: int, min_block: float = MIN_BLOCK) -> tuple[int, int, float]:
    """Logical grid (gw, gh) for ``cols`` logical pixels across the frame; coarsened so a logical pixel
    is never smaller than ``min_block`` output px, and capped for the diffusion budget."""
    W, H = size
    b = max(W / max(cols, 4), min_block)
    gw, gh = max(int(round(W / b)), 4), max(int(round(H / b)), 4)
    while gw * gh > MAX_CELLS:
        b *= 1.2
        gw, gh = int(round(W / b)), int(round(H / b))
    return gw, gh, b


def to_grid(img: np.ndarray, gw: int, gh: int) -> np.ndarray:
    return cv2.resize(np.ascontiguousarray(img), (gw, gh), interpolation=cv2.INTER_AREA)


def to_frame(idx: np.ndarray, pal_rgb: np.ndarray, size, gap: float = 0.0, gap_color=None) -> np.ndarray:
    """Palette index grid -> frame, nearest-neighbour blocks; optional gap lines between logical pixels."""
    W, H = size
    gh, gw = idx.shape
    img = pal_rgb[idx].astype(np.float32)
    out = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    if gap > 0:
        b = W / gw
        g = max(gap * b, 1.0 if b >= 5 else 0.0)
        if g > 0:
            xs = (np.arange(W) % b) < g
            ys = (np.arange(H) % (H / gh)) < g
            m = (xs[None, :] | ys[:, None])
            col = np.asarray(gap_color if gap_color is not None else pal_rgb[0], np.float32)
            out[m] = out[m] * 0.35 + col * 0.65
    return out


def prep_tone(img: np.ndarray, *, local: float = 0.0, contrast: float = 1.0, gamma: float = 1.0, saturation: float = 1.0, bright: float = 0.0) -> np.ndarray:
    """Tone prep in OKLab before quantising: local equalisation, contrast about mid, gamma, saturation."""
    lab = F.to_oklab(img)
    L = lab[..., 0]
    if local > 0:
        L = F.local_equalize(L, 2.5, mix=local)
    L = F.stretch(L, 1, 99) if contrast > 0 else L
    L = np.clip((L - 0.5) * contrast + 0.5 + bright, 0, 1) ** gamma
    lab = np.stack([L, lab[..., 1] * saturation, lab[..., 2] * saturation], -1)
    return lab
