"""mckit: helpers for the multichan_* effects (multichannel / scientific composites).

A *channel* is one scalar field. Real channels come from ctx.data (dict name -> 2D array, a list of 2D arrays, or a
C x H x W array) and are normalised with an EXPLICIT range (param ``ranges``) or, if none is given, the 1..99
percentile range, which is reported. With no data and no source the channels are stand-ins: luminance of library
physics effects, flagged ``standin``. An RGB image (ctx.source) can feed the same machinery as aesthetic channels
(R, G, B, luma, saturation, edges, shadow / highlight masks), flagged ``aesthetic``: such a result is never
infrared, satellite or measured data.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from . import datakit as dk
from . import palette as pal
from .features import fit_cover, stretch


@dataclass
class Chan:
    name: str
    f: np.ndarray                 # H x W float32 0..1 at frame size
    rng: tuple[float, float]      # (lo, hi) in the channel's own units that map to 0..1
    kind: str = "data"            # data | standin | aesthetic


@dataclass
class Intake:
    chans: list[Chan]
    kind: str                     # data | standin | aesthetic
    note: str

    @property
    def meta(self) -> dict:
        d = {"channels": [c.name for c in self.chans], "ranges": {c.name: list(c.rng) for c in self.chans},
             "synthetic": self.kind == "standin", "standin": self.kind == "standin", "aesthetic": self.kind == "aesthetic",
             "note": self.note}
        return d


DERIVED = ("red", "green", "blue", "luma", "saturation", "edges", "shadows", "highlights", "warmth", "detail")


def derive_from_rgb(img: np.ndarray, name: str) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    L = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    if name == "red": return r
    if name == "green": return g
    if name == "blue": return b
    if name == "luma": return L
    if name == "saturation": return img.max(-1) - img.min(-1)
    if name == "warmth": return stretch(r - b)
    if name == "edges":
        gx = cv2.Sobel(cv2.GaussianBlur(L, (0, 0), 1.2), cv2.CV_32F, 1, 0); gy = cv2.Sobel(cv2.GaussianBlur(L, (0, 0), 1.2), cv2.CV_32F, 0, 1)
        return stretch(np.hypot(gx, gy), 2, 99)
    if name == "shadows": return 1 - np.clip(L / 0.5, 0, 1)
    if name == "highlights": return np.clip((L - 0.5) / 0.5, 0, 1)
    if name == "detail": return stretch(np.abs(L - cv2.GaussianBlur(L, (0, 0), 6)), 2, 99)
    raise KeyError(f"unknown derived channel {name!r}; known: {DERIVED}")


def _ranges(spec, names):
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {k: tuple(v) for k, v in spec.items()}
    return {n: tuple(v) for n, v in zip(names, spec)}


def intake(ctx, n: int, which=None, ranges=None, derived=("shadows", "luma", "edges"), standin_which=None, smooth: float = 7.0) -> Intake:
    W, H = ctx.size
    d = ctx.data
    rs = ranges
    if d is not None:
        if isinstance(d, dict):
            items = [(str(k), np.asarray(v, np.float32)) for k, v in d.items()]
        else:
            a = np.asarray(d, np.float32)
            items = [(f"ch{i}", a[i]) for i in range(a.shape[0])] if a.ndim == 3 else [("ch0", a)]
        if which:
            items = [it for it in items if it[0] in which] or items
        items = items[: max(n, 1)] if n else items
        rr = _ranges(rs, [k for k, _ in items])
        chans = []
        for k, a in items:
            f, used = dk.norm_range(a, rr.get(k))
            chans.append(Chan(k, dk.fit_field(f, ctx.size), used, "data"))
        return Intake(chans, "data", "channels from ctx.data; ranges explicit where given, else 1..99 percentile")
    if ctx.source is not None:
        src = fit_cover(ctx.source, ctx.size, "feature") if ctx.source.shape[:2] != (H, W) else ctx.source
        names = list(which or derived)
        chans = []
        for k in names[: max(n, 1)]:
            f = derive_from_rgb(src, k)
            f, used = dk.norm_range(f, None, (1, 99))
            chans.append(Chan(k, f.astype(np.float32), used, "aesthetic"))
        return Intake(chans, "aesthetic", "derived from an RGB image as an aesthetic input; not infrared, satellite or measured data")
    ch, names = dk.standin_fields(n, ctx.size, ctx.seed, which=standin_which, smooth=smooth)
    chans = [Chan(k, dk.fit_field(v, ctx.size), (0.0, 1.0), "standin") for k, v in ch.items()]
    return Intake(chans, "standin", "stand-in channels (luminance of library physics fields); decorative, not measurements")


# ----------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=128)
def tint_lut(hue: float, mode: str = "emit", n: int = 256, c: float = 0.17) -> np.ndarray:
    """Colour table for one channel at a hue. emit: dark -> bright glow (for add/screen on a dark base);
    ink: paper -> deep saturated colour (for multiply on a light base)."""
    t = np.linspace(0, 1, n)
    out = []
    for v in t:
        if mode == "emit":
            L = 0.10 + 0.76 * v ** 0.95
            C = c * min(1.0, v * 3.2) * (1 - 0.5 * v ** 3)
            h = hue + 14 * v
        else:
            L = 0.975 - 0.64 * v
            C = 0.015 + c * np.sin(min(1.0, v * 1.05) * np.pi * 0.52) ** 0.9
            h = hue
        out.append(pal.oklch_to_rgb(L, C, h % 360))
    return np.stack(out).astype(np.float32)


def soft_clip(x, top=0.965):
    return top * np.tanh(np.maximum(x, 0) / top)


def blend_stack(base: np.ndarray, layers: list[np.ndarray], opac: list[float], mode: str) -> np.ndarray:
    out = base.astype(np.float32).copy()
    for lay, op in zip(layers, opac):
        if mode == "multiply":
            out = out * (1 - op + op * lay)
        elif mode == "screen":
            out = 1 - (1 - out) * (1 - lay * op)
        elif mode == "max":
            out = np.maximum(out, base * (1 - op) + lay * op)
        elif mode == "over":
            out = out * (1 - op) + lay * op
        else:  # add
            out = out + (lay - base) * op if False else out + lay * op
    return out


def glow(img: np.ndarray, amount: float, sigma: float, thr: float = 0.55, tint=None) -> np.ndarray:
    if amount <= 0:
        return img
    L = img.max(-1)
    bright = np.clip((L - thr) / (1 - thr), 0, 1)[..., None] * img
    g = cv2.GaussianBlur(bright, (0, 0), sigma) * 0.7 + cv2.GaussianBlur(bright, (0, 0), sigma * 2.8) * 0.5
    if tint is not None:
        g = g * np.asarray(tint, np.float32)
    return 1 - (1 - img) * (1 - np.clip(g * amount, 0, 1))


def grain(img: np.ndarray, amount: float, rng, sigma: float = 0.7, lum_weight: float = 0.6) -> np.ndarray:
    if amount <= 0:
        return img
    H, W = img.shape[:2]
    g = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigma)
    g /= g.std() + 1e-6
    L = img.mean(-1, keepdims=True)
    w = (1 - lum_weight) + lum_weight * 4 * L * (1 - L)
    return img + (g[..., None] * amount * 0.07) * w


def hillshade(h: np.ndarray, az: float = 315.0, alt: float = 40.0, z: float = 6.0) -> np.ndarray:
    """Lambert shading of a 0..1 height field (z = vertical exaggeration relative to the frame width)."""
    H, W = h.shape
    gy, gx = np.gradient(cv2.GaussianBlur(h, (0, 0), 0.8))
    k = z * W / 100.0
    nx, ny, nz = -gx * k, -gy * k, np.ones_like(h)
    inv = 1 / np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    a, e = np.radians(az), np.radians(alt)
    lx, ly, lz = np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)
    return np.clip((nx * lx + ny * ly + nz * lz) * inv, 0, 1).astype(np.float32)


def contour_mask(f: np.ndarray, levels: int, width: float = 1.2, index_every: int = 5, index_boost: float = 1.0) -> np.ndarray:
    """Anti-aliased isolines of a 0..1 field at ``levels`` equal steps; every ``index_every``-th is heavier."""
    g = f * levels
    gy, gx = np.gradient(cv2.GaussianBlur(g, (0, 0), 0.8))
    mag = np.hypot(gx, gy) + 1e-4
    d = np.abs(g - np.round(g)) / mag                      # distance in px to the nearest level
    idx = (np.round(g).astype(int) % max(1, index_every)) == 0
    w = np.where(idx, width * (1 + index_boost), width)
    return np.clip(w / 2 + 0.5 - d, 0, 1).astype(np.float32) * (np.round(g) > 0)


def local_contrast(f: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    if amount <= 0:
        return f
    return np.clip(f + (f - cv2.GaussianBlur(f, (0, 0), sigma)) * amount, 0, 1).astype(np.float32)


def transform(f: np.ndarray, kind: str, strength: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    """Structure transforms of one channel: raw | ridge (thin filaments) | blobs (soft threshold discs) | haze
    (blur) | dots (sparse bright speckle) | edges."""
    if kind == "raw":
        return f
    if kind == "ridge":
        g = cv2.GaussianBlur(f, (0, 0), sigma)
        r = 1 - np.abs(2 * g - 1)
        return np.clip((r - 0.62) / 0.38, 0, 1) ** (1 / max(strength, 0.2))
    if kind == "blobs":
        g = cv2.GaussianBlur(f, (0, 0), sigma * 1.5)
        t = np.clip((g - 0.70) / 0.06, 0, 1)
        t = cv2.GaussianBlur(t, (0, 0), 1.0)
        return t * (0.7 + 0.3 * f)
    if kind == "haze":
        return cv2.GaussianBlur(f, (0, 0), sigma * 6)
    if kind == "dots":
        loc = f - cv2.GaussianBlur(f, (0, 0), sigma * 4)
        return np.clip((loc - 0.2) / 0.1, 0, 1)
    if kind == "edges":
        gy, gx = np.gradient(cv2.GaussianBlur(f, (0, 0), sigma))
        return stretch(np.hypot(gx, gy), 2, 99)
    raise KeyError(f"unknown transform {kind!r}")


def hue_set(ctx, hues, shift=None) -> list[float]:
    """Mood hues rotated with the palette: the palette hue defines the reference hue 40 (ember)."""
    ph = pal.rgb_to_oklch(ctx.palette.structure)[2]
    sh = (ph - 40.0) if shift is None else float(shift)
    return [(float(h) + sh) % 360 for h in hues]
