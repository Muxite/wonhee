"""Colour grading and finishing core for the wonhee art library ("looks").

Everything is deterministic numpy / cv2 on float32 sRGB RGB in 0..1 (the
``ctx.source`` convention; NOT the BGR uint8 of ``wonhee.edit``). No AI.

A LOOK = ``Grade`` (pointwise colour transform, bakeable into a 33^3 LUT, a
.cube file, an SVG filter or a CSS filter chain) + ``Finish`` (spatial
finishing: grain, halation, bloom, vignette, chromatic aberration, flash
falloff, scanlines ... which no LUT can hold) + metadata. Looks live in
``LOOKS`` and are also registered as ``grade-<name>`` effects (see
``effects/grade_*.py``).

Pointwise ops (order in ``Grade.apply``): exposure, white balance, highlight
shoulder, channel matrix, master + per-channel pchip curves, log/linear
contrast, black lift / white cap, OKLCH saturation / vibrance / selective
hue ranges / hue pull, split-toning, desaturation to mono, gradient maps,
overlay-luma (bleach bypass), skin-tone protection, gamut mapping.

Rendering a look always goes through the 33^3 LUT (``Grade.lut``), so what
you see equals what ``export-cube`` writes (minus the Finish).

Public API
    Grade, Finish, Look, LOOKS, register_look, render_look, apply_lut, build_lut,
    write_cube, read_cube, describe, nearest_looks, svg_filter, css_filter,
    contact_sheet (see grade_cli)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from . import features as F

# ------------------------------------------------------------------ colour maths

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1i = np.linalg.inv(_M1)
_M2i = np.linalg.inv(_M2)
LUMA = np.array([0.2126, 0.7152, 0.0722])


def srgb_to_lin(x):
    x = np.asarray(x)
    return np.where(x <= 0.04045, x / 12.92, ((np.clip(x, 0, None) + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(np.asarray(x), 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def rgb_to_oklab(rgb):
    """sRGB (...,3) -> OKLab (...,3), float64, no clipping."""
    lin = srgb_to_lin(np.clip(rgb, 0, 1))
    lms = np.cbrt(lin @ _M1.T)
    return lms @ _M2.T


def oklab_to_lin(lab):
    lms = (np.asarray(lab) @ _M2i.T) ** 3
    return lms @ _M1i.T


def oklab_to_rgb(lab, gamut: bool = True):
    """OKLab -> sRGB. With ``gamut`` out-of-gamut colours lose chroma (toward the
    same lightness) instead of clipping per channel, so hue is kept."""
    lab = np.asarray(lab, np.float64)
    lin = oklab_to_lin(lab)
    if gamut:
        bad = (lin.min(-1) < -2e-3) | (lin.max(-1) > 1 + 2e-3)
        if bad.any():
            lb = lab[bad]
            lo = np.zeros(len(lb)); hi = np.ones(len(lb))
            for _ in range(10):
                mid = (lo + hi) / 2
                t = np.concatenate([lb[:, :1], lb[:, 1:] * mid[:, None]], -1)
                li = oklab_to_lin(t)
                ok = (li.min(-1) >= -2e-3) & (li.max(-1) <= 1 + 2e-3)
                lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
            lin = lin.copy()
            lin[bad] = oklab_to_lin(np.concatenate([lb[:, :1], lb[:, 1:] * lo[:, None]], -1))
    return lin_to_srgb(np.clip(lin, 0, 1))


def lab_to_lch(lab):
    return lab[..., 0], np.hypot(lab[..., 1], lab[..., 2]), np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


def lch_to_lab(L, C, h):
    r = np.radians(h)
    return np.stack([L, C * np.cos(r), C * np.sin(r)], -1)


def hue_diff(a, b):
    return (a - b + 180) % 360 - 180


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def ink(L: float, C: float, h: float) -> np.ndarray:
    """OKLCH colour as an sRGB triple (gamut mapped)."""
    return np.asarray(oklab_to_rgb(lch_to_lab(np.float64(L), np.float64(C), np.float64(h))[None])[0])


# ------------------------------------------------------------------ curves

class Curve:
    """Monotone pchip tone curve through control points (x, y) in 0..1."""

    def __init__(self, pts: Sequence[tuple[float, float]] | None = None):
        pts = sorted((float(x), float(y)) for x, y in (pts or [(0, 0), (1, 1)]))
        if pts[0][0] > 0:
            pts.insert(0, (0.0, pts[0][1]))
        if pts[-1][0] < 1:
            pts.append((1.0, pts[-1][1]))
        xs, ys = zip(*pts)
        self.pts = list(zip(xs, ys))
        self._xs = np.linspace(0, 1, 1025)
        self._ys = np.clip(PchipInterpolator(xs, ys, extrapolate=True)(self._xs), 0, 1)

    def __call__(self, x):
        return np.interp(x, self._xs, self._ys)

    @property
    def identity(self) -> bool:
        return all(abs(x - y) < 1e-9 for x, y in self.pts)


def s_curve(strength: float = 0.5, pivot: float = 0.5) -> Curve:
    """Gentle filmic S: pushes shadows down and highlights up around ``pivot``."""
    s = float(strength)
    return Curve([(0, 0), (0.15, 0.15 - 0.09 * s), (0.3, 0.3 - 0.07 * s * (0.5 / max(pivot, 0.2))),
                  (pivot, pivot), (0.72, 0.72 + 0.075 * s), (0.88, 0.88 + 0.06 * s), (1, 1)])


# ------------------------------------------------------------------ Grade (pointwise)

def _cv(c):
    return c if isinstance(c, Curve) else Curve(c)


@dataclass
class Grade:
    """Pointwise colour transform. All fields optional; defaults are identity.

    exposure       stops (linear light)
    temp, tint     -1..1 white balance (+temp warmer, +tint magenta)
    shoulder       0..1 highlight roll-off strength (knee ``knee``)
    matrix         3x3 channel mixer (rows = output r,g,b) applied to encoded rgb
    curve, curve_r/g/b   pchip control points in encoded space
    contrast       linear-light contrast around 0.18 (gamma-like, 1 = none)
    lift, cap      black lift / white cap in encoded space (out = lift + x*(cap-lift))
    sat, vib       OKLCH chroma scale / vibrance (boost of low-chroma pixels)
    ranges         selective colour: [(hue, width, dh, ds, dl)] in OKLCH
    pull           (hue, amount, keep): rotate every hue toward ``hue``
    split          dict(sh=(hue, chroma), hi=(hue, chroma), mid=(hue, chroma), bal=0)
    mono           0..1 mix toward luma (weights ``mono_w``) before gradient map
    gmap           [(pos, rgb)] gradient map on OKLab lightness, mixed by ``gmap_mix``
    bypass         0..1 overlay-luma blend (bleach bypass)
    skin           0..1 protection: keep the source's skin chroma/hue
    """
    exposure: float = 0.0
    temp: float = 0.0
    tint: float = 0.0
    shoulder: float = 0.0
    knee: float = 0.72
    matrix: Any = None
    curve: Any = None
    curve_r: Any = None
    curve_g: Any = None
    curve_b: Any = None
    contrast: float = 1.0
    lift: float = 0.0
    cap: float = 1.0
    sat: float = 0.0
    vib: float = 0.0
    ranges: list = field(default_factory=list)
    pull: Any = None
    split: dict = field(default_factory=dict)
    mono: float = 0.0
    mono_w: tuple = (0.30, 0.59, 0.11)
    gmap: list = field(default_factory=list)
    gmap_mix: float = 1.0
    bypass: float = 0.0
    skin: float = 0.0

    # ---- application
    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """rgb (..., 3) in 0..1 -> graded rgb 0..1. Intended for LUT-sized or small arrays."""
        src = np.clip(np.asarray(rgb, np.float64), 0, 1)
        x = src
        if self.exposure or self.temp or self.tint:
            lin = srgb_to_lin(x) * 2.0 ** self.exposure
            t, g = self.temp, self.tint
            gains = np.array([1 + 0.34 * t + 0.10 * g, 1 - 0.18 * g, 1 - 0.40 * t + 0.10 * g])
            x = lin_to_srgb(lin * gains)
        if self.shoulder:
            k = self.knee
            t = np.clip((x - k) / (1 - k), 0, None)
            roll = np.where(x > k, k + (1 - k) * (1 - np.exp(-t)), x)
            x = x + (roll - x) * self.shoulder
        x = np.clip(x, 0, 1)
        if self.matrix is not None:
            x = np.clip(x @ np.asarray(self.matrix, np.float64).T, 0, 1)
        if self.curve is not None:
            x = _cv(self.curve)(x)
        if self.curve_r is not None or self.curve_g is not None or self.curve_b is not None:
            x = np.stack([(_cv(c)(x[..., i]) if c is not None else x[..., i])
                          for i, c in enumerate((self.curve_r, self.curve_g, self.curve_b))], -1)
        if self.contrast != 1.0:
            lin = srgb_to_lin(np.clip(x, 0, 1))
            lin = 0.18 * (np.maximum(lin, 1e-6) / 0.18) ** self.contrast
            x = np.clip(lin_to_srgb(np.clip(lin, 0, 1)), 0, 1)
        if self.lift or self.cap != 1.0:
            x = self.lift + x * (self.cap - self.lift)
        x = np.clip(x, 0, 1)

        needs_lab = (self.sat or self.vib or self.ranges or self.pull or self.split or self.mono or self.gmap or self.skin)
        if self.bypass:
            y = (x @ LUMA)[..., None]
            ov = np.where(x < 0.5, 2 * x * y, 1 - 2 * (1 - x) * (1 - y))
            x = x + (np.clip(ov, 0, 1) - x) * self.bypass
        if needs_lab:
            lab = rgb_to_oklab(x)
            L, C, h = lab_to_lch(lab)
            if self.mono:
                C = C * (1 - self.mono)
            if self.sat:
                C = C * max(0.0, 1 + self.sat)
            if self.vib:
                C = C * (1 + self.vib * np.clip(1 - C / 0.22, 0, 1))
            for (h0, wd, dh, ds, dl) in self.ranges:
                w = np.exp(-(hue_diff(h, h0) / wd) ** 2) * smooth(0.012, 0.05, C)
                h = h + dh * w
                C = C * (1 + ds * w)
                L = L + dl * w
            if self.pull is not None:
                th, amt, keep = self.pull
                w = smooth(0.0, 0.035, C) * amt
                h = h + hue_diff(th, h) * w
                C = C * (1 + (1 - keep) * 0.0)
            lab = lch_to_lab(L, C, h)
            if self.mono:   # mono mix uses chosen channel weights for lightness
                yl = rgb_to_oklab(np.repeat((x @ np.asarray(self.mono_w))[..., None], 3, -1))[..., 0]
                lab = np.concatenate([(lab[..., :1] + (yl[..., None] - lab[..., :1]) * self.mono), lab[..., 1:]], -1)
            if self.gmap:
                lab = self._gmap(lab)
            if self.split:
                lab = self._split(lab)
            if self.skin:
                lab = self._skin(lab, rgb_to_oklab(src))
            x = oklab_to_rgb(lab)
        return np.clip(x, 0, 1)

    def _gmap(self, lab):
        pos = np.array([p for p, _ in self.gmap], np.float64)
        cols = np.array([rgb_to_oklab(np.asarray(c, np.float64)) for _, c in self.gmap])
        L = np.clip(lab[..., 0], 0, 1)
        m = np.stack([np.interp(L, pos, cols[:, i]) for i in range(3)], -1)
        mix = self.gmap_mix
        return lab * (1 - mix) + m * mix

    def _split(self, lab):
        L = lab[..., 0]
        s = self.split
        bal = s.get("bal", 0.0)
        sh = 1 - smooth(0.22 - 0.2 * bal, 0.72 - 0.2 * bal, L)
        hi = smooth(0.38 + 0.2 * bal, 0.88 + 0.1 * bal, L)
        mid = np.exp(-((L - 0.55) / 0.22) ** 2)
        ab = np.zeros(lab.shape[:-1] + (2,))
        for key, w in (("sh", sh), ("mid", mid), ("hi", hi)):
            if key in s:
                hh, cc = s[key]
                ab = ab + w[..., None] * cc * np.array([math.cos(math.radians(hh)), math.sin(math.radians(hh))])
        return np.concatenate([lab[..., :1], lab[..., 1:] + ab], -1)

    def _skin(self, lab, src_lab):
        w = skin_weight(src_lab) * self.skin
        ab = lab[..., 1:] * (1 - w[..., None]) + src_lab[..., 1:] * w[..., None]
        return np.concatenate([lab[..., :1], ab], -1)

    # ---- LUT
    def lut(self, n: int = 33, intensity: float = 1.0) -> np.ndarray:
        """(n, n, n, 3) float32 indexed [b, g, r] (so flattening gives .cube order: red fastest)."""
        g = np.linspace(0, 1, n)
        b, gg, r = np.meshgrid(g, g, g, indexing="ij")
        grid = np.stack([r, gg, b], -1).reshape(-1, 3)
        out = self.apply(grid)
        if intensity != 1.0:
            out = grid + (out - grid) * intensity
        return np.clip(out, 0, 1).reshape(n, n, n, 3).astype(np.float32)


def skin_weight(lab) -> np.ndarray:
    """0..1 skin likelihood from OKLab: orange-pink hue wedge, moderate chroma, mid-to-light
    (the idea of wonhee.edit.filters.skin_weight, in OKLab)."""
    L, C, h = lab_to_lch(np.asarray(lab, np.float64))
    dh = hue_diff(h, 45.0)
    return (np.exp(-(dh / 24.0) ** 2) * smooth(0.025, 0.06, C) * (1 - smooth(0.14, 0.2, C))
            * smooth(0.25, 0.45, L) * (1 - smooth(0.93, 0.99, L)))


# ------------------------------------------------------------------ LUT apply / .cube

def identity_lut(n: int = 33) -> np.ndarray:
    return Grade().lut(n)


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinear 3D LUT on float RGB (H, W, 3) 0..1. lut[b, g, r]."""
    n = lut.shape[0]
    x = np.clip(img.astype(np.float32), 0, 1) * (n - 1)
    i0 = np.minimum(x.astype(np.int32), n - 2)
    f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    flat = lut.reshape(-1, 3)
    base = (b0 * n + g0) * n + r0
    out = 0
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = (fb if db else 1 - fb) * (fg if dg else 1 - fg) * (fr if dr else 1 - fr)
                out = out + w * flat[base + (db * n + dg) * n + dr]
    return out.astype(np.float32)


def build_lut(fn: Callable[[np.ndarray], np.ndarray], n: int = 33) -> np.ndarray:
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([r, gg, b], -1).reshape(-1, 3)
    return np.clip(fn(grid), 0, 1).reshape(n, n, n, 3).astype(np.float32)


def write_cube(lut: np.ndarray, path: str, title: str = "wonhee look") -> None:
    n = lut.shape[0]
    with open(path, "w") as fh:
        fh.write(f'TITLE "{title}"\n# wonhee.art.grading (deterministic; finishing effects such as grain are not part of a LUT)\n')
        fh.write(f"LUT_3D_SIZE {n}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        for r, g, b in lut.reshape(-1, 3):
            fh.write(f"{r:.6f} {g:.6f} {b:.6f}\n")


def read_cube(path: str) -> np.ndarray:
    n, rows, dmin, dmax = None, [], 0.0, 1.0
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("TITLE"):
            continue
        if line.startswith("LUT_3D_SIZE"):
            n = int(line.split()[1])
        elif line.startswith("DOMAIN_MIN"):
            dmin = float(line.split()[1])
        elif line.startswith("DOMAIN_MAX"):
            dmax = float(line.split()[1])
        elif line.startswith("LUT_1D_SIZE"):
            raise ValueError("1D LUTs are not supported")
        else:
            rows.append([float(v) for v in line.split()[:3]])
    if n is None or len(rows) != n ** 3:
        raise ValueError(f".cube: expected {n}^3 rows, got {len(rows)}")
    return ((np.asarray(rows, np.float32) - dmin) / (dmax - dmin)).reshape(n, n, n, 3)


# ------------------------------------------------------------------ Finish (spatial)

def _blur(a, sigma):
    if sigma < 0.3:
        return a
    return cv2.GaussianBlur(np.ascontiguousarray(a), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


@dataclass
class Finish:
    """Spatial finishing. Sizes are px at a 1500 px master and scale with the frame.

    grain       0..1 amount; grain_size px; grain_chroma 0..1 (0 = mono grain, 1 = independent per channel);
                response shape: strongest in shadows/mids, calmer in highlights ("luminance dependent")
    halation    (amount, radius, (r,g,b) tint, threshold)   red glow around bright lights
    bloom       (amount, radius, threshold, (r,g,b) tint)   soft white glow
    soften      0..1 diffusion (pro-mist): highlights bleed into a blurred copy
    clarity     0..1 local contrast (unsharp, big radius); sharpen: unsharp small radius (halos)
    vignette    (amount, power, (r,g,b) tint)
    flash       (amount, radius_frac) flash falloff: bright centre, dark corners, in linear light
    ca          px lateral chromatic aberration at the frame edge
    dither      triangular dither before 8-bit quantisation (kills banding)
    chroma_noise  0..1 low-frequency colour blotches (small sensors)
    vhs         dict(bleed px, luma_blur px, jitter px, ghost (px, amt), noise 0..1, tracking 0..1)
    crt         dict(scan 0..1, period px, mask 0..1, barrel 0..1)
    """
    grain: float = 0.0
    grain_size: float = 1.1
    grain_chroma: float = 0.25
    halation: Any = None
    bloom: Any = None
    soften: float = 0.0
    clarity: float = 0.0
    sharpen: float = 0.0
    vignette: Any = None
    flash: Any = None
    ca: float = 0.0
    dither: bool = True
    chroma_noise: float = 0.0
    vhs: dict = field(default_factory=dict)
    crt: dict = field(default_factory=dict)

    def scaled(self, k: float, grain_mult: float = 1.0) -> "Finish":
        """Strength scaled by intensity k (0 = no finishing at all)."""
        f = Finish(**{**asdict(self)})
        f.grain = self.grain * k * grain_mult
        if self.halation:
            a, r, c, t = self.halation; f.halation = (a * k, r, c, t)
        if self.bloom:
            a, r, t, c = self.bloom; f.bloom = (a * k, r, t, c)
        f.soften = self.soften * k; f.clarity = self.clarity * k; f.sharpen = self.sharpen * k
        if self.vignette:
            a, p, c = self.vignette; f.vignette = (a * k, p, c)
        if self.flash:
            a, r = self.flash; f.flash = (a * k, r)
        f.ca = self.ca * k; f.chroma_noise = self.chroma_noise * k
        f.dither = self.dither and k > 0
        if self.vhs:
            f.vhs = {**self.vhs, "scale": k}
        if self.crt:
            f.crt = {**self.crt, "scale": k}
        return f

    def apply(self, img: np.ndarray, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(int(seed) + 7919)
        H, W = img.shape[:2]
        s = max(W, H) / 1500.0
        x = img.astype(np.float32)
        lum = x @ LUMA.astype(np.float32)
        if self.soften > 0:
            for sg, wt in ((6 * s + 1, 0.5), (18 * s + 1, 0.5)):
                bl = _blur(x, sg)
                x = x + (bl - x) * (self.soften * wt * (0.35 + 0.65 * F.smoothstep(0.2, 0.8, lum))[..., None])
        if self.clarity > 0:
            x = x + (x - _blur(x, 30 * s + 2)) * self.clarity * 0.8
        if self.sharpen > 0:
            x = x + (x - _blur(x, max(0.6, 1.2 * s))) * self.sharpen * 1.5
        x = np.clip(x, 0, 1)
        if self.ca:
            x = _lateral_ca(x, self.ca * s)
        if self.vhs:
            x = _vhs(x, self.vhs, s, rng)
        lum = x @ LUMA.astype(np.float32)
        if self.halation:
            amt, rad, tint, thr = self.halation
            bright = (np.clip(lum - thr, 0, 1) / max(1e-3, 1 - thr)) ** 1.2
            glow = 0.6 * _blur(bright, rad * s + 0.5) + 0.4 * _blur(bright, rad * 3 * s + 1)
            x = 1 - (1 - x) * (1 - np.clip(glow[..., None] * np.asarray(tint, np.float32) * amt * 2.0, 0, 1))
        if self.bloom:
            amt, rad, thr, tint = self.bloom
            lum = x @ LUMA.astype(np.float32)
            bright = np.clip(lum - thr, 0, 1) / max(1e-3, 1 - thr)
            glow = (_blur(x * bright[..., None], rad * s + 0.5) + _blur(x * bright[..., None], rad * 3.2 * s + 1)) * 0.5
            x = 1 - (1 - x) * (1 - np.clip(glow * np.asarray(tint, np.float32) * amt * 1.6, 0, 1))
        if self.flash:
            amt, rf = self.flash
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r2 = ((xx - W / 2) / (W * rf)) ** 2 + ((yy - H * 0.46) / (H * rf)) ** 2
            g = 1 + amt * 0.55 * np.exp(-r2 * 1.2) - amt * 0.55 * (1 - np.exp(-r2 * 0.35))
            x = lin_to_srgb(srgb_to_lin(x) * np.clip(g, 0.2, 2)[..., None]).astype(np.float32)
        if self.vignette:
            amt, pw, tint = self.vignette
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / 1.41
            v = np.clip(r, 0, 1) ** pw * amt
            x = x * (1 - v[..., None]) + np.asarray(tint, np.float32) * (v[..., None] * 0.3)
        if self.chroma_noise > 0:
            n = _blur(rng.standard_normal((H, W, 3)).astype(np.float32), 3.0 * s + 0.8)
            n = n / (n.std() + 1e-6)
            nl = n.mean(-1, keepdims=True)
            x = x + (n - nl) * self.chroma_noise * 0.03
        if self.grain > 0:
            x = _grain(x, self.grain, self.grain_size * s, self.grain_chroma, rng, s)
        if self.crt:
            x = _crt(x, self.crt, s)
        x = np.clip(x, 0, 1)
        if self.dither:
            x = np.clip(x + (rng.random((H, W, 1), np.float32) - rng.random((H, W, 1), np.float32)) / 255.0, 0, 1)
        return x.astype(np.float32)


def _grain(x, amount, size, chroma, rng, s):
    H, W = x.shape[:2]
    sg = max(0.0, size - 0.5) * 0.9
    def field(shape):
        n = rng.standard_normal(shape).astype(np.float32)
        n = _blur(n, sg) if sg > 0.2 else n
        return n / (n.std() + 1e-6)
    mono = field((H, W))[..., None]
    col = np.stack([field((H, W)) for _ in range(3)], -1) if chroma > 0 else 0
    g = mono * (1 - chroma) + col * chroma if chroma > 0 else mono
    lum = (x @ LUMA.astype(np.float32))[..., None]
    # film grain: strongest in shadows/mids, rolls off in highlights; never exactly zero in blacks
    resp = 0.30 + 0.70 * np.sqrt(np.clip(4 * lum * (1 - lum), 0, 1)) * (1 - 0.55 * F.smoothstep(0.55, 1.0, lum))
    # small frames: amplitude falls a little with scale so a 288 px thumb is not static
    amp = amount * 0.10 * (0.15 + 0.85 * min(1.0, (s * 2.0) ** 1.5))
    return x + g * resp * amp


def _lateral_ca(x, px):
    H, W = x.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2, H / 2
    dx, dy = (xx - cx) / cx, (yy - cy) / cy
    r = np.hypot(dx, dy) / 1.41
    out = x.copy()
    for ch, sign in ((0, 1.0), (2, -1.0)):
        k = sign * px * r
        mx = (xx - dx * k / max(1e-6, 1) * 1.0).astype(np.float32)
        my = (yy - dy * k).astype(np.float32)
        out[..., ch] = cv2.remap(np.ascontiguousarray(x[..., ch]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def _vhs(x, p, s, rng):
    H, W = x.shape[:2]
    k = p.get("scale", 1.0)
    # luma / chroma in YCrCb-ish space
    y = x @ LUMA.astype(np.float32)
    cb = (x[..., 2] - y) ; cr = (x[..., 0] - y)
    bleed = p.get("bleed", 8) * s * k + 0.01
    cb = cv2.GaussianBlur(cb, (0, 0), bleed, borderType=cv2.BORDER_REFLECT)
    cr = cv2.GaussianBlur(cr, (0, 0), bleed * 1.2)
    # chroma shifted right a little (tape chroma delay)
    sh = int(round(p.get("shift", 3) * s * k))
    if sh:
        cb = np.roll(cb, sh, 1); cr = np.roll(cr, sh, 1)
    lb = p.get("luma_blur", 1.2) * s * k
    if lb > 0.25:
        y = cv2.GaussianBlur(y, (0, 0), lb, borderType=cv2.BORDER_REFLECT)
        y = y + (y - cv2.GaussianBlur(y, (0, 0), lb * 2.5)) * 0.35   # tape edge enhancement
    ghost = p.get("ghost")
    if ghost:
        gpx, ga = ghost
        y = y * (1 - ga * k * 0.5) + np.roll(y, max(1, int(gpx * s)), 1) * ga * k * 0.5
    # row jitter: slow wobble + a tracking band near the bottom
    jit = p.get("jitter", 2.0) * s * k
    rows = np.arange(H, dtype=np.float32)
    off = np.sin(rows / (H / 9.0) + rng.uniform(0, 6.28)) * jit + _blur(rng.standard_normal(H).astype(np.float32)[:, None], 1.0)[:, 0] * jit * 0.6
    tr = p.get("tracking", 0.5) * k
    if tr > 0:
        band = np.exp(-((rows - H * 0.93) / (H * 0.025)) ** 2)
        off = off + band * 28 * s * tr * rng.choice([-1, 1])
    xs = (np.arange(W, dtype=np.float32)[None, :] - off[:, None]).astype(np.float32)
    ys = np.repeat(rows[:, None], W, 1)
    def warp(a):
        return cv2.remap(np.ascontiguousarray(a), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    y, cb, cr = warp(y), warp(cb), warp(cr)
    # noise: horizontal streaks
    nz = p.get("noise", 0.5) * k * min(1.0, s * 2.5)
    if nz > 0:
        n = rng.standard_normal((H, W)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 0.5 * s + 0.4) + cv2.GaussianBlur(n[::-1], (0, 0), 0.01) * 0
        streak = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), sigmaX=10 * s + 1, sigmaY=0.4)
        y = y + (n * 0.5 + streak * 5.0) * nz * 0.05
    r = y + cr; b = y + cb
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.clip(np.stack([r, g, b], -1), 0, 1)


def _crt(x, p, s):
    H, W = x.shape[:2]
    k = p.get("scale", 1.0)
    if p.get("barrel", 0) > 0:
        b = p["barrel"] * k
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx, ny = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
        r2 = nx ** 2 + ny ** 2
        f = 1 + b * 0.08 * r2
        mx, my = (nx * f * W / 2 + W / 2).astype(np.float32), (ny * f * H / 2 + H / 2).astype(np.float32)
        x = cv2.remap(np.ascontiguousarray(x), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    period = max(2.0, p.get("period", 4.0) * s)
    scan = p.get("scan", 0.5) * k * min(1.0, s * 3.0)
    if scan > 0:
        rows = np.arange(H, dtype=np.float32)
        line = 0.5 + 0.5 * np.cos(2 * np.pi * rows / period)
        x = x * (1 - scan * 0.55 * (1 - line) ** 1.5)[:, None, None]
        x = x * (1 + scan * 0.20 * line)[:, None, None]
    mask = p.get("mask", 0.3) * k * min(1.0, s * 2.5)
    if mask > 0:
        cols = np.arange(W)
        trio = np.zeros((W, 3), np.float32)
        trio[np.arange(W), cols % 3] = 1
        trio = 1 - mask * 0.45 + mask * 0.9 * trio
        x = x * trio[None]
    return x


# ------------------------------------------------------------------ Look registry

@dataclass
class Look:
    name: str
    mood: str
    doc: str
    best: str
    fails: str
    grade: Callable[[dict], Grade]
    finish: Callable[[dict], Finish]
    params: dict = field(default_factory=dict)     # extra params: name -> (default, description)
    tags: list = field(default_factory=list)
    web_note: str = ""


LOOKS: dict[str, Look] = {}

COMMON = {
    "intensity": (1.0, "0..1 strength of the whole look (colour and finishing)"),
    "grain": (1.0, "multiplier for the look's film grain (0 = none)"),
    "finish": (1.0, "0 = colour only (what a .cube holds), 1 = full finishing"),
    "skin": (-1.0, "skin-tone protection 0..1; -1 = the look's own default"),
    "crop": ("center", "center | feature, how a source of another aspect is cropped"),
}


def register_look(look: Look) -> Look:
    """Add to LOOKS and register the ``grade-<name>`` effect (idempotent per module)."""
    from .registry import effect
    LOOKS[look.name] = look
    params = dict(COMMON); params.update(look.params)
    doc = f"{look.mood}. {look.doc} Best on: {look.best} Fails on: {look.fails}"

    def fn(ctx, _name=look.name):
        return render_look(ctx.source, _name, ctx.size, ctx.seed, ctx.params)
    fn.__doc__ = doc
    fn.__name__ = "grade_" + look.name.replace("-", "_")
    fn.__module__ = look.grade.__module__
    effect("grade-" + look.name, family="grade", tags=["source", "grade", "look", *look.tags], params=params)(fn)
    return look


def ensure_loaded() -> dict[str, Look]:
    from .registry import discover
    discover()
    return LOOKS


@lru_cache(maxsize=256)
def _lut_cached(name: str, key: str, intensity: float) -> np.ndarray:
    look = LOOKS[name]
    p = json.loads(key)
    return look.grade(p).lut(33, intensity)


def look_params(name: str, params: dict | None) -> dict:
    look = LOOKS[name]
    p = {k: v[0] for k, v in COMMON.items()}
    p.update({k: v[0] for k, v in look.params.items()})
    p.update(params or {})
    return p


def look_lut(name: str, params: dict | None = None, intensity: float | None = None) -> np.ndarray:
    ensure_loaded()
    p = look_params(name, params)
    inten = float(p["intensity"] if intensity is None else intensity)
    key = json.dumps({k: v for k, v in p.items() if k not in ("intensity", "grain", "finish", "crop")}, sort_keys=True, default=str)
    return _lut_cached(name, key, round(inten, 6))


def look_lut_n(name: str, n: int, intensity: float = 1.0, params: dict | None = None) -> np.ndarray:
    """LUT of arbitrary size (65 recommended for hue-heavy looks such as infrared-false-colour)."""
    ensure_loaded()
    return LOOKS[name].grade(look_params(name, params)).lut(n, intensity)


def render_look(source: np.ndarray | None, name: str, size: tuple[int, int], seed: int = 0, params: dict | None = None) -> np.ndarray:
    """Apply a look to ``source`` (float32 RGB 0..1, any aspect; fitted to ``size`` W x H)."""
    ensure_loaded()
    p = look_params(name, params)
    W, H = size
    src = source if source is not None else F.synthetic_source((W, H), seed)
    img = F.fit_cover(np.asarray(src, np.float32)[..., :3], (W, H), p.get("crop", "center"))
    k = float(np.clip(p["intensity"], 0, 1))
    if k > 0:
        # 8-bit sources band when a grade stretches shadows: dither the source's quantisation first
        rng = np.random.default_rng(int(seed) + 104729)
        img = np.clip(img + (rng.random((H, W, 1), np.float32) - rng.random((H, W, 1), np.float32)) * (1.5 / 255), 0, 1)
    out = apply_lut(img, look_lut(name, p, k)) if k > 0 else img
    if float(p["finish"]) > 0 and k > 0:
        fin = LOOKS[name].finish(p).scaled(k * float(p["finish"]), float(p["grain"]))
        out = fin.apply(out, seed)
    return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ vibe finder

CHART_SIZE = (384, 256)


@lru_cache(maxsize=1)
def test_chart() -> np.ndarray:
    """Neutral reference chart: grey ramp, 12-hue wheel at two lightness/chroma levels, skin tones,
    and a smooth natural-ish gradient field (so tone, cast, saturation, black point and grain
    are all measurable)."""
    W, H = CHART_SIZE
    img = np.zeros((H, W, 3), np.float64)
    x = np.linspace(0, 1, W)
    img[: H // 4] = x[None, :, None]
    # hue patches
    ph = H // 4
    for row, (L, C) in enumerate(((0.72, 0.11), (0.5, 0.13))):
        for i in range(12):
            col = ink(L, C, i * 30 + 15)
            img[ph * (row + 1) : ph * (row + 2), i * (W // 12) : (i + 1) * (W // 12)] = col
    # skin tones + smooth field
    sk = [ink(L, c, h) for L, c, h in ((0.82, 0.05, 55), (0.72, 0.07, 52), (0.6, 0.08, 48), (0.45, 0.07, 45), (0.32, 0.06, 42))]
    y0 = ph * 3
    for i, c in enumerate(sk):
        img[y0 : y0 + (H - y0) // 2, i * (W // 5) : (i + 1) * (W // 5)] = c
    yy, xx = np.mgrid[0 : H - y0 - (H - y0) // 2, 0:W]
    fh = (H - y0) // 2
    fld = 0.5 + 0.32 * np.sin(xx / 61.0) * np.cos(yy / 17.0)
    hh = 200 + 160 * xx / W
    lab = lch_to_lab(0.25 + 0.65 * fld, 0.03 + 0.07 * fld, hh)
    img[y0 + fh :] = oklab_to_rgb(lab.reshape(-1, 3)).reshape(lab.shape)
    return np.clip(img, 0, 1).astype(np.float32)


VIBE_FIELDS = ["mean_L", "contrast", "sat", "warm", "sh_a", "sh_b", "hi_a", "hi_b", "grain", "black", "white"]
VIBE_W = np.array([1.0, 1.0, 1.4, 1.5, 1.6, 1.6, 1.6, 1.6, 0.8, 1.2, 0.9])
VIBE_SCALE = np.array([0.25, 0.3, 0.12, 0.08, 0.03, 0.03, 0.03, 0.03, 0.02, 0.12, 0.2])


def describe(img: np.ndarray) -> dict:
    """Small vibe vector of an image (float RGB 0..1 or a path).

    mean_L / contrast (p95-p5) in OKLab lightness, sat (mean OKLCH chroma), warm (mean chroma-weighted
    projection on the orange axis; + warm, - cool), shadow / highlight tint as OKLab (a, b) of the
    darkest / lightest quartile, grain (high-frequency noise in flat areas), black / white point
    (p1 / p99 lightness), dominant hues (OKLCH degrees with chroma share)."""
    if isinstance(img, str):
        from .compose import load
        img = load(img)
    im = np.asarray(img, np.float32)[..., :3]
    h0 = im.shape[0]
    if max(im.shape[:2]) > 640:
        k = 640 / max(im.shape[:2])
        im = cv2.resize(im, (max(1, int(im.shape[1] * k)), max(1, int(im.shape[0] * k))), interpolation=cv2.INTER_AREA)
    lab = rgb_to_oklab(im)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    p1, p5, p25, p75, p95, p99 = np.percentile(L, (1, 5, 25, 75, 95, 99))
    sh = L <= p25; hi = L >= p75
    ax = np.array([math.cos(math.radians(55)), math.sin(math.radians(55))])
    warm = float(((a * ax[0] + b * ax[1]) * (C > 0.02)).mean())
    # grain: residual after a small blur, only where the local gradient is small
    g = im @ LUMA.astype(np.float32)
    res = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    gm = np.hypot(*np.gradient(cv2.GaussianBlur(g, (0, 0), 2.0)))
    flat = gm < np.percentile(gm, 40)
    grain = float(res[flat].std())
    # dominant hues
    h = np.degrees(np.arctan2(b, a)) % 360
    m = C > 0.03
    doms = []
    if m.sum() > 0.01 * C.size:
        hist, _ = np.histogram(h[m], bins=36, range=(0, 360), weights=C[m])
        hist = np.convolve(np.r_[hist[-1:], hist, hist[:1]], [0.25, 0.5, 0.25], "valid")
        tot = hist.sum()
        for i in np.argsort(hist)[::-1][:6]:
            if hist[i] > 0.08 * tot and all(abs(hue_diff(i * 10 + 5, d["hue"])) > 35 for d in doms):
                doms.append({"hue": float(i * 10 + 5), "share": float(hist[i] / tot)})
            if len(doms) == 3:
                break
    v = {"mean_L": float(L.mean()), "contrast": float(p95 - p5), "sat": float(C.mean()), "warm": warm,
         "sh_a": float(a[sh].mean()), "sh_b": float(b[sh].mean()), "hi_a": float(a[hi].mean()), "hi_b": float(b[hi].mean()),
         "grain": grain, "black": float(p1), "white": float(p99), "dominant_hues": doms}
    return v


def vibe_vector(d: dict | np.ndarray) -> np.ndarray:
    if isinstance(d, np.ndarray):
        return d
    return np.array([d[k] for k in VIBE_FIELDS], np.float64)


@lru_cache(maxsize=1)
def _look_signatures() -> dict[str, np.ndarray]:
    """Per-look vibe vector measured on the neutral test chart (computed once)."""
    ensure_loaded()
    chart = test_chart()
    sig = {}
    for n in sorted(LOOKS):
        sig[n] = vibe_vector(describe(render_look(chart, n, CHART_SIZE, 0, {"crop": "center"})))
    return sig


def nearest_looks(img_or_vec, k: int = 5) -> list[tuple[str, float]]:
    """Rank looks by closeness of their effect on the neutral chart to a picture's vibe.
    Returns [(look, distance)] ascending (0 = identical statistics)."""
    is_vec = isinstance(img_or_vec, dict) or (isinstance(img_or_vec, np.ndarray) and img_or_vec.ndim == 1)
    v = vibe_vector(img_or_vec if is_vec else describe(img_or_vec))
    out = []
    for n, s in _look_signatures().items():
        d = float(np.sqrt((((v - s) / VIBE_SCALE) ** 2 * VIBE_W).sum() / VIBE_W.sum()))
        out.append((n, d))
    return sorted(out, key=lambda t: t[1])[:k]


# ------------------------------------------------------------------ web export (SVG filter / CSS chain)

def _samples() -> np.ndarray:
    rng = np.random.default_rng(1)
    g = np.linspace(0, 1, 33)
    ramp = np.repeat(g[:, None], 3, 1)
    return np.concatenate([rng.random((3000, 3)), np.tile(ramp, (8, 1))])


def _fit_svg(grade: Grade, intensity: float = 1.0):
    S = _samples()
    T = S + (grade.apply(S) - S) * intensity
    g = np.linspace(0, 1, 17)
    ramp = np.repeat(g[:, None], 3, 1)
    tab = ramp + (grade.apply(ramp) - ramp) * intensity       # per-channel shaper along the grey axis
    sh = np.stack([np.interp(S[:, i], g, tab[:, i]) for i in range(3)], -1)
    A = np.concatenate([sh, np.ones((len(S), 1))], 1)
    w = np.ones(len(S)); w[3000:] = 6.0
    M = np.linalg.lstsq(A * w[:, None], T * w[:, None], rcond=None)[0]      # (4, 3)
    pred = np.clip(A @ M, 0, 1)
    err = np.sqrt(((pred - T) ** 2).mean()) * 255
    return tab, M, float(err), float(np.abs(pred - T).max() * 255)


def svg_filter(name: str, intensity: float = 1.0, fid: str | None = None) -> tuple[str, dict]:
    """SVG <filter> approximating the look's colour grade: feComponentTransfer (per-channel table
    along the grey axis) then feColorMatrix (least-squares fitted). Returns (snippet, fidelity)."""
    ensure_loaded()
    grade = LOOKS[name].grade(look_params(name, None))
    tab, M, rms, mx = _fit_svg(grade, intensity)
    fid = fid or f"look-{name}"
    t = lambda i: " ".join(f"{v:.4f}" for v in tab[:, i])  # noqa: E731
    rows = []
    for c in range(3):
        rows.append(" ".join(f"{M[i, c]:.4f}" for i in range(3)) + f" 0 {M[3, c]:.4f}")
    rows.append("0 0 0 1 0")
    svg = (f'<svg width="0" height="0" style="position:absolute" aria-hidden="true"><filter id="{fid}" color-interpolation-filters="sRGB">\n'
           f'  <feComponentTransfer>\n    <feFuncR type="table" tableValues="{t(0)}"/>\n    <feFuncG type="table" tableValues="{t(1)}"/>\n'
           f'    <feFuncB type="table" tableValues="{t(2)}"/>\n  </feComponentTransfer>\n'
           f'  <feColorMatrix type="matrix" values="{" ".join(rows)}"/>\n</filter></svg>')
    return svg, {"rms_8bit": round(rms, 2), "max_8bit": round(mx, 1), "grade": _fidelity(rms)}


def _fidelity(rms: float) -> str:
    return "close" if rms < 4 else "good" if rms < 8 else "rough" if rms < 14 else "poor"


def _css_apply(x, th):
    b, c, s, sep, hue = th
    x = np.clip(x * b, 0, 1)
    x = np.clip((x - 0.5) * c + 0.5, 0, 1)
    if sep:
        m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
        x = np.clip(x + (x @ m.T - x) * sep, 0, 1)
    if hue:
        a = math.radians(hue); ca, sa = math.cos(a), math.sin(a)
        m = np.array([[0.213 + ca * 0.787 - sa * 0.213, 0.715 - ca * 0.715 - sa * 0.715, 0.072 - ca * 0.072 + sa * 0.928],
                      [0.213 - ca * 0.213 + sa * 0.143, 0.715 + ca * 0.285 + sa * 0.140, 0.072 - ca * 0.072 - sa * 0.283],
                      [0.213 - ca * 0.213 - sa * 0.787, 0.715 - ca * 0.715 + sa * 0.715, 0.072 + ca * 0.928 + sa * 0.072]])
        x = np.clip(x @ m.T, 0, 1)
    m = np.array([[0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
                  [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
                  [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s]])
    return np.clip(x @ m.T, 0, 1)


def css_filter(name: str, intensity: float = 1.0) -> tuple[str, dict]:
    """Best-fit CSS ``filter:`` chain (brightness contrast sepia hue-rotate saturate) by Powell
    search against the look's colour grade. Returns (css, fidelity)."""
    from scipy.optimize import minimize
    ensure_loaded()
    grade = LOOKS[name].grade(look_params(name, None))
    S = _samples(); T = S + (grade.apply(S) - S) * intensity
    w = np.ones(len(S)); w[3000:] = 4.0
    f = lambda th: float((((_css_apply(S, th) - T) ** 2).sum(-1) * w).mean())  # noqa: E731
    best = None
    for st in ((1, 1, 1, 0, 0), (1, 1.15, 0.8, 0.2, 0), (1, 1.1, 0.2, 0.6, -20), (0.95, 1.2, 1.2, 0, 180), (1, 1.1, 0.0, 0.8, 0)):
        r = minimize(f, st, method="Powell", bounds=[(0.5, 1.6), (0.5, 2.0), (0, 3), (0, 1), (-180, 180)], options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    b, c, s, sp, hu = best.x
    parts = []
    if abs(b - 1) > 0.01: parts.append(f"brightness({b:.2f})")
    if abs(c - 1) > 0.01: parts.append(f"contrast({c:.2f})")
    if sp > 0.02: parts.append(f"sepia({sp:.2f})")
    if abs(hu) > 1: parts.append(f"hue-rotate({hu:.0f}deg)")
    if abs(s - 1) > 0.01: parts.append(f"saturate({s:.2f})")
    rms = float(np.sqrt(((_css_apply(S, best.x) - T) ** 2).mean()) * 255)
    return "filter: " + (" ".join(parts) or "none") + ";", {"rms_8bit": round(rms, 2), "grade": _fidelity(rms)}


def css_overlays(name: str) -> str:
    """CSS for the non-colour parts that CSS can express: vignette as a radial-gradient overlay."""
    ensure_loaded()
    fin = LOOKS[name].finish(look_params(name, None))
    out = []
    if fin.vignette:
        amt = min(0.85, fin.vignette[0] * 1.1)
        out.append(f"/* vignette: put on an ::after overlay */\nbackground: radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,{amt:.2f}) 100%);")
    if fin.grain:
        out.append("/* grain: overlay an SVG feTurbulence noise tile (baseFrequency 0.8) with mix-blend-mode: overlay; opacity ~ "
                   f"{min(0.5, fin.grain * 0.35):.2f} */")
    if fin.halation or fin.bloom:
        out.append("/* glow: duplicate the image, blur(12px) brightness(1.2), mix-blend-mode: screen at low opacity */")
    return "\n".join(out)
