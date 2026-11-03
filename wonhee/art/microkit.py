"""Shared helpers for the `micro_*` (fractured microstructure) and `arch_*`
(architectural / perspective) effects.

Nothing here is an effect. Contents:
    hash01                    deterministic integer-lattice hash -> 0..1
    fbm                       cheap multi-octave smooth noise at any size
    families                  hue families + role colours for dark and light surfaces
    surface                   shared surface pass (grain, wear, bloom, tone)
    MOODS                     the mood vocabulary used in effect tags
Colour, structure, source and surface are separate parameter groups on every
effect built on this kit: `palette`/`families`/`hue_spread` (colour),
effect-specific structure params, `source_mix`/`hue_from_source` (source),
`grain`/`wear`/`glow`/`bevel` (surface)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import palette as pal

MOODS = ("engineered", "tactile", "precise", "urban", "monumental", "nocturnal")

SURFACE_PARAMS = {
    "grain": (0.35, "surface: film / substrate grain 0..1"),
    "wear": (0.25, "surface: dust, scratches and pitting 0..1"),
    "glow": (0.25, "surface: bloom of the brightest traces / lit windows 0..1"),
}
COLOUR_PARAMS = {
    "families": (3, "colour: hue families used (1..3): structure hue, second hue, accent hue"),
    "hue_spread": (0.0, "colour: degrees added between hue families (0 = take them from the palette)"),
    "chroma": (1.0, "colour: chroma multiplier on every role"),
    "contrast": (1.0, "colour: lightness gap between substrate and traces / lit windows"),
}
SOURCE_PARAMS = {
    "source_mix": (1.0, "source: how strongly ctx.source steers structure (0 ignores it)"),
    "hue_from_source": (False, "source: take hue families from the source's dominant colours"),
}


def with_groups(params: dict, colour=True, surface=True, source=True) -> dict:
    out = dict(params)
    for on, grp in ((colour, COLOUR_PARAMS), (surface, SURFACE_PARAMS), (source, SOURCE_PARAMS)):
        if on:
            for k, v in grp.items():
                out.setdefault(k, v)
    return out


# ------------------------------------------------------------------ hashing / noise

def hash01(a, b=0, c=0, seed: int = 0) -> np.ndarray:
    """Integer lattice hash -> float32 in 0..1 (broadcasts, deterministic)."""
    x = (np.asarray(a).astype(np.int64) * 73856093) ^ (np.asarray(b).astype(np.int64) * 19349663) \
        ^ (np.asarray(c).astype(np.int64) * 83492791) ^ ((seed + 1) * 2654435761)
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x45D9F3B) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x45D9F3B) & 0xFFFFFFFF
    x ^= x >> 16
    return ((x & 0xFFFFFF) / float(0x1000000)).astype(np.float32)


def fbm(shape, scale: float, rng, octaves: int = 3, gain: float = 0.5) -> np.ndarray:
    """Smooth noise 0..1 of size `shape` = (H, W); `scale` = coarsest feature in px.
    Computed on a small grid and resized, so it is cheap at any resolution."""
    H, W = shape
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(2.0, scale / (2 ** o))
        gh, gw = max(2, int(H / s) + 2), max(2, int(W / s) + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= gain
    out /= tot
    lo, hi = np.percentile(out[::3, ::3], [1, 99])
    return np.clip((out - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ colour

def hue_of(rgb) -> float:
    return pal.rgb_to_oklch(rgb)[2]


def hue_list(P, n: int, spread: float, src_hues=None) -> list[float]:
    """Hue families for an image. Default: structure hue, second/accent hues of the palette."""
    base = [hue_of(P.structure)]
    if P.structure2 is not None:
        base.append(hue_of(P.structure2))
    base.append(hue_of(P.accent))
    if src_hues:
        base = list(src_hues) + base
    out = []
    for h in base:
        if all(min(abs(h - o), 360 - abs(h - o)) > 12 for o in out):
            out.append(h)
    # make sure we always have at least n distinct families (add a split complement)
    k = 1
    while len(out) < max(1, n):
        out.append((out[0] + (45 if k % 2 else -45) * ((k + 1) // 2 + 0.5)) % 360)
        k += 1
    out = out[:max(1, n)]
    if spread:
        out = [(h + spread * i) % 360 for i, h in enumerate(out)]
    return out


def role_colours(P, hue: float, chroma: float = 1.0, contrast: float = 1.0, vary: float = 0.0):
    """Role colours for one hue family on the palette's polarity.
    returns dict body (substrate plate), body_hi, trace, bright (coarse / hot)."""
    dark = P.polarity == "dark"
    c = float(chroma)
    k = float(contrast)

    def mk(L, C, dh=0.0):
        return pal.oklch_to_rgb(float(np.clip(L, 0.04, 0.97)), C * c, (hue + dh) % 360)

    v = vary
    if dark:
        return dict(body=mk(0.20 + 0.03 * v, 0.05), body_hi=mk(0.29 + 0.04 * v, 0.07),
                    trace=mk(0.48 + 0.09 * k + 0.05 * v, 0.105, 6), bright=mk(0.66 + 0.09 * k, 0.115, 14))
    return dict(body=mk(0.89 - 0.03 * v, 0.045), body_hi=mk(0.82 - 0.03 * v, 0.06),
                trace=mk(0.52 - 0.08 * k - 0.04 * v, 0.14, 6), bright=mk(0.36 - 0.08 * k, 0.14, 14))


def source_hues(src, n: int = 3) -> list[float]:
    """Dominant hues of a source (chroma-weighted histogram peaks)."""
    from . import features as F
    lab = F.to_oklab(cv2.resize(np.asarray(src, np.float32)[..., :3], (96, 64), interpolation=cv2.INTER_AREA))
    c = np.hypot(lab[..., 1], lab[..., 2]).ravel()
    h = (np.degrees(np.arctan2(lab[..., 2], lab[..., 1])).ravel()) % 360
    hist, _ = np.histogram(h, bins=24, range=(0, 360), weights=c ** 2)
    hist = ndi.gaussian_filter1d(hist, 1.0, mode="wrap")
    out = []
    for i in np.argsort(-hist):
        hv = (i + 0.5) * 15
        if hist[i] > 0 and all(min(abs(hv - o), 360 - abs(hv - o)) > 40 for o in out):
            out.append(hv)
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------ surface

def surface(img: np.ndarray, rng, *, grain: float = 0.3, wear: float = 0.2, glow: float = 0.2,
            dark: bool = True, hot: np.ndarray | None = None) -> np.ndarray:
    """Shared surface pass. `hot` = optional (H, W) emitter mask for the bloom
    (default: brightest 6% of luminance on dark, none on light)."""
    H, W = img.shape[:2]
    s = max(0.5, W / 1500.0)
    out = img.astype(np.float32)
    if glow > 0 and dark:
        if hot is None:
            lum = out @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            thr = np.percentile(lum[::4, ::4], 93)
            hot = np.clip((lum - thr) / max(1e-3, lum.max() - thr), 0, 1)
        g = sum(cv2.GaussianBlur(hot.astype(np.float32), (0, 0), sg * s) for sg in (2.5, 8, 22)) / 3.0
        out = 1 - (1 - out) * (1 - np.clip(g[..., None] * glow * 1.6 * out / np.maximum(out.max(), 1e-3), 0, 0.6))
    if wear > 0:
        n = rng.random((H, W)).astype(np.float32)
        dust = (n > 1 - 0.0016 * wear).astype(np.float32)
        dust = cv2.GaussianBlur(dust, (0, 0), 0.7 * s) * 3.0
        mott = fbm((H, W), 60 * s, rng, 3)
        out = out * (1 - 0.10 * wear * (mott - 0.5))[..., None]
        sign = 1.0 if dark else -1.0
        out = out + sign * dust[..., None] * 0.18 * wear
    if grain > 0:
        g = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.6)
        out = out + (g * grain * 0.045)[..., None]
    return np.clip(out, 0.015, 0.985).astype(np.float32)


def fit_source(ctx, size=None, crop="feature"):
    """ctx.source fitted to the frame (feature crop, native aspect) or None."""
    if ctx.source is None:
        return None
    from . import features as F
    W, H = size or ctx.size
    return F.fit_cover(np.asarray(ctx.source, np.float32)[..., :3], (W, H), crop, 0.0)
