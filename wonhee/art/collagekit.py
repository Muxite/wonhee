"""Collage kit: cut-out alpha shapes, paper edges, hard / soft shadows, colour unification and
sub-renders of other library effects as generated surfaces."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import regionkit as R


def surface(name: str, size, seed: int, palette, params: dict | None = None, source=None) -> np.ndarray:
    """Render another library effect as a generated surface (float RGB). Unknown / failing effects give a flat canvas."""
    from .compose import render_effect
    try:
        return render_effect(name, size=size, seed=seed, palette=palette, params=params or {}, source=source)
    except Exception:  # an effect that needs data it was not given
        return np.ones((size[1], size[0], 3), np.float32) * np.asarray(palette.canvas if hasattr(palette, "canvas") else (0.2, 0.2, 0.2), np.float32)


def unify(img: np.ndarray, hue: float | None, tint: float, dark: bool, s: float, chroma: float = 0.9, spread: float = 0.0) -> np.ndarray:
    """Push a piece toward one hue family (tint 0..1) keeping its light and texture."""
    if tint <= 0 or hue is None:
        return img
    one = R.RegionSet(np.zeros(img.shape[:2], np.int32), 1, 0.0)
    st = R.RegionStyle(hue=hue, hue_keep=1.0 - tint, chroma=chroma, c_abs=0.05 * tint)
    return R.restyle(img, one, [st], s_low=30 * s, s_tex=6 * s, dark=dark)


def torn_alpha(a: np.ndarray, torn: float, s: float, seed: int) -> np.ndarray:
    """Rough 'torn paper' edge: noise-displaced threshold of a blurred mask. torn 0 = clean."""
    if torn <= 0:
        return a
    rng = np.random.default_rng(seed)
    n1 = ndi.gaussian_filter(rng.normal(0, 1, a.shape).astype(np.float32), 2.5 * s)
    n2 = ndi.gaussian_filter(rng.normal(0, 1, a.shape).astype(np.float32), 0.9 * s)
    n = n1 / (n1.std() + 1e-9) * 0.65 + n2 / (n2.std() + 1e-9) * 0.35
    b = ndi.gaussian_filter(a, 3.5 * s)
    return np.clip((b - 0.5 + n * 0.09 * torn) * 7.0 + 0.5, 0, 1).astype(np.float32)


def grow(a: np.ndarray, px: float) -> np.ndarray:
    if px <= 0.3:
        return a
    d = cv2.dilate((a > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(px) * 2 + 1,) * 2))
    return np.maximum(a, ndi.gaussian_filter(d.astype(np.float32), 0.8))


def over(base: np.ndarray, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return base * (1 - a) + rgb * a


def rim_light(alpha: np.ndarray, px: float) -> np.ndarray:
    """Thin bright rim just inside an alpha edge (paper thickness catching light)."""
    if px <= 0:
        return np.zeros_like(alpha)
    er = ndi.gaussian_filter(cv2.erode((alpha > 0.5).astype(np.uint8), np.ones((int(px) * 2 + 1,) * 2, np.uint8)).astype(np.float32), 0.7)
    return np.clip(alpha - er, 0, 1)
