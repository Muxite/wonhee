"""Image checks for the art library. The owner's rule (2026-11-03, and
DESIGN.md "Project art"): an image has NO focal point; detail is equal or
evenly spread across the whole frame, readable at card size."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

from .palette import rgb_to_oklch


def _lum(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class Report:
    coverage: float          # share of the 8x8 tiles carrying detail (want >= 0.90)
    center_bias: float       # detail in the central tiles / mean (want <= 1.6)
    focal_ratio: float       # strongest tile / mean (want <= 3.0)
    ink_span_x: float
    ink_span_y: float
    dominant_hue_share: float
    lightness_range: float   # p95 - p5 of OKLCH-ish lightness proxy
    ok: bool
    reasons: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def analyze(img: np.ndarray, tiles: int = 8) -> Report:
    """img: float RGB H x W x 3 in 0..1."""
    H, W, _ = img.shape
    L = ndimage.gaussian_filter(_lum(img), 1.0)
    gy, gx = np.gradient(L)
    mag = np.hypot(gx, gy)
    h, w = H // tiles, W // tiles
    e = np.array([[mag[r * h:(r + 1) * h, c * w:(c + 1) * w].mean() for c in range(tiles)] for r in range(tiles)])
    mean = float(e.mean()) + 1e-9
    med = float(np.median(e))
    coverage = float((e >= 0.25 * med).mean()) if med > 0 else 0.0
    a, b = tiles // 2 - 1, tiles // 2 + 1
    center = float(e[a:b, a:b].mean())
    center_bias = center / mean
    focal_ratio = float(e.max()) / mean
    # ink span: columns/rows whose luminance differs from the canvas (median) by > 0.05
    bg = np.median(L)
    ink = np.abs(L - bg) > 0.05
    span_x, span_y = float(ink.any(axis=0).mean()), float(ink.any(axis=1).mean())
    # dominant hue share among saturated pixels (OKLCH hue histogram, 12 bins, best 2 adjacent)
    small = img[:: max(1, H // 120), :: max(1, W // 180)].reshape(-1, 3)
    hs = []
    for px in small[:: max(1, len(small) // 4000)]:
        Lc, C, hh = rgb_to_oklch(px)
        if C > 0.04:
            hs.append(hh)
    if hs:
        hist, _ = np.histogram(hs, bins=12, range=(0, 360))
        share = float(max(hist[i] + hist[(i + 1) % 12] for i in range(12)) / len(hs))
    else:
        share = 0.0
    p5, p95 = np.percentile(L, [5, 95])
    reasons = []
    if coverage < 0.90:
        reasons.append(f"coverage {coverage:.2f} < 0.90 (empty regions)")
    if center_bias > 1.6:
        reasons.append(f"center_bias {center_bias:.2f} > 1.6 (detail piles up in the middle)")
    if focal_ratio > 3.0:
        reasons.append(f"focal_ratio {focal_ratio:.2f} > 3.0 (one dominant spot)")
    if p95 - p5 < 0.25:
        reasons.append(f"lightness range {p95 - p5:.2f} < 0.25 (flat at card size)")
    return Report(coverage, center_bias, focal_ratio, span_x, span_y, share, float(p95 - p5), not reasons, reasons)
