"""Lens and print effects for stills, as numpy/cv2 operations: fisheye, soft edges, motion blur, bloom, vignette,
grain, halftone and two-ink screen print. Each takes and returns a BGR uint8 image; `amount` 0 = untouched.

These are the traits the K-pop night references share (fisheye, smeared edges, glowing practicals) plus the
graphic-print formats (halftone, duotone) that work as joke templates. Masks (wonhee.edit.mask) keep a subject
sharp while the background moves.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np


def _radius(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2)) / np.sqrt(2)  # 0 centre .. 1 corner


def fisheye(im: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Barrel distortion that still fills the frame (the centre bulges toward the lens, straight lines bow)."""
    if amount <= 0:
        return im
    h, w = im.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    r2 = nx * nx + ny * ny
    k = 0.35 * amount
    scale = (1 + k * r2) / (1 + 2 * k)  # corners map to corners, so nothing outside the frame is sampled
    return cv2.remap(im, (nx * scale * w / 2 + w / 2).astype(np.float32), (ny * scale * h / 2 + h / 2).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def edge_blur(im: np.ndarray, amount: float = 0.5, keep: np.ndarray | None = None) -> np.ndarray:
    """Sharp centre, smeared edges (cheap wide lens / handheld at a slow shutter). `keep` = mask kept sharp."""
    if amount <= 0:
        return im
    h, w = im.shape[:2]
    sigma = max(1.0, amount * max(h, w) / 120)
    blur = cv2.GaussianBlur(im, (0, 0), sigma)
    wgt = np.clip((_radius(h, w) - 0.35) / 0.55, 0, 1) ** 1.5
    if keep is not None:
        wgt = wgt * (1 - keep.astype(np.float32))
    wgt = wgt[..., None]
    return (im * (1 - wgt) + blur * wgt).astype(np.uint8)


def motion_blur(im: np.ndarray, amount: float = 0.5, angle: float = 0.0, keep: np.ndarray | None = None) -> np.ndarray:
    """Directional smear (a train passing, a pan). With `keep`, only the background moves."""
    if amount <= 0:
        return im
    n = max(3, int(amount * max(im.shape[:2]) / 25) | 1)
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1
    rot = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, rot, (n, n))
    k /= k.sum()
    out = cv2.filter2D(im, -1, k)
    if keep is None:
        return out
    m = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 3)[..., None]
    return (out * (1 - m) + im * m).astype(np.uint8)


def bloom(im: np.ndarray, amount: float = 0.5, threshold: float = 0.8) -> np.ndarray:
    """Glow around the brightest lights (neon tubes, vending machines, monitors), screen-blended."""
    if amount <= 0:
        return im
    f = im.astype(np.float32) / 255
    lum = f.max(axis=2, keepdims=True)
    hot = f * np.clip((lum - threshold) / (1 - threshold + 1e-6), 0, 1)
    glow = cv2.GaussianBlur(hot, (0, 0), max(im.shape[:2]) / 60) * (1.6 * amount)
    return np.clip((1 - (1 - f) * (1 - glow)) * 255 + 0.5, 0, 255).astype(np.uint8)


def vignette(im: np.ndarray, amount: float = 0.5) -> np.ndarray:
    if amount <= 0:
        return im
    r = _radius(*im.shape[:2])
    g = 1 - amount * np.clip((r - 0.3) / 0.7, 0, 1) ** 2
    return (im * g[..., None]).astype(np.uint8)


def grain(im: np.ndarray, amount: float = 0.3, seed: int = 0) -> np.ndarray:
    """Luma grain: strongest in the midtones, weak in deep shadows and clipped highlights, and sized with the
    image (~0.6 px blur at 1080). A fixed `seed` re-renders identically; seed 0 derives one from the picture, so
    different photos do not share a noise pattern."""
    if amount <= 0:
        return im
    if not seed:
        thumb = cv2.resize(im, (16, 16), interpolation=cv2.INTER_AREA)
        seed = zlib.crc32(thumb.tobytes()) or 1
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 18 * amount, im.shape[:2]).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), max(0.4, 0.6 * max(im.shape[:2]) / 1080))
    y = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 if im.ndim == 3 else im.astype(np.float32) / 255
    wgt = 0.1 + 0.9 * np.clip(4 * y * (1 - y), 0, 1) ** 0.8
    n = (n * wgt)[..., None] if im.ndim == 3 else n * wgt
    return np.clip(im.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _grey(im: np.ndarray) -> np.ndarray:
    """0..1 luminance stretched so the darkest 2% are black and the brightest 2% white (prints need full range)."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = np.percentile(g, (2, 98))
    return np.clip((g - lo) / max(hi - lo, 1.0), 0, 1)


def _hex(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], np.float32)  # BGR


def duotone(im: np.ndarray, ink: str = "#1a2fb0", paper: str = "#f3ead8", amount: float = 1.0,
            texture: float = 0.5, seed: int = 0) -> np.ndarray:
    """Two-ink screen print: shadows in `ink`, lights in `paper`, with paper grain and uneven ink density."""
    g = _grey(im) ** 1.5  # pull the mids into the ink, as a screen print does
    if texture > 0:
        rng = np.random.default_rng(seed)
        fib = cv2.GaussianBlur(rng.normal(0, 1, g.shape).astype(np.float32), (0, 0), 1.2)
        blot = cv2.GaussianBlur(rng.normal(0, 1, g.shape).astype(np.float32), (0, 0), max(g.shape) / 40)
        g = np.clip(g + texture * (0.06 * fib + 0.5 * blot / (blot.std() + 1e-6) * 0.08), 0, 1)
    out = _hex(ink) * (1 - g[..., None]) + _hex(paper) * g[..., None]
    return np.clip(im * (1 - amount) + out * amount, 0, 255).astype(np.uint8)


def halftone(im: np.ndarray, cell: int = 0, ink: str = "#e8141c", paper: str = "#ffffff", angle: float = 45,
             amount: float = 1.0) -> np.ndarray:
    """Dot-screen print in one ink: dark areas get big dots. `cell` 0 = sized from the image."""
    h, w = im.shape[:2]
    cell = cell or max(5, round(max(h, w) / 110))
    g = _grey(im)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    t = np.deg2rad(angle)
    u, v = xx * np.cos(t) + yy * np.sin(t), -xx * np.sin(t) + yy * np.cos(t)
    du = (u % cell) / cell - 0.5
    dv = (v % cell) / cell - 0.5
    d = np.hypot(du, dv)
    local = cv2.blur(g, (cell, cell))
    radius = np.sqrt(np.clip(1 - local, 0, 1)) * 0.62  # dot area tracks darkness
    dot = np.clip((radius - d) * cell, 0, 1)  # soft-edged dot
    out = _hex(paper) * (1 - dot[..., None]) + _hex(ink) * dot[..., None]
    return np.clip(im * (1 - amount) + out * amount, 0, 255).astype(np.uint8)
