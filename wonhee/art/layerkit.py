"""Layer / exploded-view kit: planes (RGB + alpha) projected in an oblique, isometric or
perspective view with spacing, thickness (extruded walls), edge lines and soft shadows.

Splitting helpers produce the planes: by tone, colour region, depth proxy, RGB/Lab channel, frequency band
or a user mask. NOTE: depth inferred from an ordinary 2-D image is an approximation (vertical position,
focus, haze and tone heuristics), not measured geometry; use ctx.data to supply a real depth map / height
field when you have one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import features as F
from . import regionkit as R


@dataclass
class Plane:
    rgb: np.ndarray                  # (H, W, 3)
    alpha: np.ndarray                # (H, W) 0..1
    edge: np.ndarray | None = None   # (H, W) 0..1 outline in plane space
    wall: np.ndarray | None = None   # colour for the extruded side (3,), default rgb darkened
    label: str = ""


# ------------------------------------------------------------------ projection

def view_matrix(W: int, H: int, view: str = "iso", zoom: float = 1.0, spin: float = 45.0, squash: float = 0.5,
                persp: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """3x3 homography taking plane coords (0..W, 0..H) to screen. view: iso | oblique | top.
    iso = rotate by spin then squash vertically (rhombus); oblique = shear (cabinet projection);
    top = a plain scale with a vertical squash (a table seen from a low angle)."""
    c = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]], np.float64)
    if view == "oblique":
        A = np.array([[1, -0.55, 0], [0, squash * 1.5, 0], [0, 0, 1]], np.float64)
    elif view == "top":
        A = np.array([[1, 0, 0], [0, squash * 1.6, 0], [0, 0, 1]], np.float64)
    else:
        a = math.radians(spin)
        Rm = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], np.float64)
        A = np.array([[1, 0, 0], [0, squash, 0], [0, 0, 1]], np.float64) @ Rm
    P = np.array([[1, 0, 0], [0, 1, 0], [persp / W, persp * 1.2 / H, 1]], np.float64)
    S = np.array([[zoom, 0, 0], [0, zoom, 0], [0, 0, 1]], np.float64)
    t = np.array([[1, 0, W / 2 + shift[0]], [0, 1, H / 2 + shift[1]], [0, 0, 1]], np.float64)
    return t @ S @ P @ A @ c


def _shift_down(a: np.ndarray, dy: int) -> np.ndarray:
    if dy <= 0:
        return a
    out = np.zeros_like(a)
    out[dy:] = a[:-dy]
    return out


def project(planes: list[Plane], size, M: np.ndarray, *, spacing: float, thick: float = 0.0, shadow: float = 0.5,
            shadow_dir: tuple[float, float] = (0.35, 0.55), shadow_blur: float = 14.0, edge_color=None, wall_dark: float = 0.5,
            bg: np.ndarray | None = None, bg_color=(0.1, 0.1, 0.1), dark: bool = True, s: float = 1.0, lift: float = 0.0) -> np.ndarray:
    """Composite planes bottom -> top. spacing = screen px between consecutive planes (vertical);
    thick = px of extruded wall under each plane; edge_color = outline colour (None = no outline).
    lift: shadow offset grows with the gap to the plane below."""
    W, H = size
    out = np.ones((H, W, 3), np.float32) * np.asarray(bg_color, np.float32) if bg is None else bg.copy()
    n = len(planes)
    for k, pl in enumerate(planes):
        T = np.array([[1, 0, 0], [0, 1, -k * spacing + (n - 1) * spacing * 0.5], [0, 0, 1]], np.float64) @ M
        pm = np.concatenate([pl.rgb * pl.alpha[..., None], pl.alpha[..., None]], -1).astype(np.float32)
        wp = cv2.warpPerspective(pm, T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        A = np.clip(wp[..., 3], 0, 1)
        C = wp[..., :3] / np.maximum(A[..., None], 1e-4)
        # soft shadow on everything below
        if shadow > 0 and k > 0:
            gap = max(spacing, 1.0) * (0.6 + lift)
            sh = R.drop_shadow(A, (shadow_dir[0] * gap, shadow_dir[1] * gap), shadow_blur * s)
            out = out * (1 - shadow * sh[..., None] * (1 - 0.4 * (0 if dark else 1)))
        # extruded walls: vertical running-max of the plane (rows y-T..y), drawn under the plane
        if thick > 0:
            T_ = max(1, int(thick))
            ker = np.ones((T_ + 1, 1), np.uint8)
            Aw = cv2.dilate(A, ker, anchor=(0, T_))
            if pl.wall is not None:
                wcol = np.asarray(pl.wall, np.float32)[None, None, :]
            else:
                Cw = cv2.dilate(C * A[..., None], ker, anchor=(0, T_)) / np.maximum(Aw[..., None], 1e-4)
                wcol = Cw * wall_dark
            out = out * (1 - Aw[..., None]) + wcol * Aw[..., None]
        out = out * (1 - A[..., None]) + C * A[..., None]
        if pl.edge is not None and edge_color is not None:
            we = cv2.warpPerspective(pl.edge.astype(np.float32), T, (W, H), flags=cv2.INTER_LINEAR)
            e = np.clip(we, 0, 1)[..., None]
            out = out * (1 - e) + np.asarray(edge_color, np.float32) * e
    return out


# ------------------------------------------------------------------ plane building

def plane_edge(alpha: np.ndarray, width: float) -> np.ndarray:
    """Outline of an alpha shape in plane space (anti-aliased line along its boundary)."""
    if width <= 0:
        return np.zeros_like(alpha)
    a = ndi.gaussian_filter(alpha.astype(np.float32), max(0.6, width * 0.35))
    g = np.abs(a - 0.5)
    return np.clip(1 - g / max(0.18, width * 0.12), 0, 1).astype(np.float32) * (alpha.max() > 0)


def depth_proxy(img: np.ndarray, w_pos: float = 1.0, w_focus: float = 0.6, w_haze: float = 0.5, w_tone: float = 0.0) -> np.ndarray:
    """Heuristic 0..1 'nearness' map from an ordinary image (APPROXIMATE, not geometry): lower in the frame =
    nearer, sharper = nearer, hazier / lower contrast / bluer = farther, optional brighter = nearer."""
    H, W = img.shape[:2]
    L = F.luminance(img)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    sharp = F.blur(F.edges_multiscale(L, (1.0, 2.0)), 14 * F.scale_of((H, W)))
    sharp = sharp / (sharp.max() + 1e-9)
    lab = F.to_oklab(img)
    haze = F.blur(np.hypot(lab[..., 1], lab[..., 2]), 16 * F.scale_of((H, W)))
    haze = haze / (haze.max() + 1e-9)
    tone = F.blur(L, 12 * F.scale_of((H, W)))
    d = w_pos * yy + w_focus * sharp + w_haze * haze + w_tone * tone
    d = F.blur(d, 6 * F.scale_of((H, W)))
    return F.stretch(d, 1, 99)


def quantile_planes(field_: np.ndarray, k: int, soft: float) -> list[np.ndarray]:
    """k cumulative masks from a scalar field: plane 0 everywhere, plane i where field >= i-th quantile."""
    qs = np.percentile(field_, np.linspace(0, 100, k + 1)[1:-1])
    masks = [np.ones_like(field_, np.float32)]
    for q in qs:
        m = (field_ >= q).astype(np.float32)
        masks.append(ndi.gaussian_filter(m, soft) if soft > 0.3 else m)
    return masks


def simplify_mask(m: np.ndarray, smooth: float, s: float, thresh: float = 0.5, min_area: float = 0.0025) -> np.ndarray:
    """Round off a mask: blur then threshold, drop islands / holes smaller than min_area of the frame (anti-aliased)."""
    if smooth <= 0:
        return m
    b = ndi.gaussian_filter(m.astype(np.float32), smooth * 14 * s + 0.5)
    hard = b > thresh
    thr = max(4, int(min_area * hard.size * (0.4 + smooth)))
    for val in (True, False):
        cc, k = ndi.label(hard == val)
        if k:
            sizes = np.bincount(cc.ravel())
            small = sizes < thr
            small[0] = False
            hard = np.where(small[cc], not val, hard)
    return np.clip(ndi.gaussian_filter(hard.astype(np.float32), 0.9 * s + 0.4) * 1.6 - 0.3, 0, 1).astype(np.float32)
