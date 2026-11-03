"""Shared toolkit for the ``nature_*``, ``motion_*`` and ``contour_*`` effects
(no effect is registered here).

Design: every effect separates four independently adjustable layers.

* STRUCTURE  scalar fields (anisotropic multiscale noise, flow maps, strata
  coordinates, contour levels). Computed at a *working size* (<= ~640 px wide)
  and upsampled, so a 1500 px render stays well under the 8 s budget.
* SOURCE     ``source_fields``: a photo's luminance / orientation / colour as
  the flow or material field (``use_source`` 0..1 blends it with procedural).
* COLOUR     ``ramp`` / ``paint``: a scalar 0..1 becomes colour through an
  OKLCH ramp built from the palette roles (lightness carries contrast; hue
  drifts between the structure and accent hues; ``tint`` moves the highlight).
* SURFACE    ``surface``: grain, haze / bloom, paper tooth. Spatial, not a LUT.

Everything is deterministic in (seed, size, params) and coordinate based, so
the same look can later run per frame on video (``tnoise`` is a seeded
space-time value noise).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import features as F
from . import palette as pal

# ---------------------------------------------------------------- shared params
COLOUR_PARAMS = {
    "tint": (0.0, "COLOUR: degrees the highlight hue is rotated away from the palette accent (0 = palette)"),
    "contrast": (1.0, "COLOUR: tone contrast 0.5-2 (lightness range stays inside the palette-safe band)"),
    "ramp": ("mono", "COLOUR: mono (one hue family) | dual (structure -> accent hue drift) | spectrum (broad hue sweep, controlled lightness)"),
    "invert": (False, "COLOUR: swap light-on-dark / dark-on-light inside the palette polarity"),
}
SURFACE_PARAMS = {
    "grain": (0.35, "SURFACE: fine grain 0..1"),
    "haze": (0.0, "SURFACE: 0..1 soft veil / bloom of the highlights (lifts darks slightly toward canvas)"),
}
SOURCE_PARAMS = {
    "use_source": (0.0, "SOURCE: 0..1 how much the source photo's luminance steers the field (0 = procedural only)"),
    "source_mode": ("luminance", "SOURCE: luminance | orientation | both (flow follows the photo's own structure)"),
    "crop": ("feature", "SOURCE: how a source of another aspect is cropped: feature | center"),
}


def with_common(params: dict, colour=True, surface=True, source=True) -> dict:
    out = dict(params)
    for on, grp in ((colour, COLOUR_PARAMS), (surface, SURFACE_PARAMS), (source, SOURCE_PARAMS)):
        if on:
            for k, v in grp.items():
                out.setdefault(k, v)
    return out


# ---------------------------------------------------------------- sizes
def sc(ctx) -> float:
    """Pixel scale relative to a 1500 px wide frame."""
    return ctx.size[0] / 1500.0


def work_size(ctx, max_w: int = 640) -> tuple[int, int]:
    W, H = ctx.size
    w = min(W, max_w)
    return w, max(8, int(round(w * H / W)))


def up(a: np.ndarray, size: tuple[int, int], cubic: bool = True) -> np.ndarray:
    W, H = size
    if a.shape[1] == W and a.shape[0] == H:
        return a
    return cv2.resize(np.ascontiguousarray(a, np.float32), (W, H),
                      interpolation=cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def pnorm(f, lo=2.0, hi=98.0):
    a, b = np.percentile(f, [lo, hi])
    return np.clip((f - a) / (b - a + 1e-9), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- noise
def noise(rng, h, w, sigma, sigma2=None, angle=0.0, normalise=True):
    """Gaussian-filtered white noise; sigma along x, sigma2 along y (an
    anisotropic blob), rotated by ``angle`` radians. Unit variance. Large
    features are generated on a reduced grid and upsampled (same statistics,
    much faster)."""
    sigma2 = sigma if sigma2 is None else sigma2
    f = int(max(1, min(sigma, sigma2) // 3))
    if f > 1:
        n = noise(rng, max(4, int(math.ceil(h / f))), max(4, int(math.ceil(w / f))), sigma / f, sigma2 / f, angle, False)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    elif abs(angle) < 1e-6 or abs(sigma - sigma2) < 1e-6:
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        n = ndi.gaussian_filter(n, (max(sigma2, 0.01), max(sigma, 0.01)), mode="wrap")
    else:
        d = int(math.hypot(h, w) * 1.05) + 8
        n = rng.normal(0, 1, (d, d)).astype(np.float32)
        n = ndi.gaussian_filter(n, (max(sigma2, 0.01), max(sigma, 0.01)), mode="wrap")
        M = cv2.getRotationMatrix2D((d / 2, d / 2), -math.degrees(angle), 1.0)
        n = cv2.warpAffine(n, M, (d, d), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        y0, x0 = (d - h) // 2, (d - w) // 2
        n = n[y0:y0 + h, x0:x0 + w]
    if normalise:
        n = n / (n.std() + 1e-9)
    return np.ascontiguousarray(n, np.float32)


def fbm(rng, h, w, sigma, octaves=4, gain=0.55, aniso=1.0, angle=0.0, lacunarity=2.0):
    """Multi-octave anisotropic noise. ``sigma`` = largest feature size (px, along
    the angle); ``aniso`` > 1 stretches features along ``angle`` (streaks/layers)."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    s = float(sigma)
    for _ in range(int(octaves)):
        out += amp * noise(rng, h, w, max(s, 0.6), max(s / aniso, 0.6), angle)
        tot += amp * amp
        amp *= gain
        s /= lacunarity
    return out / math.sqrt(tot)


def _hash(ix, iy, it, seed):
    n = (ix * 374761393 + iy * 668265263 + it * 2147483647 + seed * 362437) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    n = n ^ (n >> 16)
    return (n & 0xFFFFFF).astype(np.float64) / 0xFFFFFF


def tnoise(x, y, t, seed=0, period=1.0):
    """Seeded space-time value noise in [-1, 1]: a pure function of (x, y, t),
    so frame k of a video equals frame k of any other run (temporal
    coherence). x, y in any unit; ``period`` is the lattice size in that unit,
    t in frames/seconds with a lattice of 1."""
    x = np.asarray(x, np.float64) / period
    y = np.asarray(y, np.float64) / period
    t = np.broadcast_to(np.asarray(t, np.float64), np.broadcast(x, y).shape)
    x0, y0, t0 = np.floor(x), np.floor(y), np.floor(t)
    fx, fy, ft = x - x0, y - y0, t - t0
    sx, sy, st = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), ft * ft * (3 - 2 * ft)
    ix, iy, it = x0.astype(np.int64), y0.astype(np.int64), t0.astype(np.int64)
    acc = 0.0
    for dt, wt in ((0, 1 - st), (1, st)):
        for dy, wy in ((0, 1 - sy), (1, sy)):
            for dx, wx in ((0, 1 - sx), (1, sx)):
                acc = acc + wt * wy * wx * _hash(ix + dx, iy + dy, it + dt, seed)
    return (acc * 2 - 1).astype(np.float32)


# ---------------------------------------------------------------- flow
def curl(psi: np.ndarray):
    """Divergence-free velocity from a stream function (vx, vy)."""
    gy, gx = np.gradient(psi)
    return gy.astype(np.float32), (-gx).astype(np.float32)


def trace_map(vx, vy, steps, dt, size=None, mask=None):
    """Backward semi-Lagrangian flow map: for every pixel the position it came
    from after ``steps`` of ``dt`` along (-v). Sampling an initial field c0 at
    (px, py) gives the field advected for that time (tracers, ribbons)."""
    h, w = vx.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    px, py = gx.copy(), gy.copy()
    for _ in range(int(steps)):
        ux = cv2.remap(vx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        uy = cv2.remap(vy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if mask is not None:
            m = cv2.remap(mask, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            ux, uy = ux * m, uy * m
        px = px - ux * dt
        py = py - uy * dt
    return px, py


def sample(a, px, py, border=cv2.BORDER_REFLECT):
    return cv2.remap(np.ascontiguousarray(a, np.float32), px.astype(np.float32), py.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=border)


# ---------------------------------------------------------------- source
def source_fields(ctx, size, crop=None, blur=0.0):
    """(img, L, theta, coh) of the source fitted to ``size``; None-safe.
    theta is the direction ALONG structure (radians)."""
    if ctx.source is None:
        return None
    W, H = size
    img = F.fit_cover(np.asarray(ctx.source, np.float32)[..., :3], (W, H), crop or ctx.p("crop", "feature"))
    L = F.luminance(img)
    L = F.stretch(F.local_equalize(L, 2.0, mix=0.5), 1, 99)
    if blur > 0:
        L = F.blur(L, blur)
    theta, coh = F.structure_orientation(F.blur(L, 0.8 * F.scale_of((H, W))), sigma=1.5, rho=float(ctx.p("rho", 10.0)), seed=ctx.seed)
    return img, L, theta, coh


# ---------------------------------------------------------------- colour
def _lch(rgb):
    return pal.rgb_to_oklch(rgb)


def _hue_lerp(h0, h1, t, lim=None):
    d = ((h1 - h0 + 180) % 360) - 180
    if lim is not None:
        d = float(np.clip(d, -lim, lim))
    return h0 + d * t


def ramp(p, n=512, mode="mono", tint=0.0, contrast=1.0, invert=False, hi=None, lo=None):
    """t in 0..1 (0 = background, 1 = strongest ink / highlight) -> RGB (n, 3).

    Dark polarity: from the palette canvas up through the structure colour to a
    pale, low-chroma tinted highlight. Light polarity: from paper down through
    the structure colour to a dark ink. ``mode``: mono keeps one hue family;
    dual drifts structure hue -> accent hue; spectrum sweeps about 140 degrees
    (several hue families, lightness still monotone so it never muddies)."""
    dark = (p.polarity == "dark") != bool(invert)
    cL, cC, cH = _lch(p.canvas)
    sL, sC, sH = _lch(p.structure)
    aL, aC, aH = _lch(p.accent)
    aH = (aH + tint) % 360
    t = np.linspace(0, 1, n, dtype=np.float32)
    tc = np.clip((t - 0.5) * contrast + 0.5, 0, 1)
    if p.polarity == "dark" and not invert:
        L0, L1 = (cL if lo is None else lo), (0.93 if hi is None else hi)
        Lm = max(sL, L0 + 0.25)
        Ls = np.where(tc < 0.55, L0 + (Lm - L0) * (np.clip(tc, 0, 0.55) / 0.55) ** 1.1, Lm + (L1 - Lm) * (np.clip(tc - 0.55, 0, 0.45) / 0.45) ** 1.3)
    elif p.polarity == "dark":      # inverted: pale paper from a dark-polarity palette
        L0, L1 = 0.95, 0.30
        Ls = L0 + (L1 - L0) * tc ** 1.1
    elif not invert:                # light palette
        L0, L1 = (cL if lo is None else lo), (0.28 if hi is None else hi)
        Ls = L0 + (L1 - L0) * tc ** 1.1
    else:                           # light palette inverted: dark ground, pale ink
        L0, L1 = 0.17, 0.93
        Ls = L0 + (L1 - L0) * tc ** 1.2
    if mode == "spectrum":
        H = sH + (tc - 0.2) * 150.0
        Ch = (0.04 + 0.12 * np.sin(np.pi * np.clip(tc * 1.05, 0, 1)) + 0.03)
    elif mode == "dual":
        H = _hue_lerp(sH, aH, smooth(0.35, 1.0, tc), 75.0)
        Ch = np.maximum(sC * 0.25, sC * np.sin(np.pi * np.clip(tc, 0, 1)) * 0.9 + 0.02) * (1 - 0.45 * smooth(0.8, 1.0, tc))
    else:
        H = _hue_lerp(sH, aH, smooth(0.4, 1.0, tc), 28.0)
        Ch = np.maximum(sC * 0.2, sC * np.sin(np.pi * np.clip(tc * 0.98, 0, 1)) * 0.85 + 0.015) * (1 - 0.5 * smooth(0.82, 1.0, tc))
    Ch = np.minimum(Ch, max(sC, 0.12) * 1.1)
    return F.oklch_field(Ls, Ch, H).astype(np.float32)


def paint(t, R):
    """Look up scalar t (any shape) in a ramp R (n, 3)."""
    n = len(R)
    x = np.clip(t, 0, 1) * (n - 1)
    i = np.minimum(x.astype(np.int32), n - 2)
    f = (x - i)[..., None].astype(np.float32)
    return R[i] * (1 - f) + R[i + 1] * f


def tone_curve(t, contrast=1.0, gamma=1.0):
    t = np.clip(t, 0, 1) ** gamma
    return np.clip((t - 0.5) * contrast + 0.5, 0, 1)


# ---------------------------------------------------------------- surface
def surface(rgb, ctx, grain=None, haze=None, seed_off=0):
    """Spatial surface treatment. grain: gently blurred luminance noise scaled
    to the frame; haze: a veil of the blurred highlights screened back and a
    slight lift of the darks (lens / atmosphere / print)."""
    grain = float(ctx.p("grain", 0.35)) if grain is None else grain
    haze = float(ctx.p("haze", 0.0)) if haze is None else haze
    H, W = rgb.shape[:2]
    s = max(W / 1500.0, 0.2)
    out = rgb
    if haze > 0:
        lum = out @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        br = np.clip((lum - 0.45) / 0.55, 0, 1)[..., None] * out
        veil = cv2.GaussianBlur(br, (0, 0), 14 * s) * 0.6 + cv2.GaussianBlur(br, (0, 0), 40 * s) * 0.4
        out = 1 - (1 - out) * (1 - np.clip(veil * haze * 1.6, 0, 1))
        out = out * (1 - 0.06 * haze) + 0.06 * haze * np.asarray(ctx.palette.canvas, np.float32) * 0 + 0.03 * haze
    if grain > 0:
        rng = np.random.default_rng(ctx.seed + 9173 + seed_off)
        g = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), max(0.55, 0.7 * min(1.0, s * 1.6)))
        out = out + (g * grain * 0.045)[..., None]
    return np.clip(out, 0.02, 0.985).astype(np.float32)


def smear_along(img, vx, vy, steps, dt, taper=True, keep_sign=True):
    """Line-integral smear of an image (H, W[, C]) along the field (vx, vy),
    both directions, triangular weights (LIC with an image instead of noise)."""
    img = np.ascontiguousarray(img, np.float32)
    h, w = vx.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    acc = img.copy()
    wsum = 1.0
    for sign in (1.0, -1.0):
        px, py = gx.copy(), gy.copy()
        pvx, pvy = sign * vx, sign * vy
        for k in range(int(steps)):
            cx = cv2.remap(vx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            cy = cv2.remap(vy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if keep_sign:
                sg = np.where(cx * pvx + cy * pvy < 0, -1.0, 1.0).astype(np.float32)
                cx, cy = cx * sg, cy * sg
            else:
                cx, cy = cx * sign, cy * sign
            px = px + cx * dt
            py = py + cy * dt
            pvx, pvy = cx, cy
            wt = (1.0 - k / steps) if taper else 1.0
            acc += wt * cv2.remap(img, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum += wt
    return acc / wsum
