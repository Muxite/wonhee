"""Ink-and-print material system (families ``print-*``; also used by ``swiss-*``).

A print is built like a press run, in separate, independently adjustable parts:

* SEPARATION  an image (photo, data, generated shapes) becomes 1-4 ink coverage
              fields by tone, hue, colour channels or explicit masks.
* SCREEN      each coverage becomes marks: flat shapes (``none``), a dot screen
              (frequency + angle per ink), a line screen, or a checker-free FM grain.
* SURFACE     paper tone/mottle/fibre, ink grain (starved on fibre valleys),
              registration offsets, edge wear. ``texture=0`` is the clean variant.
* OVERPRINT   inks are transparent filters (multiply) on paper, or opaque (over)
              on dark stock.

Everything is deterministic in the rng and scale-aware: sizes are authored for a
1500 px wide master and scaled by ``size[0] / 1500`` but clamped to floors
(``PX_FLOOR``) so dots, fibres and registration still read at 288x192, while big
light/dark shapes are left untouched.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from . import features as F
from . import palette as pal

# smallest sizes (output px) at which each device still reads
PX_FLOOR = {"pitch": 4.2, "reg": 1.4, "fibre": 6.0, "grain": 0.8}


def U(size) -> float:
    return size[0] / 1500.0


def sc(size, v: float, floor: float = 0.0) -> float:
    """Master-pixel length -> output pixels, never below ``floor``."""
    return max(v * U(size), floor)


# ---------------------------------------------------------------- colours

_INKS = {   # authored as printed colours (OKLCH L, C, h), rich not neon
    "ultramarine": (0.40, 0.19, 265), "cobalt": (0.48, 0.20, 262), "navy": (0.26, 0.09, 262),
    "cyan": (0.72, 0.13, 220), "sky": (0.78, 0.09, 235), "teal": (0.55, 0.10, 195),
    "green": (0.58, 0.14, 155), "yellow": (0.88, 0.16, 95), "gold": (0.80, 0.15, 85),
    "orange": (0.70, 0.17, 52), "red": (0.56, 0.20, 28), "vermilion": (0.62, 0.20, 36),
    "pink": (0.72, 0.19, 355), "magenta": (0.58, 0.22, 350), "violet": (0.48, 0.17, 300),
    "black": (0.22, 0.015, 270), "ink": (0.22, 0.03, 262),
    "cream": (0.95, 0.025, 90), "paper": (0.95, 0.02, 85), "bone": (0.91, 0.02, 80), "warm-grey": (0.82, 0.01, 75),
    "night": (0.17, 0.04, 265), "slate": (0.23, 0.03, 250),
}


def color(spec, palette: pal.Palette | None = None) -> np.ndarray:
    """'#hex' | 'ultramarine' ... | palette role ('structure') | (r,g,b) -> float rgb."""
    if isinstance(spec, (tuple, list, np.ndarray)):
        return np.asarray(spec, np.float32)[:3]
    s = str(spec).strip().lower()
    if s.startswith("#"):
        return pal.hex_to_rgb(s)
    if s in _INKS:
        return pal.oklch_to_rgb(*_INKS[s])
    if palette is not None and s in ("canvas", "structure", "structure2", "accent", "ink", "edge", "band"):
        v = getattr(palette, s, None)
        return np.asarray(v if v is not None else palette.structure, np.float32)
    raise KeyError(f"unknown colour {spec!r}")


def colors(spec, palette=None) -> list[np.ndarray]:
    if isinstance(spec, str):
        spec = [t for t in spec.replace(";", ",").split(",") if t.strip()]
    return [color(c, palette) for c in spec]


def lighten(c, t):
    return np.asarray(c, np.float32) * (1 - t) + t


# ---------------------------------------------------------------- noise

def noise(shape, rng, sigma: float) -> np.ndarray:
    """Zero-mean unit-ish gaussian-filtered noise."""
    n = rng.normal(0, 1, shape).astype(np.float32)
    if sigma > 0:
        n = F.blur(n, sigma)
    n -= n.mean()
    return n / (n.std() + 1e-6)


def fbm(shape, rng, base: float, octaves: int = 3) -> np.ndarray:
    out = np.zeros(shape, np.float32)
    a, s = 1.0, base
    for _ in range(octaves):
        out += a * noise(shape, rng, s)
        a *= 0.5
        s *= 0.5
    return out / (out.std() + 1e-6)


# ---------------------------------------------------------------- paper

def paper(size, rng, base, fibre: float = 0.6, mottle: float = 0.5, tooth: float = 0.5):
    """Paper tone with soft mottle, visible fibres and tooth. Returns (rgb HxWx3, height 0..1)
    where ``height`` (fibres + tooth only, no mottle) is the relief that later starves ink."""
    W, H = size
    base = np.asarray(base, np.float32)
    u = U(size)
    mot = np.zeros((H, W), np.float32)
    if mottle > 0:
        mot = 0.7 * noise((H, W), rng, max(14 * u, 7)) + 0.3 * noise((H, W), rng, max(4 * u, 2))
    tth = np.zeros((H, W), np.float32)
    if tooth > 0:
        tth = noise((H, W), rng, max(0.8, 0.8 * u + 0.4))
    fib = np.zeros((H, W), np.float32)
    if fibre > 0:
        n = int(min(fibre * W * H / (max(u, 0.3) ** 2 * 260), 9000))
        L = max(40 * u, PX_FLOOR["fibre"] * 1.3)
        cnv = np.zeros((H, W), np.float32)
        for sign in (1.0, -1.0):          # light and dark fibres
            layer = np.zeros((H, W), np.uint8)
            for _ in range(n // 2):
                x, y = rng.uniform(0, W), rng.uniform(0, H)
                a = rng.uniform(0, math.pi)
                ln = L * rng.uniform(0.4, 1.5)
                bend = rng.normal(0, 0.6)
                pts = [(x + math.cos(a + bend * t) * ln * t, y + math.sin(a + bend * t) * ln * t) for t in (0, 1 / 3, 2 / 3, 1)]
                cv2.polylines(layer, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, int(rng.integers(110, 255)), 1, cv2.LINE_AA)
            cnv += sign * layer.astype(np.float32) / 255
        fib = F.blur(cnv, 0.5)
        fib /= (fib.std() + 1e-6)
    shade = 1 + 0.032 * mottle * mot + 0.045 * fibre * fib + 0.018 * tooth * tth
    rgb = base[None, None, :] * shade[..., None]
    hh = F.smoothstep(-2.0, 2.0, (0.8 * fib * min(fibre * 2, 1) + 0.8 * tth * min(tooth * 2, 1)))
    return np.clip(rgb, 0, 1).astype(np.float32), hh


# ---------------------------------------------------------------- screens

def soft(cov, a: float = 0.5, w: float = 0.06):
    return F.smoothstep(a - w, a + w, cov)


_QCACHE: dict = {}


def _pattern_quantiles(kind: str, n: int = 513) -> np.ndarray:
    if kind not in _QCACHE:
        g = (np.arange(256) + 0.5) / 256
        u, v = np.meshgrid(g, g)
        if kind == "dot":
            pat = 0.5 - 0.25 * (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v))
        elif kind == "line":
            pat = 0.5 - 0.5 * np.cos(2 * np.pi * u)
        else:
            pat = 1 - np.maximum(np.abs(u - 0.5), np.abs(v - 0.5)) * 2
        _QCACHE[kind] = np.quantile(pat.ravel(), np.linspace(0, 1, n))
    return _QCACHE[kind]


def screen(cov: np.ndarray, size, *, kind: str = "dot", pitch: float = 14.0, angle: float = 45.0,
           gain: float = 1.0, rng=None, edge: float = 1.0) -> np.ndarray:
    """Coverage 0..1 -> printed mask 0..1.

    kind  none: flat shape (soft-thresholded; big shapes stay intact)
          dot : round AM screen, dots grow and merge (pitch master px, angle deg)
          line: line screen, width follows coverage
          sq  : square (checkerboard-free) pixel screen
          fm  : fine stochastic grain screen (no pattern, no moire)
    pitch is clamped to the small-size floor so the screen reads at 288x192."""
    H, W = cov.shape
    c = np.clip(cov, 0, 1) ** (1.0 / max(gain, 1e-3))
    if kind == "raw":
        return np.clip(cov, 0, 1).astype(np.float32)
    if kind == "none":
        return soft(c, 0.5, 0.5 / max(sc(size, 8, 1.5), 1.5) + 0.02).astype(np.float32)
    p = max(pitch * U(size), PX_FLOOR["pitch"])
    th = math.radians(angle)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xx * math.cos(th) + yy * math.sin(th)) / p
    v = (-xx * math.sin(th) + yy * math.cos(th)) / p
    if kind == "dot":
        pat = 0.5 - 0.25 * (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v))     # 0 at dot centre, 1 between dots
        w = 1.57 / p * edge * 1.1
    elif kind == "line":
        pat = 0.5 - 0.5 * np.cos(2 * np.pi * u)
        w = 3.14 / p * edge * 1.1
    elif kind == "sq":
        pat = 1 - np.maximum(np.abs(u % 1 - 0.5), np.abs(v % 1 - 0.5)) * 2     # 1 at cell centre, 0 at edge
        pat = 1 - pat                                                         # ink grows from the centre
        w = 2.0 / p * edge
    elif kind == "fm":
        r = rng if rng is not None else np.random.default_rng(0)
        pat = noise((H, W), r, max(0.7, 0.6 * U(size) * 2))
        sm = np.sort(pat.ravel()[:: max(1, pat.size // 20000)])
        pat = np.interp(pat, sm, np.linspace(0, 1, len(sm))).astype(np.float32)     # uniform -> tone-accurate
        w = 0.06
    else:
        raise ValueError(kind)
    if kind == "fm":
        return np.clip((c - pat) / w + 0.5, 0, 1).astype(np.float32)
    # tone-accurate screen: threshold = quantile of the pattern, so area covered == coverage
    q = _pattern_quantiles(kind)
    c = np.clip(c, 0, 1)
    t = np.interp(c, np.linspace(0, 1, len(q)), q) + w * (F.smoothstep(0.85, 1.0, c) - (1 - F.smoothstep(0.0, 0.12, c)))    # exact empty / solid at the ends
    return np.clip((t - pat) / w + 0.5, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- separations

def stretch_tone(L, lo=2, hi=98, local: float = 0.0):
    if local > 0:
        L = F.local_equalize(L, 2.5, mix=local)
    return F.stretch(L, lo, hi)


def sep_tone(L: np.ndarray, n: int, *, soft_w: float = 0.04, spread: float = 0.0) -> list[np.ndarray]:
    """n coverage layers from tone. L 0..1 (1 = light). Layer 0 = lightest tint (covers
    everything darker than the lightest threshold); the last = darkest key. Thresholds sit
    at tone quantiles so every layer has area. Overprinted they rebuild the tonal range."""
    D = 1 - L
    qs = (np.arange(1, n + 1) / (n + 1)) ** (1.0 + spread)
    ths = np.quantile(D, 1 - (1 - qs * 0.82) * 0.98)       # share of area covered falls with k
    ths = np.sort(ths)
    return [F.smoothstep(t - soft_w, t + soft_w, D) for t in ths]


def sep_levels(L: np.ndarray, n: int, soft_w: float = 0.03) -> list[np.ndarray]:
    """Equal-lightness bands (posterise): layer k covers D > k/(n+1)."""
    D = 1 - L
    return [F.smoothstep((k + 1) / (n + 1) - soft_w, (k + 1) / (n + 1) + soft_w, D) for k in range(n)]


def sep_hue(img: np.ndarray, n: int, rng, key: bool = True, depth: float = 1.0) -> list[np.ndarray]:
    """Colour regions: hues clustered in OKLab (a,b); each of the first n-1 inks covers its
    cluster with density ~ darkness-from-paper; the last ink is the dark key (shadows)."""
    lab = F.to_oklab(img)
    L = F.stretch(lab[..., 0], 2, 98)
    ab = lab[..., 1:]
    ch = np.hypot(ab[..., 0], ab[..., 1])
    k = max(1, n - 1 if key else n)
    flat = ab.reshape(-1, 2)
    wts = ch.reshape(-1)
    idx = rng.choice(len(flat), min(6000, len(flat)), replace=False, p=(wts + 1e-3) / (wts + 1e-3).sum())
    cent, _ = F._kmeans(flat[idx].astype(np.float64), k, rng, 12)
    # order clusters by hue angle so ink assignment is stable
    order = np.argsort(np.arctan2(cent[:, 1], cent[:, 0]))
    cent = cent[order]
    d = np.stack([np.hypot(ab[..., 0] - c[0], ab[..., 1] - c[1]) for c in cent], -1)
    wgt = np.exp(-(d / 0.05) ** 2 * 0.6)
    wgt /= wgt.sum(-1, keepdims=True) + 1e-6
    wgt = F.blur(wgt, max(1.0, 2.0 * U(img.shape[1::-1])))
    dens = F.smoothstep(0.12, 0.9, (1 - L) * 0.8 * depth + 0.4 * np.clip(ch / 0.12, 0, 1))
    covs = [np.clip(wgt[..., j] * dens * 1.3, 0, 1) for j in range(k)]
    if key:
        covs.append(F.smoothstep(0.55, 0.9, 1 - L))
    return covs


def sep_cmy(img: np.ndarray, n: int = 3, key: bool = False, invert: bool = False) -> list[np.ndarray]:
    """Channel separation: ink k covers 1 - channel k (cyan, magenta, yellow style). With
    the right inks this rebuilds the photo, with arbitrary inks it gives false colour."""
    ch = [1 - img[..., 0], 1 - img[..., 1], 1 - img[..., 2]]
    if invert:
        ch = [1 - c for c in ch]
    ch = [F.stretch(c, 1, 99) for c in ch]
    covs = ch[:min(n, 3)]
    if key or n > 3:
        k = np.minimum.reduce(ch) ** 1.5
        covs.append(F.stretch(k, 20, 99))
    return covs


def sep_masks(masks: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.clip(np.asarray(m, np.float32), 0, 1) for m in masks]


# ---------------------------------------------------------------- surface / registration

def shift(a: np.ndarray, dx: float, dy: float, rot: float = 0.0, scale: float = 1.0) -> np.ndarray:
    if abs(dx) + abs(dy) + abs(rot) < 1e-4 and scale == 1.0:
        return a
    H, W = a.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, scale)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(np.ascontiguousarray(a), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def wear_edges(m: np.ndarray, size, rng, amount: float) -> np.ndarray:
    """Roughen shape edges and drop a few worn patches: ink erodes where noise is high,
    mostly near edges (m between 0 and 1) so solid areas stay readable."""
    if amount <= 0:
        return m
    H, W = m.shape
    n = 0.7 * noise((H, W), rng, max(1.0, 1.4 * U(size) + 0.4)) + 0.5 * noise((H, W), rng, max(4.0, 9 * U(size)))
    out = F.smoothstep(0.5 - 0.08, 0.5 + 0.08, np.clip(m, 0, 1) + 0.22 * amount * n)
    # worn patches inside solids
    patch = F.smoothstep(1.9, 2.7, noise((H, W), rng, max(3.0, 7 * U(size))) + 0.5 * noise((H, W), rng, max(1.0, 1.6 * U(size))))
    return np.clip(out * (1 - 0.5 * amount * patch), 0, 1).astype(np.float32)


def starve(m: np.ndarray, height: np.ndarray, rng, size, grain: float) -> np.ndarray:
    """Ink grain: ink fails to fill the paper's valleys (fibre/tooth relief) and pools slightly elsewhere."""
    if grain <= 0:
        return m
    H, W = m.shape
    valley = F.smoothstep(0.42, 0.12, height)
    pool = 1 + 0.05 * grain * noise((H, W), rng, max(6.0, 14 * U(size)))
    void = np.clip(valley * 0.85 * grain, 0, 0.9)
    return np.clip(m * (1 - void) * pool, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- the press

def lay(out: np.ndarray, m: np.ndarray, ink, opacity: float = 1.0, mode: str = "multiply") -> np.ndarray:
    a = (np.clip(m, 0, 1) * opacity)[..., None]
    ink = np.asarray(ink, np.float32)
    if mode == "multiply":
        return out * (1 - a * (1 - ink))
    if mode == "screen":
        return 1 - (1 - out) * (1 - a * ink)
    return out * (1 - a) + ink * a


def press(covs: Sequence[np.ndarray], inks: Sequence, size, rng, *, paper_rgb, screens: Sequence[dict] | dict | None = None,
          opacity: float | Sequence[float] = 0.92, mode: str = "multiply", texture: float = 1.0, fibre: float = 0.6,
          mottle: float = 0.5, grain: float = 0.5, reg: float = 6.0, wear: float = 0.3,
          paper_override=None, plain: Sequence[int] = ()) -> np.ndarray:
    """Print ``covs`` (list of 0..1 fields) with ``inks`` over paper.

    screens: dict or per-ink list of dicts for :func:`screen` (kind/pitch/angle/gain).
    texture scales fibre, mottle, grain and wear together (0 = clean print); ``reg`` (master px)
    is registration drift and is NOT scaled by texture, so a clean print can stay misregistered."""
    W, H = size
    t = float(np.clip(texture, 0, 1.5))
    if paper_override is not None:
        prgb, ph = paper_override
    else:
        prgb, ph = paper(size, rng, paper_rgb, fibre=fibre * t, mottle=mottle * t, tooth=0.5 * t)
    out = prgb.copy()
    n = len(covs)
    if isinstance(screens, dict) or screens is None:
        screens = [screens or {}] * n
    ops = list(opacity) if isinstance(opacity, (list, tuple)) else [opacity] * n
    rr = max(reg * U(size), PX_FLOOR["reg"] if reg > 0 else 0)
    base_ang = rng.uniform(0, 2 * math.pi)
    for k in range(n):
        sp = dict(screens[k] if k < len(screens) else {})
        m = screen(covs[k], size, rng=rng, **sp)
        ang = base_ang + k * 2.1
        m = shift(m, rr * math.cos(ang) * (0.5 + 0.5 * k), rr * math.sin(ang) * (0.5 + 0.5 * k), rot=0.18 * math.sin(ang * 1.7) * (reg > 0) * 2, scale=1.0 + 0.0008 * (k % 2) * (reg > 0))
        if k not in plain:
            m = wear_edges(m, size, rng, wear * t)
            m = starve(m, ph, rng, size, grain * t * (1.0 if mode == "multiply" else 0.45))
        out = lay(out, m, inks[k], ops[k], mode)
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- sources

def prep_source(ctx, local: float = 0.0, crop: str = "feature", zoom: float = 0.0):
    """Source fitted to the frame (or a generated stand-in) -> rgb."""
    W, H = ctx.size
    if ctx.source is not None:
        return F.fit_cover(np.asarray(ctx.source, np.float32)[..., :3], (W, H), crop, zoom)
    return shapes_rgb((W, H), ctx.seed)


def shapes_rgb(size, seed: int, n: int = 7) -> np.ndarray:
    """Stand-in 'subject': big overlapping soft-edged discs, bars and half discs in 3 hues over a
    light ground, spread over the whole frame (no focal point). Used when no source is supplied."""
    rng = np.random.default_rng(seed + 991)
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.ones((H, W, 3), np.float32) * 0.94
    hues = rng.uniform(0, 360) + np.array([0, 140, 215])
    for i in range(n):
        h = hues[i % 3]
        col = pal.oklch_to_rgb(rng.uniform(0.3, 0.8), 0.14, h)
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(0.18, 0.36) * W
        if i % 3 == 0:
            m = np.hypot(xx - cx, yy - cy) < r
        elif i % 3 == 1:
            m = (np.abs(xx - cx) < r * rng.uniform(0.3, 1.2)) & (np.abs(yy - cy) < r * rng.uniform(0.25, 0.6))
        else:
            m = (np.hypot(xx - cx, yy - cy) < r) & (yy < cy)
        shade = 0.85 + 0.15 * (xx / W)
        img[m] = (col[None] * shade[m][:, None])
    return img


# ---------------------------------------------------------------- shared look resolution for effects

COLOUR_PARAMS = {
    "variant": ("auto", "auto (palette polarity) | light (ink on paper, multiply) | dark (light inks on dark stock)"),
    "inks": ("", "comma list of ink colours: #hex, names (ultramarine, vermilion, pink ...) or palette roles; '' = from the palette hue"),
    "paper": ("", "paper colour (#hex / name); '' = palette canvas"),
}
SOURCE_PARAMS = {
    "crop": ("feature", "feature | center: how a source of another aspect is cropped"),
    "zoom": (0.0, "0 auto; 0.3-0.9 zooms into the densest region"),
    "local": (0.3, "0..1 local tone equalisation of the source (spreads ink over the whole frame)"),
}
SURFACE_PARAMS = {
    "clean": (False, "true = clean variant: no paper fibre, grain or wear (registration stays)"),
    "texture": (1.0, "0..1.5 overall surface strength (fibre, mottle, grain, wear)"),
    "fibre": (0.6, "paper fibre visibility"),
    "grain": (0.5, "ink grain: ink starves in paper valleys"),
    "reg": (6.0, "registration drift between inks (px at 1500 wide, floor ~1.4 px)"),
    "wear": (0.3, "edge roughening and worn patches"),
}


def params(**extra):
    d = {}
    d.update(COLOUR_PARAMS); d.update(SOURCE_PARAMS); d.update(SURFACE_PARAMS)
    d.update(extra)
    return d


def surface(ctx) -> dict:
    t = 0.0 if ctx.p("clean", False) else float(ctx.p("texture", 1.0))
    return dict(texture=t, fibre=float(ctx.p("fibre", 0.6)), grain=float(ctx.p("grain", 0.5)),
                reg=float(ctx.p("reg", 6.0)), wear=float(ctx.p("wear", 0.3)))


def resolve_look(ctx, n: int, *, light_ls=(0.64, 0.38), accent_last: bool = False):
    """-> dict(dark, palette, paper, inks[n], mode). Colour is fully independent of structure."""
    P = ctx.palette
    var = str(ctx.p("variant", "auto"))
    pol = P.polarity if var == "auto" else var
    h = pal.rgb_to_oklch(P.structure)[2]
    if pol != P.polarity:
        P = pal.make(h, polarity=pol, harmony="analogous", three=P.structure2 is not None)
    dark = pol == "dark"
    paper_rgb = color(ctx.p("paper"), P) if ctx.p("paper") else P.canvas
    spec = ctx.p("inks", "")
    if spec:
        inks = colors(spec, P)
        while len(inks) < n:
            inks.append(inks[-1])
        inks = inks[:n]
    else:
        ah = pal.rgb_to_oklch(P.accent)[2]
        if dark:
            ls = np.linspace(0.36, 0.76, n) if n > 1 else [0.62]
            inks = [pal.oklch_to_rgb(float(l), 0.14, h + 8 * (k - n / 2)) for k, l in enumerate(ls)]
        else:
            ls = np.linspace(0.68, 0.36, n) if n > 1 else [0.42]
            inks = [pal.oklch_to_rgb(float(l), 0.12 + 0.07 * k / max(n - 1, 1), h) for k, l in enumerate(ls)]
        if accent_last and n >= 2:
            inks[0] = P.accent
    return dict(dark=dark, palette=P, paper=paper_rgb, inks=inks, mode="over" if dark else "multiply",
                opacity=0.93 if dark else 0.95)


def source_rgb(ctx, crop=None, zoom=None):
    """ctx.source / 2-D ctx.data / generated shapes -> rgb fitted to the frame."""
    W, H = ctx.size
    crop = ctx.p("crop", "feature") if crop is None else crop
    zoom = float(ctx.p("zoom", 0.0)) if zoom is None else zoom
    if ctx.source is not None:
        return F.fit_cover(np.asarray(ctx.source, np.float32)[..., :3], (W, H), crop, zoom)
    d = ctx.data
    if d is not None:
        a = np.asarray(d, np.float32)
        if a.ndim == 2:
            a = (a - a.min()) / (np.ptp(a) + 1e-9)
            return np.repeat(cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)[..., None], 3, 2)
    return shapes_rgb((W, H), ctx.seed)


def tone_of(img, local: float, dark: bool):
    """Ink presence tone as 'L' (1 = light = no ink): dark mode puts ink where the source is bright."""
    L = stretch_tone(F.luminance(img), 2, 98, local)
    return 1 - L if dark else L
