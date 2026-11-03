"""Shared toolkit for the atmosphere (``atmos-*``) and optics (``optics-*``) effects.

Everything is deterministic numpy / cv2 on float32 sRGB (H, W, 3) 0..1, the
``ctx.source`` convention. Light transport (glow, haze, reflections) is done in
approximately linear light (gamma 2.2) so a glow adds energy instead of
washing the picture grey. Sizes are expressed at a 1500 px master and scale with
the frame (``scale_of``), so a 288x192 render has the same proportions.

Noise and textures are functions of (x, y, seed) only (``hash_noise``,
``value_noise``): a video of N frames with the same seed never crawls.

Building blocks
    lin / srgb               gamma 2.2 pair
    blur / blur_big          gaussian; blur_big works on a small copy for big sigmas
    emission                 emitter extractor (luminance / colour / threshold / local peak, skin-safe)
    glow_stack               multi-scale additive glow of an emission field
    streak_blur              directional scattering (anamorphic / diagonal streaks)
    halation_field           red-orange fringe: tight, hue-tinted, highlights only
    depth_field              depth proxy (gradient / dark-channel / hand-authored polygons or array)
    veil                     depth-dependent haze with in-scattered light
    fog_banks                patchy fog density (seeded fractal noise), lit by the emission
    wet_reflect              vertical stretch of light sources inside a ground mask
    light_pools              a few large soft pools of light under the strongest lights
    anchor_blacks            keep the dark end where the input had it
    perspective_fill / radial_warp / screen_grid / focus_field   optics
"""
from __future__ import annotations

import math

import cv2
import numpy as np


LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ------------------------------------------------------------------ basics

def lin(x):
    return np.power(np.clip(x, 0, None), 2.2, dtype=np.float32) if isinstance(x, np.ndarray) else x ** 2.2


def srgb(x):
    return np.power(np.clip(x, 0, None), 1 / 2.2, dtype=np.float32) if isinstance(x, np.ndarray) else x ** (1 / 2.2)


def scale_of(shape) -> float:
    return max(shape[0], shape[1]) / 1500.0


def luma(x):
    return (x @ LUM).astype(np.float32)


def blur(a, sigma):
    if sigma < 0.3:
        return a
    return cv2.GaussianBlur(np.ascontiguousarray(a), (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT)


def blur_big(a, sigma):
    """Gaussian of any sigma at roughly constant cost (blurs a reduced copy)."""
    if sigma < 6:
        return blur(a, sigma)
    f = int(min(sigma / 3.0, 16))
    H, W = a.shape[:2]
    sw, sh = max(8, W // f), max(8, H // f)
    sm = cv2.resize(np.ascontiguousarray(a), (sw, sh), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), sigma / f, borderType=cv2.BORDER_REFLECT)
    out = cv2.resize(sm, (W, H), interpolation=cv2.INTER_CUBIC)
    return out.reshape(a.shape) if out.shape != a.shape else out


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def mix(a, b, t):
    t = np.asarray(t, np.float32)
    if t.ndim == 2 and np.ndim(a) == 3:
        t = t[..., None]
    return a * (1 - t) + b * t


def screen(a, b):
    return 1 - (1 - np.clip(a, 0, 1)) * (1 - np.clip(b, 0, 1))


def hue_rgb(h_deg: float, C: float = 0.12, L: float = 0.75) -> np.ndarray:
    from .palette import oklch_to_rgb
    return oklch_to_rgb(L, C, h_deg).astype(np.float32)


def coords(shape):
    H, W = shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return xx, yy


# ------------------------------------------------------------------ coordinate noise (video-stable)

def _hash(ix, iy, seed):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + int(seed) * 2147483647) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


def hash_noise(shape, seed: int = 0) -> np.ndarray:
    """White noise in [-1, 1] that is a pure function of (pixel coordinate, seed)."""
    H, W = shape[:2]
    iy, ix = np.mgrid[0:H, 0:W]
    return _hash(ix, iy, seed) * 2 - 1


def value_noise(shape, period: float, seed: int = 0, octaves: int = 1, stretch=(1.0, 1.0)) -> np.ndarray:
    """Smooth coordinate-based noise 0..1; ``period`` in px; frame-independent lattice
    anchored at the image origin (the same for any frame count)."""
    H, W = shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        p = max(2.0, period / (2 ** o))
        fx, fy = xx / (p * stretch[0]), yy / (p * stretch[1])
        ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        tx, ty = fx - ix, fy - iy
        tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
        s = seed * 31 + o * 101
        a, b = _hash(ix, iy, s), _hash(ix + 1, iy, s)
        c, d = _hash(ix, iy + 1, s), _hash(ix + 1, iy + 1, s)
        out += amp * ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty)
        tot += amp
        amp *= 0.5
    return out / tot


# ------------------------------------------------------------------ skin / emission

def skin_mask(img: np.ndarray) -> np.ndarray:
    from .grading import skin_weight, rgb_to_oklab
    small = cv2.resize(img, (max(32, img.shape[1] // 3), max(32, img.shape[0] // 3)), interpolation=cv2.INTER_AREA)
    w = skin_weight(rgb_to_oklab(small)).astype(np.float32)
    w = blur(w, 2.0)
    return cv2.resize(w, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def emission(img: np.ndarray, mode: str = "luminance", threshold: float = 0.6, softness: float = 0.25,
             hue: float = -1.0, hue_width: float = 40.0, local: float = 0.5, skin: float = 0.85,
             mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Extract emitters. Returns (weight 0..1 (H,W), emitted colour (H,W,3) in linear light).

    mode  luminance   bright pixels (smooth knee at ``threshold``)
          colour      saturated AND bright pixels (neon, signs), optionally only a hue range
          peak        small bright things: luminance minus its wide local mean (lamps, not skies)
          both        max(luminance, colour)
          mask        use ``mask`` (H, W)
    threshold  0..1 in display luminance; softness = knee width
    local      0..1 how much the local-peak test suppresses big bright areas (walls, sky)
    skin       0..1 suppress emission on skin-tone pixels (people do not glow)
    """
    s = scale_of(img.shape)
    L = luma(img)
    mx, mn = img.max(-1), img.min(-1)
    sat = (mx - mn) / (mx + 1e-4)
    lo, hi = max(0.0, threshold - softness / 2), min(1.0, threshold + softness / 2 + 1e-3)
    wl = smoothstep(lo, hi, L)
    wc = smoothstep(lo * 0.8, hi * 0.9, mx) * smoothstep(0.35, 0.7, sat)
    if hue >= 0:
        from .grading import rgb_to_oklab
        sm = cv2.resize(img, (max(32, img.shape[1] // 3), max(32, img.shape[0] // 3)), interpolation=cv2.INTER_AREA)
        lab = rgb_to_oklab(sm)
        h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
        dh = np.abs((h - hue + 180) % 360 - 180)
        hm = cv2.resize(smoothstep(hue_width, hue_width * 0.5, dh).astype(np.float32), (img.shape[1], img.shape[0]))
        wc = wc * hm
        wl = wl * hm if mode == "luminance" else wl
    if mode == "luminance":
        w = wl
    elif mode == "colour" or mode == "color":
        w = wc
    elif mode == "both":
        w = np.maximum(wl, wc)
    elif mode == "peak":
        pk = L - blur_big(L, 40 * s)
        w = smoothstep(0.08, 0.35, pk) * smoothstep(lo * 0.6, hi, L)
    elif mode == "mask" and mask is not None:
        w = np.clip(mask, 0, 1).astype(np.float32)
    else:
        w = wl
    if local > 0 and mode in ("luminance", "both"):
        pk = L - blur_big(L, 60 * s)
        w = w * (1 - local + local * smoothstep(0.0, 0.18, pk))
    if skin > 0:
        w = w * (1 - skin * skin_mask(img))
    w = np.clip(w, 0, 1).astype(np.float32)
    col = lin(img) * w[..., None]
    return w, col


def anchor_blacks(out: np.ndarray, ref: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """If ``out`` lifted the dark end of ``ref`` (haze, glow), pull it back so
    the 1st-percentile luminance keeps its place (dark anchors). strength 0..1."""
    if strength <= 0:
        return out
    pr = float(np.percentile(luma(ref), 1.5))
    po = float(np.percentile(luma(out), 1.5))
    d = max(0.0, po - pr) * strength
    if d <= 1e-4:
        return out
    return np.clip((out - d) / max(1e-3, 1 - d), 0, 1)


def core_mask(img: np.ndarray, lo: float = 0.86, hi: float = 0.97) -> np.ndarray:
    return smoothstep(lo, hi, img.max(-1))


# ------------------------------------------------------------------ glow families

def glow_stack(col: np.ndarray, radii=(6, 18, 54, 160), weights=None, falloff: float = 0.5) -> np.ndarray:
    """Multi-scale glow of an emission colour field (linear light). Radii px at
    1500 wide (scaled here). Larger scales carry less energy by ``falloff``
    (0 = equal energy per octave, 1 = tight core dominates) and each scale is
    unit-sum, so total energy is conserved."""
    s = scale_of(col.shape)
    n = len(radii)
    w = np.array(weights if weights is not None else [(1.0 - falloff * 0.8) ** i for i in range(n)], np.float32)
    w = w / w.sum()
    acc = np.zeros_like(col)
    for r, wi in zip(radii, w):
        acc += wi * blur_big(col, max(0.5, r * s))
    return acc


def streak_blur(col: np.ndarray, length: float, angle_deg: float = 0.0, decay: float = 0.5) -> np.ndarray:
    """Directional scattering: each light is smeared along ``angle`` (0 = horizontal)
    both ways with a two-sided exponential kernel of mean ``length`` px at 1500 wide
    (anamorphic horizontal streak, rain/vertical streak, diagonal). Energy-conserving."""
    s = scale_of(col.shape)
    L = max(1.0, length * s)
    H, W = col.shape[:2]
    f = 2 if L > 20 else 1
    c = cv2.resize(col, (W // f, H // f), interpolation=cv2.INTER_AREA) if f > 1 else col
    Lh = L / f
    ang = angle_deg
    h, w = c.shape[:2]
    # rotate onto the horizontal axis inside a padded canvas
    diag = int(math.hypot(h, w)) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 2] += (diag - w) / 2
    M[1, 2] += (diag - h) / 2
    rot = cv2.warpAffine(np.ascontiguousarray(c), M, (diag, diag), flags=cv2.INTER_LINEAR)
    k = int(min(Lh * 5, diag // 2 - 1)) | 1
    x = np.arange(-(k // 2), k // 2 + 1, dtype=np.float32)
    ker = np.exp(-np.abs(x) / (Lh * max(0.15, 1.0 - 0.6 * decay)))
    ker /= ker.sum()
    rot = cv2.sepFilter2D(rot, -1, ker.reshape(1, -1), np.array([1.0], np.float32), borderType=cv2.BORDER_CONSTANT)
    rot = cv2.GaussianBlur(rot, (0, 0), 0.7)
    Mi = cv2.invertAffineTransform(M)
    back = cv2.warpAffine(rot, Mi, (w, h), flags=cv2.INTER_LINEAR)
    if f > 1:
        back = cv2.resize(back, (W, H), interpolation=cv2.INTER_CUBIC)
    return back.reshape(col.shape)


def halation_field(w: np.ndarray, col: np.ndarray, tint, radius: float = 10.0, spread: float = 2.6) -> np.ndarray:
    """Halation: light that passes the emulsion, reflects off the film base and
    exposes it again as a fringe. Warm (red-orange) and *outside* the source:
    the glow of the highlight minus its tight core, multiplied by a tint."""
    s = scale_of(col.shape)
    g1 = blur(col, max(0.7, radius * s))
    g2 = blur_big(col, max(1.5, radius * spread * s))
    fringe = np.clip(0.55 * g1 + 0.45 * g2 - 0.25 * col, 0, None)
    t = np.asarray(tint, np.float32)
    lum = fringe @ LUM
    return (lum[..., None] * (t / (t @ LUM + 1e-4)))


# ------------------------------------------------------------------ depth, haze, fog

def _poly_mask(shape, polys, feather: float) -> np.ndarray:
    H, W = shape[:2]
    m = np.zeros((H, W), np.float32)
    for p in polys:
        pts = np.array([[x * W, y * H] for x, y in p], np.int32)
        cv2.fillPoly(m, [pts], 1.0)
    return blur(m, feather)


def mask_from_data(data, shape, feather: float = 10.0):
    """Hand-authored mask from ctx.data: a 2D array, {'mask': array or 'polygons': [[(x,y),...]]}
    (normalised 0..1 coordinates, y down), or a list of polygons. None if absent."""
    if data is None:
        return None
    H, W = shape[:2]
    if isinstance(data, dict):
        if data.get("mask") is not None:
            data = data["mask"]
        elif data.get("polygons") is not None:
            return _poly_mask(shape, data["polygons"], feather)
        else:
            return None
    try:
        a = np.asarray(data, np.float32)
    except Exception:
        return None
    if a.ndim == 2 and a.shape[1] != 2:
        return np.clip(cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR), 0, 1)
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple)) and len(data[0]) and isinstance(data[0][0], (list, tuple)):
        return _poly_mask(shape, data, feather)
    return None


def depth_field(img: np.ndarray, mode: str = "gradient", horizon: float = 0.45, angle: float = 0.0,
                mask=None, gamma: float = 1.0) -> np.ndarray:
    """Depth proxy 0 (near) .. 1 (far).
    gradient       far at the horizon row, near at the bottom; above the horizon it stays far.
    top            far at the top, near at the bottom (landscapes)
    dark-channel   thickness from the dark-channel prior (bright, low-contrast = far); photo-driven
    mask           a hand-authored far-mask (1 = far)
    flat           uniform 0.5
    """
    H, W = img.shape[:2]
    yy, xx = coords(img.shape)
    if mode == "mask" and mask is not None:
        d = np.clip(mask, 0, 1)
    elif mode == "dark-channel":
        s = scale_of(img.shape)
        dc = cv2.erode(img.min(-1), np.ones((max(3, int(15 * s)) | 1,) * 2, np.uint8))
        dc = blur_big(dc, 12 * s)
        d = smoothstep(0.02, 0.6, dc)
    elif mode == "top":
        d = 1 - yy / H
    elif mode == "flat":
        d = np.full((H, W), 0.5, np.float32)
    else:
        a = math.radians(angle)
        yy2 = yy - np.tan(a) * (xx - W / 2)
        hy = horizon * H
        d = np.where(yy2 < hy, 1.0, 1 - (yy2 - hy) / max(1.0, H - hy))
        d = np.clip(d, 0, 1)
        d = blur_big(d.astype(np.float32), 20 * scale_of(img.shape))
    return np.clip(d, 0, 1).astype(np.float32) ** gamma


def ambient_colour(img: np.ndarray, pct: float = 92) -> np.ndarray:
    """Linear colour of the scene's ambient light: mean of the brighter blurred pixels."""
    sm = cv2.resize(img, (48, 32), interpolation=cv2.INTER_AREA)
    L = luma(sm)
    sel = L >= np.percentile(L, pct)
    c = lin(sm)[sel].mean(0)
    return c.astype(np.float32)


def veil(img: np.ndarray, depth: np.ndarray, density: float = 1.2, level: float = 0.55, colour=None,
         lift: float = 0.0, scatter: np.ndarray | None = None, protect: np.ndarray | None = None) -> np.ndarray:
    """Depth-dependent veiling haze (Koschmieder): out = img * t + A * (1 - t), t = exp(-density * depth),
    in linear light. A is the airlight (``colour`` linear rgb scaled to luminance ``level``);
    ``scatter`` (H,W,3 linear) adds in-scattered light from lamps; ``protect`` (0..1) keeps
    pixels (cores, skin) out of the veil."""
    t = np.exp(-density * depth)
    if protect is not None:
        t = t + (1 - t) * protect
    c = np.asarray(colour if colour is not None else [1, 1, 1], np.float32)
    c = c / (float(c @ LUM) + 1e-4) * (level ** 2.2)
    A = np.ones(img.shape[:2] + (3,), np.float32) * c
    if scatter is not None:
        A = A + scatter * (1.0 - t)[..., None] * 2.0
    x = lin(img)
    out = x * t[..., None] + A * (1 - t)[..., None]
    out = out + lift * (1 - t)[..., None] * 0.1
    return srgb(np.clip(out, 0, 1))


def fog_banks(shape, seed: int, scale: float = 260.0, coverage: float = 0.5, height: float = 0.0, horizon: float = 0.6,
              layers: int = 3, drift: float = 0.0) -> np.ndarray:
    """Patchy fog density 0..1: stretched fractal value noise (wider than tall), several layers
    at different scales. ``height`` 0..1 pulls the fog to the ground (density falls with y above
    ``horizon``). ``drift`` shifts the noise lattice, so a video can slide it (pure function of drift)."""
    H, W = shape[:2]
    s = scale_of(shape)
    acc = np.zeros((H, W), np.float32)
    for i in range(layers):
        n = value_noise(shape, scale * s * (0.6 ** i), seed + i * 17, octaves=3, stretch=(1.8, 1.0))
        acc += n / layers
    acc = np.roll(acc, int(drift * W), axis=1) if drift else acc
    d = smoothstep(1.0 - coverage - 0.25, 1.0 - coverage + 0.25, acc)
    if height > 0:
        yy = np.arange(H, dtype=np.float32)[:, None] / H
        g = np.exp(-np.clip(horizon - yy, 0, None) / (0.12 + 0.35 * (1 - height)))
        d = d * (1 - height + height * g)
    return d.astype(np.float32)


# ------------------------------------------------------------------ wet ground

def ground_mask(shape, horizon: float = 0.62, feather: float = 0.05, data=None, img=None, auto_dark: float = 0.0) -> np.ndarray:
    """1 on the ground. From ctx.data when authored; else everything below ``horizon`` (fraction of
    height, soft edge ``feather``). ``auto_dark`` > 0 additionally keeps only darker pixels (wet
    asphalt, water) so walls and people standing on the ground are not reflected into."""
    m = mask_from_data(data, shape)
    if m is None:
        H, W = shape[:2]
        yy = np.arange(H, dtype=np.float32)[:, None] / H
        m = np.repeat(smoothstep(horizon - feather, horizon + feather, yy), W, axis=1)
    if auto_dark > 0 and img is not None:
        L = blur(luma(img), 3.0)
        m = m * (1 - auto_dark * smoothstep(0.35, 0.65, L))
    return m.astype(np.float32)


def wet_reflect(col: np.ndarray, ground: np.ndarray, horizon_px: float, length: float = 0.16, roughness: float = 0.4,
                ripple: float = 0.3, seed: int = 0, fresnel: float = 0.6) -> np.ndarray:
    """Reflection of emitters on a wet surface (linear light, to be added inside the ground mask).

    Optics: a light above the ground is mirrored below its contact line and, on a rough wet
    surface, smeared along the vertical (the mirror image is stretched by facet slopes).
    We (1) mirror the emission about the horizon line, scaled in y (``length`` = reach as a
    fraction of height), (2) smear it down with an exponential kernel, (3) blur it sideways in
    proportion to ``roughness``, (4) warp it by a seeded coordinate-noise ripple, and (5) weight
    by a Fresnel-like gain that grows toward the horizon."""
    H, W = col.shape[:2]
    s = scale_of(col.shape)
    # mirror about the horizon row (clip to the image)
    yy = np.arange(H, dtype=np.float32)
    src_y = 2 * horizon_px - yy
    valid = (src_y >= 0) & (src_y <= H - 1)
    mx = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    my = np.tile(np.clip(src_y, 0, H - 1)[:, None], (1, W))
    # ripple: horizontal displacement from seeded noise, rows far from horizon wobble more
    if ripple > 0:
        n = value_noise(col.shape, 40 * s, seed, 2, stretch=(2.5, 0.6)) - 0.5
        dist = np.clip((yy - horizon_px) / max(1.0, H - horizon_px), 0, 1)[:, None]
        mx = mx + n * ripple * 60 * s * (0.3 + dist)
    mir = cv2.remap(np.ascontiguousarray(col), mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mir = mir * valid[:, None, None]
    # vertical smear (cumulative exponential going down), more with roughness
    sm_len = max(1.0, (0.02 + 0.2 * roughness) * H * length * 6)
    k = int(min(sm_len * 4, H - 1)) | 1
    x = np.arange(0, k, dtype=np.float32)
    ker = np.exp(-x / sm_len); ker /= ker.sum()
    kern = np.zeros(2 * k - 1, np.float32); kern[k - 1:] = ker           # causal (light goes downward)
    smear = cv2.sepFilter2D(mir, -1, np.array([1.0], np.float32).reshape(1, 1), kern.reshape(-1, 1), borderType=cv2.BORDER_CONSTANT)
    sx = (2 + 14 * roughness) * s
    smear = blur(smear, sx)
    # keep a little sharp mirror image for glossy surfaces
    sharp = blur(mir, 1.0 * s + 1.0)
    out = (1 - roughness) * 0.6 * sharp + (0.4 + 0.6 * roughness) * smear * (1.0 + 0.8 * roughness)
    dist = np.clip((yy - horizon_px) / max(1.0, H - horizon_px), 0, 1)
    fr = (fresnel * (1 - dist) ** 1.5 + (1 - fresnel) * 0.5)[:, None, None]
    reach = np.exp(-(yy - horizon_px).clip(0) / max(1.0, length * H * 2.0))[:, None, None]
    return out * fr * reach * ground[..., None]


def light_pools(w: np.ndarray, col: np.ndarray, n: int = 5, radius: float = 0.12, squash: float = 1.0,
                spread: float = 1.0) -> np.ndarray:
    """A FEW substantial pools of light: the ``n`` strongest, well separated emitters each get a big
    soft pool (radius = ``radius`` of the frame diagonal * sqrt(its share of energy)), in the
    emitter's own colour. Fewer and larger beats many small glows at thumbnail size."""
    H, W = w.shape
    s = scale_of(w.shape)
    f = 6
    ws = cv2.resize(w, (W // f, H // f), interpolation=cv2.INTER_AREA)
    cs = cv2.resize(col, (W // f, H // f), interpolation=cv2.INTER_AREA)
    en = blur(ws, 2.0)
    out = np.zeros_like(col)
    h, wd = en.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    tot = float(en.sum()) + 1e-6
    sep = max(4, int(0.12 * wd))
    work = en.copy()
    picks = []
    for _ in range(int(n)):
        i = int(np.argmax(work))
        py, px = divmod(i, wd)
        if work[py, px] < 0.02 * en.max():
            break
        y0, y1, x0, x1 = max(0, py - sep), min(h, py + sep + 1), max(0, px - sep), min(wd, px + sep + 1)
        share = float(en[y0:y1, x0:x1].sum()) / tot
        c = cs[y0:y1, x0:x1].sum((0, 1))
        c = c / (float(c @ LUM) + 1e-6)
        picks.append((px * f, py * f, share, c))
        work[y0:y1, x0:x1] = 0
    diag = math.hypot(W, H)
    for px, py, share, c in picks:
        R = radius * diag * (0.45 + 0.55 * math.sqrt(min(1.0, share * len(picks)))) * spread
        r2 = ((xx - px) / R) ** 2 + ((yy - py) / (R * squash)) ** 2
        pool = 1.0 / (1.0 + 2.2 * r2) ** 1.6 * np.exp(-r2 / 6)
        out += pool[..., None] * c * (0.25 + 0.75 * math.sqrt(min(1.0, share * len(picks) + 0.1)))
    return out


# ------------------------------------------------------------------ optics

def perspective_fill(img: np.ndarray, corners, interp=cv2.INTER_CUBIC) -> np.ndarray:
    """Warp so the screen looks off-axis while it keeps filling the frame: ``corners`` are four
    (dx, dy) offsets as fractions of the frame, applied to the destination corners TL, TR, BR, BL
    moving INWARD means the source quad is larger (so coverage is guaranteed when dx,dy >= 0 point
    outward of the source rect). We instead sample a source quad *inside* the image: positive
    offsets move the sampled corner inward, which makes that dst corner look farther away/larger."""
    H, W = img.shape[:2]
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    c = np.asarray(corners, np.float32).reshape(4, 2)
    src = np.float32([[c[0, 0] * W, c[0, 1] * H], [W - c[1, 0] * W, c[1, 1] * H],
                      [W - c[2, 0] * W, H - c[2, 1] * H], [c[3, 0] * W, H - c[3, 1] * H]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(np.ascontiguousarray(img), M, (W, H), flags=interp, borderMode=cv2.BORDER_REFLECT)


def radial_warp(img: np.ndarray, k1: float = 0.0, k2: float = 0.0, cx: float = 0.0, cy: float = 0.0,
                fill: bool = True, ca: float = 0.0) -> np.ndarray:
    """Radial lens distortion (r' = r (1 + k1 r^2 + k2 r^4), r normalised so the frame corner is 1).
    k1 > 0 barrel, < 0 pincushion. ``fill`` rescales so no border is ever visible (it zooms in just
    enough; reflect padding is the fallback). ``ca`` px of lateral chromatic aberration at the corner,
    red and blue sampled at slightly different radii."""
    H, W = img.shape[:2]
    xx, yy = coords(img.shape)
    nx = (xx - W / 2) / (W / 2) + cx
    ny = (yy - H / 2) / (H / 2) + cy
    rr = np.hypot(nx, ny) / math.sqrt(2)
    zoom = 1.0 / max(1.0, 1 + k1 + k2) if fill else 1.0   # corner sample radius stays inside the frame
    chans = []
    for c_i, dca in enumerate((-1.0, 0.0, 1.0)):
        f = 1 + k1 * rr ** 2 + k2 * rr ** 4
        f = f * zoom * (1 + dca * ca / (W / 2))
        mx = (nx * f - cx) * (W / 2) + W / 2
        my = (ny * f - cy) * (H / 2) + H / 2
        chans.append(cv2.remap(np.ascontiguousarray(img[..., c_i]), mx.astype(np.float32), my.astype(np.float32),
                               cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
    return np.stack(chans, -1)


def focus_field(shape, kind: str = "edges", amount: float = 0.5, angle: float = 0.0, centre: float = 0.5,
                width: float = 0.3) -> np.ndarray:
    """0..1 how out of focus each pixel is. edges (soft corners, sharp middle) | plane (a tilted
    sharp band, tilt-shift like) | none."""
    H, W = shape[:2]
    xx, yy = coords(shape)
    nx, ny = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    if kind == "plane":
        a = math.radians(angle)
        d = (ny - (centre - 0.5) * 2) * math.cos(a) + nx * math.sin(a)
        f = smoothstep(width, width + 0.8, np.abs(d))
    elif kind == "edges":
        f = smoothstep(0.35, 1.25, np.hypot(nx * 0.9, ny))
    else:
        f = np.zeros((H, W), np.float32)
    return (f * amount).astype(np.float32)


def variable_blur(img: np.ndarray, field: np.ndarray, max_sigma: float) -> np.ndarray:
    """Spatially varying blur: blend through a 3-level gaussian stack by ``field`` (0..1)."""
    s = scale_of(img.shape)
    sg = max_sigma * s
    levels = [img, blur(img, sg * 0.5), blur(img, sg)]
    t = np.clip(field, 0, 1) * 2
    a = np.clip(1 - t, 0, 1)[..., None]
    b = (1 - np.abs(t - 1))[..., None]
    c = np.clip(t - 1, 0, 1)[..., None]
    return levels[0] * a + levels[1] * b + levels[2] * c


def screen_grid(img: np.ndarray, pitch: float, gap: float = 0.35, subpixel: float = 0.5, seed: int = 0) -> np.ndarray:
    """What a camera sees of a display: the picture is sampled on a pixel lattice (``pitch`` px,
    given in OUTPUT pixels already), each cell a lit square with a dark gap (``gap`` 0..1) and,
    when the pitch is wide enough, RGB sub-pixel stripes (``subpixel`` 0..1). Below ~2.2 px the
    lattice cannot be resolved so its contrast fades out smoothly (a 288x192 render keeps only the
    luminance structure, never fake sub-pixels)."""
    H, W = img.shape[:2]
    p = float(pitch)
    vis = smoothstep(2.0, 4.0, p)               # fade out as the pitch approaches the Nyquist limit
    if vis <= 0.01:
        return img
    xx, yy = coords(img.shape)
    u = (xx / p) % 1.0
    v = (yy / p) % 1.0
    # cell profile: bright plateau, soft falloff to the gap
    def prof(t):
        e = gap * 0.5
        return 1 - smoothstep(1 - e - 0.12, 1 - e + 0.12, np.abs(t * 2 - 1) * 1.0)
    cell = (prof(u) * prof(v))[..., None]
    out = img * (1 - vis * gap + vis * gap * cell)
    if subpixel > 0 and p >= 3.0:
        sp = np.floor(u * 3).astype(np.int32).clip(0, 2)
        stripe = np.zeros(img.shape, np.float32)
        for c in range(3):
            stripe[..., c] = (sp == c)
        stripe = blur(stripe, 0.4)
        sv = vis * subpixel * smoothstep(3.0, 5.0, p)
        out = out * (1 - sv * 0.5 + sv * 1.5 * stripe)
    return out
