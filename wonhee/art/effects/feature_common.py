"""Shared prelude for the feature-edit effects (not an effect itself)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .. import features as F
from .. import palette as pal

COMMON_PARAMS = {
    "crop": ("feature", "how a source of another aspect is cropped: feature (by detail) | center"),
    "zoom": (0.0, "0 = auto; 0.3-0.9 forces a crop of that fraction of the frame-fitting window, centred on the densest region (use on screenshots with empty margins)"),
    "local": (0.8, "0..1 local-contrast equalisation of the source tone (spreads detail over the whole frame)"),
    "hue_from_source": (False, "re-key the palette to the source's dominant hue (keeps role lightnesses)"),
}


def with_common(params: dict) -> dict:
    out = dict(params)
    for k, v in COMMON_PARAMS.items():
        out.setdefault(k, v)
    return out


def prep(ctx, clip: float = 2.5, invert: bool = False) -> SimpleNamespace:
    """Source fitted to the frame (feature crop), tone fields and palette helpers."""
    W, H = ctx.size
    src = ctx.source if ctx.source is not None else F.synthetic_source((W, H), ctx.seed)
    img = F.fit_cover(np.asarray(src, np.float32)[..., :3], (W, H), ctx.p("crop", "feature"), float(ctx.p("zoom", 0.0)))
    L = F.luminance(img)
    mix = float(ctx.p("local", 0.8))
    t = F.stretch(F.local_equalize(L, clip, mix=mix) if mix > 0 else L, 1, 99)
    if invert:
        t = 1 - t
    p = ctx.palette
    if ctx.p("hue_from_source", False):
        p = pal.make(F.dominant_hue(img, default=pal.rgb_to_oklch(p.structure)[2]), polarity=p.polarity, harmony="analogous",
                     three=p.structure2 is not None)
    ramp = p.ramp(256, ("canvas", "structure", "accent"))

    def lut(x, r=None):
        r = ramp if r is None else r
        return r[(np.clip(x, 0, 1) * 255).astype(np.uint8)]

    def body(x):
        """Palette body colours only (canvas -> structure, no accent): the one-hue-family range."""
        return ramp[(np.clip(x, 0, 1) * 140).astype(np.uint8)]

    s = F.scale_of((H, W))
    dark = p.polarity == "dark"
    return SimpleNamespace(img=img, L=L, t=t, pal=p, ramp=ramp, lut=lut, body=body, s=s, W=W, H=H, dark=dark,
                           ink=(t if dark else 1 - t))   # "ink" = how strongly the source is present (bright on dark, dark on light)


def mixc(a, b, t):
    t = np.asarray(t, np.float32)
    if t.ndim == 2:
        t = t[..., None]
    return a * (1 - t) + b * t


def flat(P, color):
    return np.ones((P.H, P.W, 3), np.float32) * np.asarray(color, np.float32)


def finish(P, out, grain: float = 0.0, seed: int = 0):
    """Clamp away pure black/white (palette safe range) and add optional grain."""
    if grain > 0:
        g = F.blur(np.random.default_rng(seed).normal(0, 1, out.shape[:2]).astype(np.float32), 0.6)
        out = out + (g * grain * 0.06)[..., None]
    return np.clip(out, 0.02, 0.985).astype(np.float32)


def grade(P, hue=None, chroma: float = 0.12, lo: float = 0.16, hi: float = 0.74, source_chroma: float = 0.5, tone=None, img=None):
    """Re-key a source into ONE hue family in OKLCH (the 'sunset rule', docs/preview-color.md):
    lightness from the (locally equalised) tone, chroma bounded and shaped by
    the source's own chroma, a gentle analogous hue drift with lightness, and
    desaturated darks as the calm counterpart. hue None = palette structure hue."""
    img = P.img if img is None else img
    t = P.t if tone is None else tone
    h0 = pal.rgb_to_oklch(P.pal.structure)[2] if hue is None else float(hue)
    lab = F.to_oklab(img)
    c0 = np.hypot(lab[..., 1], lab[..., 2])
    L = lo + (hi - lo) * np.clip(t, 0, 1) ** 1.25
    shape = F.smoothstep(0.10, 0.45, L) * (1 - 0.35 * F.smoothstep(0.78, 0.95, L))
    C = chroma * (0.25 + 0.75 * shape) * ((1 - source_chroma) + source_chroma * np.clip(c0 / 0.14, 0, 1.3))
    h = h0 + 28 * (L - 0.5) - 14 * (1 - shape)
    return F.oklch_field(L, C, h)
