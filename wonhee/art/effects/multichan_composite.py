"""multichan-composite, multichan-fluorescence, multichan-relief, multichan-diff, multichan-channel-mix:
multichannel / scientific composites.

Inputs (ctx.data): a dict of 2D arrays {name: field} (or a list / C x H x W array) = real scalar channels. Each is
normalised with an EXPLICIT range (param ``ranges``: {"name": [lo, hi]} or a list in channel order) or, if none, the
1..99 percentile range. A ctx.source RGB image is accepted by multichan-channel-mix (and by composite when there is
no data) as AESTHETIC channels (R/G/B, luma, saturation, edges, shadow/highlight masks); the result is labelled
aesthetic and must never be described as infrared, satellite or measured data. With neither, stand-in channels
(luminance of library physics effects) are used and flagged ``synthetic`` / ``standin``.

Independent controls: colour (mood, hues, blend, opacity), structure (transforms, contrast, levels), source (data,
which, ranges), surface (glow, grain, halation). Metadata: ``channel_meta(effect, data, source)``."""
from __future__ import annotations

import numpy as np

from .. import datakit as dk
from .. import mckit as mk
from ..registry import effect

MOODS = {
    "scientific": dict(hues=[205, 35, 135, 320], glow=0.35, gamma=1.0, blend="screen"),
    "alien":      dict(hues=[300, 165, 345, 75], glow=0.6, gamma=1.15, blend="add"),
    "luminous":   dict(hues=[60, 22, 335, 200], glow=0.9, gamma=1.2, blend="add"),
    "analytical": dict(hues=[255, 25, 150, 60], glow=0.0, gamma=1.0, blend="multiply"),
}

COLOUR = {
    "mood": ("scientific", "scientific | alien | luminous | analytical: sets the channel hue family, default blend, glow"),
    "hues": (None, "explicit hue per channel (degrees, list); default = mood hues rotated with the palette hue"),
    "blend": ("auto", "auto | screen | add | max | multiply | over (auto: multiply on light palettes, else the mood's)"),
    "opacity": (0.9, "channel opacity: one number or a list per channel"),
    "gamma": (None, "per-channel tone gamma (>1 darker mids); default from the mood"),
    "floor": (None, "threshold below which a channel is transparent (sparsity: keeps the base visible); default 0.3 for stand-in / aesthetic channels, 0 for real data"),
}
SOURCE = {
    "channels": (3, "number of channels to use / stand-ins to create"),
    "which": (None, "channel names to use (dict keys, derived names, or stand-in effect names)"),
    "ranges": (None, "explicit normalisation range per channel: {'name': [lo, hi]} or a list in channel order"),
    "standin_smooth": ("16,8,5", "low-pass of stand-in fields per channel, px at 640 wide (bigger = broader structures); ignored for real data"),
}
SURFACE = {
    "glow": (None, "bloom amount (default from the mood)"),
    "grain": (0.35, "film/sensor grain amount"),
}


def channel_meta(effect_name: str, data=None, source=None) -> dict:
    """What a viewer / alt text should be told: stand-in (synthetic) vs real channels vs aesthetic RGB derivation."""
    kind = "data" if data is not None else ("aesthetic" if source is not None else "standin")
    return {"effect": effect_name, "synthetic": kind == "standin", "standin": kind == "standin", "aesthetic": kind == "aesthetic",
            "note": {"data": "channels from ctx.data, explicit or percentile ranges",
                     "aesthetic": "RGB-derived aesthetic channels; not infrared/satellite/measured data",
                     "standin": "stand-in physics fields; decorative, not measurements"}[kind]}


def _smooths(ctx):
    v = _p(ctx, "standin_smooth")
    return [float(x) for x in str(v).split(",")] if isinstance(v, str) else float(v)


def _p(ctx, k):
    return ctx.params.get(k)


def _opac(ctx, n):
    o = _p(ctx, "opacity")
    return [float(x) for x in (o if isinstance(o, (list, tuple)) else [o] * n)] + [1.0] * n


def _mood(ctx):
    return MOODS[str(_p(ctx, "mood"))]


def _hues(ctx, n):
    m = _mood(ctx)
    hs = _p(ctx, "hues") or mk.hue_set(ctx, m["hues"])
    return [float(hs[i % len(hs)]) for i in range(n)]


def _blend_mode(ctx):
    b = str(_p(ctx, "blend"))
    if b != "auto":
        return b
    return "multiply" if ctx.palette.polarity == "light" else _mood(ctx)["blend"] if _mood(ctx)["blend"] != "multiply" else "screen"


def _floor(ctx, kind):
    f = _p(ctx, "floor")
    return float(f) if f is not None else (0.0 if kind == "data" else 0.3)


def compose_channels(ctx, chans, f_list=None, kind="data"):
    """Tint each channel and blend onto the palette canvas. Returns float image."""
    W, H = ctx.size
    n = len(chans)
    mode = _blend_mode(ctx)
    light = ctx.palette.polarity == "light"
    hs = _hues(ctx, n)
    gm = _p(ctx, "gamma"); gm = _mood(ctx)["gamma"] if gm is None else gm
    gms = [float(x) for x in (gm if isinstance(gm, (list, tuple)) else [gm] * n)] + [1.0] * n
    op = _opac(ctx, n)
    base = np.empty((H, W, 3), np.float32); base[:] = ctx.palette.canvas
    layers = []
    tmode = "ink" if (mode == "multiply" or light) else "emit"
    for i, c in enumerate(chans):
        f0 = f_list[i] if f_list is not None else c.f
        fl_ = _floor(ctx, kind)
        f0 = np.clip((f0 - fl_) / (1 - fl_), 0, 1) if fl_ > 0 else f0
        f = f0 ** gms[i]
        tab = mk.tint_lut(round(hs[i], 1), tmode)
        lay = dk.apply_lut(f, tab)
        if tmode == "emit":
            lay = lay - tab[0]
        layers.append(lay)
    if tmode == "ink" and mode != "multiply":
        mode = "multiply"
    img = mk.blend_stack(base, layers, op, mode)
    if mode in ("add", "screen") and tmode == "emit":
        img = mk.soft_clip(img)
    return img


def finish(ctx, img, rng):
    g = _p(ctx, "glow"); g = _mood(ctx)["glow"] if g is None else g
    s = dk.scale_px(ctx.size[0])
    if ctx.palette.polarity == "dark":
        img = mk.glow(img, float(g), 9 * s + 1, 0.5)
    img = mk.grain(img, float(_p(ctx, "grain")), rng, 0.8 * s + 0.2)
    lo = 0.02 if ctx.palette.polarity == "dark" else 0.05
    return np.clip(img, lo, 0.975).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------
@effect("multichan-composite", family="multichan", tags=["multichannel", "scientific", "luminous", "alien", "analytical", "composite", "data", "all-over", "synthetic-default", "dark-light"],
        params={**COLOUR, **SOURCE, **SURFACE, "contrast": (0.25, "local-contrast (unsharp) on every channel, structure"), "scale_px": (6.0, "contrast scale px at 1500 wide"),
                "transforms": ("raw,ridge,raw", "structure per channel: raw | ridge | blobs | haze | dots | edges (comma list)")})
def composite(ctx):
    """Several scalar channels, each normalised with an explicit range and given its own hue, blended with
    controlled opacity on the palette canvas (screen / add on dark, multiply on light). Real data: a dict of 2D
    arrays; an RGB source becomes aesthetic channels. Stand-ins (physics fields) otherwise (flagged)."""
    rng = ctx.rng
    it = mk.intake(ctx, int(_p(ctx, "channels")), _p(ctx, "which"), _p(ctx, "ranges"), smooth=_smooths(ctx))
    s = dk.scale_px(ctx.size[0])
    tr = str(_p(ctx, "transforms")).split(",")
    fl = [mk.transform(mk.local_contrast(c.f, float(_p(ctx, "scale_px")) * s + 0.5, float(_p(ctx, "contrast"))), tr[i % len(tr)].strip(), 1.0, 1.6 * s + 0.4) for i, c in enumerate(it.chans)]
    img = compose_channels(ctx, it.chans, fl, it.kind)
    return finish(ctx, img, rng)


@effect("multichan-fluorescence", family="multichan", tags=["multichannel", "scientific", "luminous", "alien", "microscopy", "additive", "all-over", "synthetic-default", "dark"],
        params={**COLOUR, **SOURCE, **SURFACE, "mood": ("alien", COLOUR["mood"][1]), "channels": (4, SOURCE["channels"][1]), "blend": ("add", COLOUR["blend"][1]), "gamma": (1.5, COLOUR["gamma"][1]), "opacity": ([0.8, 0.75, 0.55, 0.5], COLOUR["opacity"][1]),
                "transforms": ("blobs,ridge,dots,haze", "structure per channel: raw | ridge (filaments) | blobs (nuclei-like discs) | haze | dots | edges (comma list)"),
                "sigma": (2.0, "structure scale at 1500 wide (px)"), "haze": (0.15, "overall dim haze from the first channel")})
def fluorescence(ctx):
    """Fluorescence-microscopy look: every channel is a separate 'label' with its own structure (blobs, filaments,
    speckle, haze) glowing additively on a dark ground. The transforms apply to real channels too (e.g. ridge-filter
    a measured channel). Dark palettes only make sense (add blend); light palettes fall back to ink multiply."""
    rng = ctx.rng; s = dk.scale_px(ctx.size[0])
    it = mk.intake(ctx, int(_p(ctx, "channels")), _p(ctx, "which"), _p(ctx, "ranges"),
                   standin_which=["noise:16", "noise:42", "noise:9", "noise:90"][: int(_p(ctx, "channels"))], smooth=0.0)
    tr = str(_p(ctx, "transforms")).split(",")
    fl = []
    for i, c in enumerate(it.chans):
        f = mk.transform(c.f, tr[i % len(tr)].strip(), 1.0, float(_p(ctx, "sigma")) * s + 0.4)
        fl.append(np.clip(f, 0, 1))
    img = compose_channels(ctx, it.chans, fl, "data")
    hz = float(_p(ctx, "haze"))
    if hz > 0 and ctx.palette.polarity == "dark":
        hh = mk.transform(it.chans[0].f, "haze", 1.0, float(_p(ctx, "sigma")) * s + 0.4)
        tab = mk.tint_lut(round(_hues(ctx, 1)[0], 1), "emit")
        img = img + (dk.apply_lut(hh, tab) - tab[0]) * hz * 0.5
    return finish(ctx, img, rng)


@effect("multichan-relief", family="multichan", tags=["multichannel", "scientific", "analytical", "terrain", "hillshade", "contours", "all-over", "synthetic-default", "dark-light"],
        params={"scale": ("auto", "colour scale for the value channel (any datakit scale name, 'auto' keys on the palette)"),
                "invert": (False, "flip the scale"), "ranges": (None, "explicit [lo, hi] per channel, order: height, value"),
                "fields": ("physics-vortex,physics-wavemedium", "stand-in effects for height,value (when no data)"),
                "smooth": (1.0, "extra height smoothing px at 1500 wide"),
                "az": (315.0, "light azimuth, degrees"), "alt": (38.0, "light altitude, degrees"), "z": (5.0, "vertical exaggeration"),
                "shade": (0.85, "shading strength"), "levels": (16, "contour levels (0 = none)"), "contour": (0.7, "contour line strength"),
                "line_w": (1.6, "contour width px at 1500 wide"), "value_mix": (0.8, "how much of the colour channel shows (0 = pure relief)"),
                "standin_smooth": ("10", "stand-in low-pass, px at 640 wide (one value, or per channel)"), "grain": (0.25, "grain")})
def relief(ctx):
    """Two real channels read as a map: channel 1 is HEIGHT (hillshade + contour lines), channel 2 is VALUE (colour
    scale draped over the shading). Any gridded pair (elevation + temperature, concentration + gradient). Stand-ins
    otherwise. Light and dark polarity follow the palette."""
    rng = ctx.rng; s = dk.scale_px(ctx.size[0])
    names = str(_p(ctx, "fields")).split(",")
    if ctx.data is None:
        it = mk.intake(ctx, 2, None, _p(ctx, "ranges"), standin_which=[n.strip() for n in names[:2]], smooth=_smooths(ctx))
    else:
        it = mk.intake(ctx, 2, None, _p(ctx, "ranges"))
    h = it.chans[0].f
    v = it.chans[1].f if len(it.chans) > 1 else it.chans[0].f
    sm = float(_p(ctx, "smooth")) * s
    h = mk.cv2.GaussianBlur(h, (0, 0), max(0.1, sm)) if sm > 0 else h
    h = dk.norm_range(h, None, (1, 99))[0]
    sh = mk.hillshade(h, float(_p(ctx, "az")), float(_p(ctx, "alt")), float(_p(ctx, "z")))
    sh = (sh - np.percentile(sh, 2)) / (np.percentile(sh, 98) - np.percentile(sh, 2) + 1e-6)
    tab = dk.lut(_p(ctx, "scale"), ctx.palette)
    if _p(ctx, "invert"):
        tab = tab[::-1].copy()
    col = dk.apply_lut(np.clip(v, 0, 1), tab)
    dark = ctx.palette.polarity == "dark"
    ground = ctx.palette.canvas + (ctx.palette.structure - ctx.palette.canvas) * 0.25
    vm = float(_p(ctx, "value_mix"))
    base = ground + (col - ground) * vm
    strength = float(_p(ctx, "shade"))
    light = (1 - strength * 0.5) + strength * (np.clip(sh, 0, 1) * 0.9 + 0.1) if True else 1
    img = base * np.clip(light, 0.15, 1.45)[..., None]
    nl = int(_p(ctx, "levels"))
    if nl > 0:
        cm = mk.contour_mask(h, nl, float(_p(ctx, "line_w")) * s + 0.3, 4, 0.9) * float(_p(ctx, "contour"))
        ink = np.array([0.96, 0.94, 0.9]) * 0.92 if dark else np.array([0.12, 0.1, 0.14])
        img = img * (1 - cm[..., None]) + ink * cm[..., None]
    img = mk.grain(img, float(_p(ctx, "grain")), rng, 0.8 * s + 0.2)
    return np.clip(img, 0.03, 0.97).astype(np.float32)


@effect("multichan-diff", family="multichan", tags=["multichannel", "scientific", "analytical", "difference", "diverging", "change", "all-over", "synthetic-default", "dark-light"],
        params={"scale": ("blue-red", "diverging scale (blue-red | teal-rose | violet-gold | green-plum)"),
                "ranges": (None, "explicit [lo, hi] for channels A and B (same units; the difference is symmetric about 0)"),
                "under": (0.55, "how much the combined brightness (A+B) modulates lightness under the diverging colour (0 = flat colour)"),
                "zero_line": (0.8, "strength of the zero-difference contour"), "line_w": (1.8, "contour width px at 1500 wide"),
                "fields": ("physics-vortex,physics-wavemedium", "stand-in effects for A,B (when no data)"),
                "grain": (0.25, "grain"), "smooth": (1.5, "pre-smoothing px"), "standin_smooth": ("10", "stand-in low-pass, px at 640 wide")})
def diff(ctx):
    """Change detection between two channels: A - B in a diverging scale (centre = no change), lightness carries
    the overall magnitude (A+B), the zero-difference boundary is traced as a thin line. Real inputs: any pair of
    gridded arrays with the same units (before / after, model / observation)."""
    rng = ctx.rng; s = dk.scale_px(ctx.size[0])
    names = [n.strip() for n in str(_p(ctx, "fields")).split(",")]
    it = mk.intake(ctx, 2, None, _p(ctx, "ranges"), standin_which=names[:2], smooth=_smooths(ctx))
    A, B = it.chans[0].f, it.chans[1].f
    sm = float(_p(ctx, "smooth")) * s
    if sm > 0:
        A = mk.cv2.GaussianBlur(A, (0, 0), sm); B = mk.cv2.GaussianBlur(B, (0, 0), sm)
    d = A - B
    a = np.percentile(np.abs(d), 98) + 1e-6
    t = np.clip(0.5 + 0.5 * d / a, 0, 1)
    tab = dk.lut(_p(ctx, "scale"), ctx.palette)
    img = dk.apply_lut(t, tab)
    u = float(_p(ctx, "under"))
    mag = np.clip((A + B) / 2, 0, 1)
    if u > 0:
        mod = (1 - u * 0.5) + u * (0.35 + 0.9 * mag)
        img = img * mod[..., None]
    zl = float(_p(ctx, "zero_line"))
    if zl > 0:
        gy, gx = np.gradient(mk.cv2.GaussianBlur(d, (0, 0), 1.0))
        dist = np.abs(d) / (np.hypot(gx, gy) + 1e-5)
        w = float(_p(ctx, "line_w")) * s + 0.4
        m = np.clip(w / 2 + 0.5 - dist, 0, 1) * zl
        ink = np.array([0.95, 0.94, 0.92]) if ctx.palette.polarity == "dark" else np.array([0.1, 0.1, 0.13])
        img = img * (1 - m[..., None]) + ink * m[..., None]
    img = mk.grain(img, float(_p(ctx, "grain")), rng, 0.8 * s + 0.2)
    return np.clip(img, 0.03, 0.97).astype(np.float32)


@effect("multichan-channel-mix", family="multichan", tags=["multichannel", "aesthetic", "false-colour", "photo", "print", "luminous", "analytical", "dark-light"],
        params={**COLOUR, **SURFACE, "mood": ("luminous", COLOUR["mood"][1]), "blend": ("screen", COLOUR["blend"][1]),
                "channels": ("shadows,luma,edges", "derived channels (comma list): red green blue luma saturation edges shadows highlights warmth detail; or data keys"),
                "ranges": SOURCE["ranges"],
                "shift": (0.004, "misregistration between channel layers, fraction of width (0 = registered)"),
                "levels": (0, "posterise each channel to this many levels (0 = continuous)"),
                "halation": (0.3, "warm bloom around bright channels"), "contrast": (0.3, "local contrast")})
def channel_mix(ctx):
    """AESTHETIC false-colour from a photo: derive 2-4 channels (shadow mask, luma, edges, saturation, ...), give each
    its own hue, then offset them slightly (misregistration) and optionally posterise so the result reads as
    riso / multi-plate / duotone-plus print. Labelled aesthetic: it is NOT infrared or satellite data. ctx.source
    keeps its subject (people ok); with ctx.data the dict channels are used instead; with neither, a synthetic
    source is used (flagged synthetic)."""
    rng = ctx.rng; W, H = ctx.size; s = dk.scale_px(W)
    src = ctx.source
    if ctx.data is None and src is None:
        from ..features import synthetic_source
        src = synthetic_source(ctx.size, ctx.seed)
    class _C:   # intake wants ctx.source; pass through the (possibly synthetic) source
        pass
    from ..registry import Ctx
    c2 = Ctx(size=ctx.size, seed=ctx.seed, palette=ctx.palette, params=ctx.params, source=src, data=ctx.data)
    names = [x.strip() for x in str(_p(ctx, "channels")).split(",")]
    it = mk.intake(c2, len(names), names, _p(ctx, "ranges"), derived=names)
    lv = int(_p(ctx, "levels"))
    fl = []
    for i, c in enumerate(it.chans):
        f = mk.local_contrast(c.f, 6 * s + 0.5, float(_p(ctx, "contrast")))
        if lv > 1:
            f = np.round(f * (lv - 1)) / (lv - 1)
        sh = float(_p(ctx, "shift")) * W * (i - (len(it.chans) - 1) / 2)
        if abs(sh) > 0.2:
            f = np.roll(f, int(round(sh)), axis=1)
            f = np.roll(f, int(round(sh * 0.5)), axis=0)
        fl.append(f)
    img = compose_channels(ctx, it.chans, fl, it.kind)
    ha = float(_p(ctx, "halation"))
    if ha > 0 and ctx.palette.polarity == "dark":
        img = mk.glow(img, ha, 12 * s + 1, 0.45, tint=(1.0, 0.55, 0.4))
    return finish(ctx, img, rng)
