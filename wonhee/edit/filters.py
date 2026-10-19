"""Filters: a look described in photographer's terms (exposure, contrast, fade, white balance, split toning, glow,
halation, light leak, ...) instead of copied from someone's photos. Deterministic numpy, no generation, no masks.

A `Filter` is plain numbers, so a person, a preset pack or a cheap model can write one, and it can be saved and
re-applied later (wonhee.filterstore). Point operations (tone, colour) run per pixel, so the same filter also
exports as a .cube LUT for phone and video editors; spatial ones (glow, grain, ...) only run on stills here.

Skin protection keeps skin-coloured pixels near their own colour while the rest of the frame takes a colour
grade. It is a colour-range rule, not a mask: nothing is cut out, so there are no halos.
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wonhee.edit import fx

Hue = Literal["none", "red", "orange", "yellow", "green", "teal", "cyan", "blue", "violet", "magenta", "pink"]
# CIELAB hue angle: 0 = +a* (red-magenta), 90 = +b* (yellow), 180 = green, 270 = blue
HUE_DEG = {"red": 25, "orange": 55, "yellow": 90, "green": 150, "teal": 195, "cyan": 225, "blue": 275,
           "violet": 305, "magenta": 335, "pink": 355}
LeakSide = Literal["left", "right", "top", "bottom", "corner"]
SKIN_DEG, SKIN_WIDTH = 52.0, 24.0  # skin of every tone sits in a narrow hue wedge around orange


class Filter(BaseModel):
    """0 is neutral for every number; ranges are where the result stays a photo."""
    model_config = ConfigDict(extra="forbid")
    exposure: float = Field(default=0.0, ge=-1, le=1, description="stops, -1 darker .. +1 brighter")
    contrast: float = Field(default=0.0, ge=-1, le=1, description="-1 flat .. +1 punchy (S-curve, soft toe)")
    shadows: float = Field(default=0.0, ge=-1, le=1, description="-1 deepen .. +1 open up the shadows")
    highlights: float = Field(default=0.0, ge=-1, le=1, description="-1 recover / calm .. +1 brighten highlights")
    fade: float = Field(default=0.0, ge=0, le=1, description="lifted, milky blacks (matte film)")
    rolloff: float = Field(default=0.0, ge=0, le=1, description="greyed-down whites (soft top end)")
    temperature: float = Field(default=0.0, ge=-1, le=1, description="-1 cool/blue .. +1 warm/amber")
    tint: float = Field(default=0.0, ge=-1, le=1, description="-1 green .. +1 magenta")
    saturation: float = Field(default=0.0, ge=-1, le=1, description="-1 grey .. +1 double colour")
    vibrance: float = Field(default=0.0, ge=-1, le=1, description="like saturation but mostly on muted colours")
    greens: float = Field(default=0.0, ge=-1, le=1, description="saturation of greens / foliage only")
    blues: float = Field(default=0.0, ge=-1, le=1, description="saturation of blues / sky only")
    mono: bool = Field(default=False, description="black and white (split toning still tints it)")
    shadow_hue: Hue = "none"
    shadow_tone: float = Field(default=0.0, ge=0, le=1, description="how strongly shadows take shadow_hue")
    highlight_hue: Hue = "none"
    highlight_tone: float = Field(default=0.0, ge=0, le=1, description="how strongly highlights take highlight_hue")
    skin_protect: float = Field(default=0.6, ge=0, le=1, description="keep skin near its own colour under a grade")
    clarity: float = Field(default=0.0, ge=-1, le=1, description="-1 soft, dreamy .. +1 crunchy local contrast")
    sharpen: float = Field(default=0.0, ge=0, le=1)
    glow: float = Field(default=0.0, ge=0, le=1, description="diffusion filter: the whole frame softly glows")
    bloom: float = Field(default=0.0, ge=0, le=1, description="glow only around the brightest lights")
    halation: float = Field(default=0.0, ge=0, le=1, description="red-orange film halo around bright edges")
    light_leak: float = Field(default=0.0, ge=0, le=1, description="film light leak washing in from one side")
    leak_hue: Hue = "orange"
    leak_side: LeakSide = "right"
    aberration: float = Field(default=0.0, ge=0, le=1, description="colour fringes toward the edges (cheap lens)")
    vignette: float = Field(default=0.0, ge=0, le=1)
    grain: float = Field(default=0.0, ge=0, le=1)


NUMERIC = [k for k, f in Filter.model_fields.items() if f.annotation is float and k != "skin_protect"]


def scaled(f: Filter, k: float) -> Filter:
    """The same filter at `k` intensity: every number moves toward neutral (k=0) or past the design (k up to 2)."""
    d = f.model_dump()
    for name in NUMERIC:
        lo, hi = _bounds(name)
        d[name] = float(np.clip(d[name] * k, lo, hi))
    return Filter(**d)


def _bounds(name: str) -> tuple[float, float]:
    meta = {type(m).__name__: m for m in Filter.model_fields[name].metadata}
    return float(meta["Ge"].ge), float(meta["Le"].le)


# --- tone ------------------------------------------------------------------------------------------------------

def _pchip(xs: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson) through the points: smooth, never overshoots, so no banding or reversal."""
    h = np.diff(xs)
    dlt = np.diff(ys) / h
    m = np.zeros_like(ys)
    m[0], m[-1] = dlt[0], dlt[-1]
    for i in range(1, len(xs) - 1):
        if dlt[i - 1] * dlt[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / dlt[i - 1] + w2 / dlt[i])
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    t = (x - xs[i]) / h[i]
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1]
            + (t3 - t2) * h[i] * m[i + 1])


def tone_curve(f: Filter, n: int = 1024) -> np.ndarray:
    """L (0..1) -> L, sampled at n points."""
    xs = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    ys = xs.copy()
    ys[1] += 0.05 * f.shadows
    ys[2] += 0.07 * f.shadows
    ys[4] += 0.06 * f.highlights
    ys[5] += 0.05 * f.highlights
    ys = np.clip(ys, 0, 1)
    ys = np.maximum.accumulate(ys + np.arange(len(ys)) * 1e-4)
    x = np.linspace(0, 1, n)
    y = _pchip(xs, ys, x)
    # S-curve y + c*y(1-y)(2y-1): slope 1+c/2 in the mids, 1-c at the ends, so monotone over the whole range
    y = y + f.contrast * y * (1 - y) * (2 * y - 1)
    y = np.clip(y, 0, 1)
    lo, hi = 0.12 * f.fade, 1 - 0.12 * f.rolloff
    return lo + (hi - lo) * y


# --- colour ----------------------------------------------------------------------------------------------------

def _hue_vec(name: str) -> tuple[float, float]:
    if name == "none":
        return 0.0, 0.0
    t = np.deg2rad(HUE_DEG[name])
    return float(np.cos(t)), float(np.sin(t))


def skin_weight(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """0..1: how much a pixel's colour looks like skin (hue wedge around orange, moderate chroma)."""
    c = np.hypot(a, b)
    ang = np.degrees(np.arctan2(b, a))
    dh = (ang - SKIN_DEG + 180) % 360 - 180
    return (np.exp(-(dh / SKIN_WIDTH) ** 2) * np.clip((c - 5) / 8, 0, 1) * np.clip((55 - c) / 15, 0, 1)
            ).astype(np.float32)


def point(bgr: np.ndarray, f: Filter) -> np.ndarray:
    """The per-pixel part of a filter on float32 BGR 0..1 (any shape HxWx3). Also what the .cube export samples."""
    x = np.clip(bgr, 0, 1).astype(np.float32)
    if f.exposure:
        x = np.clip((x ** 2.2 * 2 ** f.exposure) ** (1 / 2.2), 0, 1)
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0] / 100, lab[..., 1], lab[..., 2]
    a0, b0 = a.copy(), b.copy()
    curve = tone_curve(f)
    L = np.interp(L, np.linspace(0, 1, len(curve)), curve).astype(np.float32)
    c = np.hypot(a, b)
    gain = (1 + f.saturation) * (1 + f.vibrance * np.clip(1 - c / 40, 0, 1))
    if f.greens or f.blues:
        ang = np.degrees(np.arctan2(b, a))
        near = lambda deg, w: np.exp(-(((ang - deg + 180) % 360 - 180) / w) ** 2)  # noqa: E731
        gain = gain * (1 + f.greens * near(130, 35) + f.blues * near(262, 30))
    if f.mono:
        gain = gain * 0
    a, b = a * gain, b * gain
    a = a + 10 * f.tint + 2 * f.temperature
    b = b + 14 * f.temperature
    sh = np.clip((0.5 - L) / 0.35, 0, 1)
    hi = np.clip((L - 0.5) / 0.35, 0, 1)
    for hue, amt, w in ((f.shadow_hue, f.shadow_tone, sh), (f.highlight_hue, f.highlight_tone, hi)):
        va, vb = _hue_vec(hue)
        a, b = a + 20 * amt * w * va, b + 20 * amt * w * vb
    if f.skin_protect and not f.mono:
        # skin keeps its own hue but follows the brightness and saturation moves
        s = f.skin_protect * skin_weight(a0, b0)
        a, b = a * (1 - s) + a0 * gain * s, b * (1 - s) + b0 * gain * s
    out = np.stack([L * 100, a, b], axis=-1).astype(np.float32)
    return np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2BGR), 0, 1)


# --- spatial ---------------------------------------------------------------------------------------------------

def _screen(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    return 1 - (1 - base) * (1 - np.clip(top, 0, 1))


def _clarity(x: np.ndarray, k: float) -> np.ndarray:
    if not k:
        return x
    s = max(x.shape[:2]) / 60
    blur = cv2.GaussianBlur(x, (0, 0), s)
    if k > 0:
        return np.clip(x + 0.6 * k * (x - blur), 0, 1)
    return x * (1 + 0.6 * k) - blur * 0.6 * k  # k < 0: blend toward the blur


def _glow(x: np.ndarray, k: float) -> np.ndarray:
    if not k:
        return x
    soft = cv2.GaussianBlur(x, (0, 0), max(x.shape[:2]) / 45)
    return x * (1 - 0.35 * k) + _screen(x, soft * 0.8) * 0.35 * k


def _halation(x: np.ndarray, k: float) -> np.ndarray:
    if not k:
        return x
    lum = x.max(axis=2)
    hot = np.clip((lum - 0.7) / 0.3, 0, 1)
    halo = cv2.GaussianBlur(hot, (0, 0), max(x.shape[:2]) / 120)
    tint = np.array([0.08, 0.35, 1.0], np.float32)  # BGR: red-orange
    return _screen(x, halo[..., None] * tint * 0.9 * k)


def _leak(x: np.ndarray, k: float, hue: str, side: str) -> np.ndarray:
    if not k:
        return x
    h, w = x.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    u = {"left": 1 - xx / w, "right": xx / w, "top": 1 - yy / h, "bottom": yy / h,
         "corner": np.maximum(xx / w, 1 - yy / h) * 0.6 + (xx / w) * (1 - yy / h) * 0.4}[side]
    ramp = np.clip((u - 0.45) / 0.55, 0, 1) ** 1.6
    ramp = cv2.GaussianBlur(ramp, (0, 0), max(h, w) / 25)
    va, vb = _hue_vec(hue if hue != "none" else "orange")
    lab = np.array([[[78, 60 * va, 60 * vb]]], np.float32)
    colour = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0], 0, 1)
    return _screen(x, ramp[..., None] * colour * 0.85 * k)


def _aberration(x: np.ndarray, k: float) -> np.ndarray:
    if not k:
        return x
    h, w = x.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w / 2, h / 2
    out = x.copy()
    for ch, s in ((2, 1 + 0.006 * k), (0, 1 - 0.006 * k)):  # red out, blue in
        out[..., ch] = cv2.remap(x[..., ch], (xx - cx) / s + cx, (yy - cy) / s + cy, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    return out


def _sharpen(x: np.ndarray, k: float) -> np.ndarray:
    if not k:
        return x
    blur = cv2.GaussianBlur(x, (0, 0), max(1.0, max(x.shape[:2]) / 1500))
    return np.clip(x + 0.8 * k * (x - blur), 0, 1)


def apply(im: np.ndarray, f: Filter, intensity: float = 1.0, seed: int = 0) -> np.ndarray:
    """BGR uint8 in, BGR uint8 out."""
    if intensity != 1.0:
        f = scaled(f, intensity)
    x = point(im.astype(np.float32) / 255, f)
    x = _clarity(x, f.clarity)
    x = _sharpen(x, f.sharpen)
    x = _glow(x, f.glow)
    x = _halation(x, f.halation)
    x = _aberration(x, f.aberration)
    x = _leak(x, f.light_leak, f.leak_hue, f.leak_side)
    out = np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)
    out = fx.bloom(out, f.bloom)
    out = fx.vignette(out, f.vignette)
    return fx.grain(out, f.grain, seed=seed)


def cube(f: Filter, size: int = 33, title: str = "wonhee") -> str:
    """The point part as an Adobe/Resolve .cube 3D LUT (red varies fastest). Glow, grain etc. are not in it."""
    r = np.linspace(0, 1, size, dtype=np.float32)
    bb, gg, rr = np.meshgrid(r, r, r, indexing="ij")  # b outer, r inner
    grid = np.stack([bb, gg, rr], axis=-1).reshape(-1, 1, 3)  # BGR
    out = point(grid, f).reshape(-1, 3)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    lines += [f"{px[2]:.5f} {px[1]:.5f} {px[0]:.5f}" for px in out]
    return "\n".join(lines) + "\n"


# --- the pack --------------------------------------------------------------------------------------------------

def _f(**kw) -> Filter:
    return Filter(**kw)


PACK: dict[str, tuple[str, Filter]] = {
    "clean-bright": ("airy and clean: brighter, open shadows, a touch warm, gentle colour",
                     _f(exposure=0.25, shadows=0.4, highlights=-0.2, contrast=-0.1, temperature=0.1, vibrance=0.2)),
    "film-warm": ("warm negative film: soft contrast, milky blacks, amber highlights, red halation, fine grain",
                  _f(contrast=-0.2, fade=0.35, rolloff=0.2, temperature=0.25, highlight_hue="orange",
                     highlight_tone=0.25, saturation=-0.1, halation=0.25, grain=0.3)),
    "film-cool": ("cool slide-ish film: teal shadows, clean whites, a little grain",
                  _f(contrast=0.15, fade=0.15, temperature=-0.15, shadow_hue="teal", shadow_tone=0.4,
                     saturation=-0.1, grain=0.25)),
    "matte": ("matte magazine: flat, faded blacks, greyed whites, muted colour",
              _f(contrast=-0.35, fade=0.6, rolloff=0.35, saturation=-0.3, clarity=0.1)),
    "digicam": ("2000s point-and-shoot: punchy, cool whites, crunchy, colour fringes at the edges",
                _f(contrast=0.35, exposure=0.1, temperature=-0.15, saturation=0.15, sharpen=0.6, aberration=0.5,
                   vignette=0.15, grain=0.1)),
    "disposable": ("disposable camera: warm-green cast, dark corners, light leak, heavy grain",
                   _f(contrast=0.2, fade=0.2, temperature=0.2, tint=-0.15, saturation=0.1, light_leak=0.7,
                      leak_side="corner", vignette=0.5, grain=0.5, aberration=0.3)),
    "golden-hour": ("late sun: warm everywhere, orange highlights, soft glow",
                    _f(temperature=0.6, highlight_hue="orange", highlight_tone=0.55, shadows=0.15, glow=0.3,
                       vibrance=0.2, rolloff=0.1)),
    "dreamy": ("soft and hazy: diffusion glow, lifted blacks, pastel colour, a pink touch",
               _f(glow=0.6, clarity=-0.4, fade=0.3, contrast=-0.25, saturation=-0.15, highlight_hue="pink",
                  highlight_tone=0.25, exposure=0.1)),
    "pastel-pink": ("bright pastel party: high-key, pink-warm everywhere, low saturation",
                    _f(exposure=0.3, shadows=0.4, fade=0.25, tint=0.3, temperature=0.15, saturation=-0.25,
                       highlight_hue="pink", highlight_tone=0.3, glow=0.15)),
    "night-neon": ("city night: deep blue-teal shadows, magenta highlights, glowing lights",
                   _f(exposure=-0.3, contrast=0.25, shadows=-0.2, shadow_hue="cyan", shadow_tone=0.6,
                      highlight_hue="magenta", highlight_tone=0.35, bloom=0.5, vignette=0.4, grain=0.15)),
    "teal-orange": ("blockbuster: teal shadows, warm skin and highlights, punchy",
                    _f(contrast=0.3, shadow_hue="teal", shadow_tone=0.55, highlight_hue="orange",
                       highlight_tone=0.3, vibrance=0.2, clarity=0.2)),
    "mono-grit": ("black and white, hard contrast, heavy grain",
                  _f(mono=True, contrast=0.55, clarity=0.4, grain=0.55, vignette=0.3)),
    "mono-soft": ("quiet black and white: soft contrast, faded, a hint of warm paper",
                  _f(mono=True, contrast=-0.15, fade=0.3, highlight_hue="orange", highlight_tone=0.15, grain=0.2)),
}


# --- checks a reviewer (and the guard) reads -------------------------------------------------------------------

def checks(src: np.ndarray, out: np.ndarray) -> dict:
    """Before/after numbers for one photo: clipped highlights and crushed shadows (% of pixels), median brightness,
    colourfulness, and how far skin-coloured pixels drifted in hue (degrees)."""
    a, b = (cv2.cvtColor(cv2.resize(x, (480, round(480 * x.shape[0] / x.shape[1]))).astype(np.float32) / 255,
                         cv2.COLOR_BGR2LAB) for x in (src, out))
    sk = skin_weight(a[..., 1], a[..., 2]) > 0.5
    drift = 0.0
    if sk.sum() > 100:
        ha = np.degrees(np.arctan2(a[..., 2][sk], a[..., 1][sk]))
        hb = np.degrees(np.arctan2(b[..., 2][sk], b[..., 1][sk]))
        drift = float(np.median(np.abs((hb - ha + 180) % 360 - 180)))
    pct = lambda m: round(float(m.mean()) * 100, 1)  # noqa: E731
    return {"clipped": [pct(a[..., 0] > 98), pct(b[..., 0] > 98)],
            "crushed": [pct(a[..., 0] < 2), pct(b[..., 0] < 2)],
            "median_L": [round(float(np.median(a[..., 0])), 1), round(float(np.median(b[..., 0])), 1)],
            "chroma": [round(float(np.hypot(a[..., 1], a[..., 2]).mean()), 1),
                       round(float(np.hypot(b[..., 1], b[..., 2]).mean()), 1)],
            "skin_hue_drift": round(drift, 1)}


def guard(f: Filter, results: list[dict]) -> tuple[Filter, list[str]]:
    """Fix what the numbers say is broken, without asking a model: newly blown highlights pull exposure and
    highlights down; newly crushed shadows open the shadows. Returns (filter, what it changed)."""
    d, notes = f.model_dump(), []
    if any(r["clipped"][1] > max(r["clipped"][0] + 3, 4) for r in results):
        d["exposure"] = max(-1.0, d["exposure"] - 0.15)
        d["highlights"] = max(-1.0, min(d["highlights"], 0.0) - 0.2)
        notes.append("highlights were blowing out: exposure -0.15, highlights -0.2")
    if any(r["crushed"][1] > max(r["crushed"][0] + 5, 8) for r in results):
        d["shadows"] = min(1.0, d["shadows"] + 0.2)
        notes.append("shadows were crushing: shadows +0.2")
    return Filter(**d), notes


def verdicts(c: dict) -> str:
    """One photo's checks as plain words: small models misread raw numbers (a 3 degree drift is not 'high')."""
    out = []
    cb, ca = c["clipped"]
    out.append(f"highlights blowing out ({ca:g}% clipped, was {cb:g}%)" if ca > max(cb + 2, 3) else "highlights fine")
    sb, sa = c["crushed"]
    out.append(f"shadows crushing ({sa:g}% pure black, was {sb:g}%)" if sa > max(sb + 4, 6) else "shadows fine")
    lb, la = c["median_L"]
    dl = la - lb
    out.append("brightness about the same" if abs(dl) < 3 else f"{'brighter' if dl > 0 else 'darker'} by {abs(dl):.0f}"
               " (of 100)")
    hb, ha = c["chroma"]
    dc = (ha - hb) / max(hb, 1) * 100
    out.append("colour about as strong" if abs(dc) < 10 else f"{'more' if dc > 0 else 'less'} colourful by "
               f"{abs(dc):.0f}%")
    out.append(f"skin hue shifted {c['skin_hue_drift']:g} degrees" if c["skin_hue_drift"] > 8 else "skin colour natural")
    return ", ".join(out)
