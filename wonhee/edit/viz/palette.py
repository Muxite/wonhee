"""Colour roles for project art, built in OKLCH so lightness (and so contrast) holds across hues.

RGB and HSL lightness drift with hue (a green and a blue at the same HSL L differ a lot in how bright they look),
which is why art coloured from a name hash clashed. Here the lightness of every role is fixed first and only the hue
changes, so contrast against the canvas is the same for any project. The numbers are DESIGN.md › Project art › Colour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Polarity = Literal["light", "dark"]
Harmony = Literal["mono", "analogous", "complementary", "split"]

# Key hue by what the project is, not by its name (the meaning people already read into colour).
DOMAIN_HUE = {"compute": 240.0, "life": 128.0, "agent": 60.0, "diagnostic": 12.0, "editorial": 88.0}
DOMAIN_CHROMA = {"editorial": .35}  # editorial pages stay near-neutral: bone, cream and ink

# L, C per role and polarity. Light accents sit lower than dark ones: L .65-.80 fails contrast on a L .96 canvas.
_SPEC = {
    "light": dict(canvas=(.96, .012), structure=(.42, .045), accent=(.55, .20), ink=(.20, .02)),
    "dark": dict(canvas=(.15, .015), structure=(.48, .050), accent=(.72, .21), ink=(.93, .010)),
}
EDGE_DIM = .30  # an edge is this much closer to the canvas than its vertices, so dense meshes do not blur


@dataclass(frozen=True)
class Roles:
    canvas: str
    structure: str
    edge: str
    accent: str
    ink: str


def _srgb(x: float) -> float:
    return 12.92 * x if x <= .0031308 else 1.055 * x ** (1 / 2.4) - .055


def _linear(x: float) -> float:
    return x / 12.92 if x <= .04045 else ((x + .055) / 1.055) ** 2.4


def _oklch_linear(L: float, C: float, h: float) -> tuple[float, float, float]:
    a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
    l_, m_, s_ = L + .3963377774 * a + .2158037573 * b, L - .1055613458 * a - .0638541728 * b, L - .0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (4.0767416621 * l - 3.3077115913 * m + .2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - .3413193965 * s,
            -.0041960863 * l - .7034186147 * m + 1.7076147010 * s)


def to_hex(L: float, C: float, h: float) -> str:
    """OKLCH to #rrggbb. Out-of-gamut colours lose chroma, never lightness, so the contrast rules stay true."""
    def inside(c: float) -> bool:
        return all(-1e-4 <= v <= 1 + 1e-4 for v in _oklch_linear(L, c, h))

    lo, hi = 0.0, C
    if not inside(C):
        for _ in range(24):  # bisect for the most chroma that still fits sRGB
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if inside(mid) else (lo, mid)
        C = lo
    r, g, b = (min(1.0, max(0.0, _srgb(max(0.0, v)))) for v in _oklch_linear(L, C, h))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def lightness(hex_color: str) -> float:
    """OKLab L of a #rrggbb colour (what the contrast checks compare)."""
    r, g, b = (_linear(int(hex_color[i:i + 2], 16) / 255) for i in (1, 3, 5))
    l = .4122214708 * r + .5363325363 * g + .0514459929 * b
    m = .2119034982 * r + .6806995451 * g + .1073969566 * b
    s = .0883024619 * r + .2817188376 * g + .6299787005 * b
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return .2104542553 * l_ + .7936177850 * m_ - .0040720468 * s_


def accent_hue(hue: float, harmony: Harmony) -> float:
    return {"mono": hue, "analogous": hue + 30, "complementary": hue + 180, "split": 50.0}[harmony] % 360


def roles(hue: float, polarity: Polarity, harmony: Harmony = "mono", chroma_scale: float = 1.0) -> Roles:
    """The five colours of a piece. `split` is the dark glow look: slate canvas, accent at orange."""
    s = _SPEC[polarity]
    base_h = 240.0 if harmony == "split" else hue  # split tones tint the neutrals cool whatever the accent is
    cs = chroma_scale
    canvas = to_hex(s["canvas"][0], s["canvas"][1] * cs, base_h)
    structure_L, structure_C = s["structure"]
    canvas_L = s["canvas"][0]
    edge_L = canvas_L + (structure_L - canvas_L) * (1 - EDGE_DIM)
    return Roles(
        canvas=canvas,
        structure=to_hex(structure_L, structure_C * cs, base_h),
        edge=to_hex(edge_L, structure_C * cs, base_h),
        accent=to_hex(s["accent"][0], s["accent"][1] * (.5 + .5 * cs), accent_hue(hue, harmony)),
        ink=to_hex(s["ink"][0], s["ink"][1] * cs, base_h),
    )


def for_domain(domain: str, polarity: Polarity, harmony: Harmony = "mono") -> Roles:
    if domain not in DOMAIN_HUE:
        raise ValueError(f"unknown art domain {domain!r}; use one of {sorted(DOMAIN_HUE)}")
    return roles(DOMAIN_HUE[domain], polarity, harmony, DOMAIN_CHROMA.get(domain, 1.0))
