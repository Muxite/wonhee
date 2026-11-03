"""OKLCH role palettes (design: docs/DESIGN.md "Project art", docs/art-reference/METHODS.md).

Contrast comes from lightness per role, never from hue, so any hue family
works. Roles: canvas, structure, structure2 (3-colour palettes), accent, ink;
derived: edge (structure pulled toward the canvas), band (canvas toward structure).
All colours are float RGB triples in 0..1 (sRGB).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def oklch_to_rgb(L: float, C: float, h_deg: float) -> np.ndarray:
    """OKLCH -> sRGB (0..1). Chroma is reduced until the colour is in gamut."""
    def conv(c):
        a, b = c * math.cos(math.radians(h_deg)), c * math.sin(math.radians(h_deg))
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
        return np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                         -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                         -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])
    c = C
    for _ in range(40):
        lin = conv(c)
        if lin.min() >= -1e-4 and lin.max() <= 1 + 1e-4:
            break
        c *= 0.93
    lin = np.clip(conv(c), 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055).astype(np.float32)


def rgb_to_oklch(rgb) -> tuple[float, float, float]:
    r, g, b = (np.where(np.asarray(rgb) <= 0.04045, np.asarray(rgb) / 12.92, ((np.asarray(rgb) + 0.055) / 1.055) ** 2.4))
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    A = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    B = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return float(L), float(math.hypot(A, B)), float(math.degrees(math.atan2(B, A)) % 360)


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], dtype=np.float32)


def rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(float(np.clip(v, 0, 1)) * 255)):02x}" for v in rgb)


@dataclass
class Palette:
    name: str
    canvas: np.ndarray
    structure: np.ndarray
    accent: np.ndarray
    structure2: np.ndarray | None = None
    ink: np.ndarray | None = None
    polarity: str = "dark"

    @property
    def edge(self) -> np.ndarray:
        return self.structure + (self.canvas - self.structure) * 0.30

    @property
    def band(self) -> np.ndarray:
        return self.canvas + (self.structure - self.canvas) * 0.10

    def mix(self, a: str, b: str, t: float) -> np.ndarray:
        A, B = getattr(self, a), getattr(self, b)
        return A + (B - A) * t

    def ramp(self, n: int = 256, stops=("canvas", "structure", "accent")) -> np.ndarray:
        """Piecewise-linear colour ramp through role colours, n x 3."""
        cols = [getattr(self, s) if getattr(self, s) is not None else self.structure for s in stops]
        out = np.zeros((n, 3), np.float32)
        t = np.linspace(0, len(cols) - 1, n)
        i = np.clip(t.astype(int), 0, len(cols) - 2)
        f = (t - i)[:, None]
        for k in range(n):
            out[k] = cols[i[k]] * (1 - f[k]) + cols[i[k] + 1] * f[k]
        return out

    def hexes(self) -> dict[str, str]:
        d = {"canvas": self.canvas, "structure": self.structure, "accent": self.accent}
        if self.structure2 is not None:
            d["structure2"] = self.structure2
        return {k: rgb_to_hex(v) for k, v in d.items()}


def make(hue: float, *, polarity: str = "dark", harmony: str = "analogous", three: bool = False, name: str | None = None) -> Palette:
    """Build a palette from a key hue. Lightness per role follows the measured
    ranges in METHODS.md: dark canvas L .13-.16, structure .47-.62, accent
    .74-.90; light canvas .95-.97, structure .40-.50, accent .50-.62."""
    if harmony == "mono":
        ah = hue
    elif harmony == "complement":
        ah = (hue + 180) % 360
    elif harmony == "splittone":
        ah = (hue + 150) % 360
    else:
        ah = (hue + 30) % 360
    if polarity == "light":
        canvas, structure = oklch_to_rgb(0.96, 0.02, hue), oklch_to_rgb(0.45, 0.14, hue)
        accent, s2 = oklch_to_rgb(0.56, 0.22, ah), oklch_to_rgb(0.62, 0.12, (hue + 40) % 360)
    else:
        canvas, structure = oklch_to_rgb(0.15, 0.03, hue), oklch_to_rgb(0.56, 0.15, hue)
        accent, s2 = oklch_to_rgb(0.84, 0.17, ah), oklch_to_rgb(0.66, 0.13, (hue + 40) % 360)
    return Palette(name or f"{polarity}-{int(hue)}-{harmony}", canvas, structure, accent, s2 if three else None, None, polarity)


_NAMED = {
    "ember": lambda: make(40, harmony="analogous", name="ember"),
    "ultramarine": lambda: make(265, harmony="complement", name="ultramarine"),
    "teal-coral": lambda: make(190, harmony="complement", name="teal-coral"),
    "violet-dusk": lambda: make(300, harmony="analogous", three=True, name="violet-dusk"),
    "green-gold": lambda: make(145, harmony="analogous", name="green-gold"),
    "ink-on-paper": lambda: make(255, polarity="light", harmony="mono", name="ink-on-paper"),
    "red-on-cream": lambda: make(28, polarity="light", harmony="mono", name="red-on-cream"),
}


def named(name: str) -> Palette:
    if name not in _NAMED:
        raise KeyError(f"unknown palette {name!r}; known: {', '.join(sorted(_NAMED))}")
    return _NAMED[name]()


def names() -> list[str]:
    return sorted(_NAMED)
