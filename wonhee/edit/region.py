"""Region-limited effects: run a filter, look or viz effect on part of a picture and leave the rest untouched.

The effects in fx.py and filters.py act on the whole frame. A title slide needs the opposite: blur and darken only
where the words sit so they read, keep the food sharp and bright below. A `Region` is a soft mask built from plain
geometry (no model, so it runs anywhere); `apply_layers` blends each effect through its region, in order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from wonhee.edit import filters, fx

Shape = Literal["rect", "ellipse", "ramp"]
EFFECTS = ("blur", "darken", "mono", "bloom", "vignette", "grain", "halftone", "duotone", "look")


@dataclass(frozen=True)
class Region:
    """Where an effect applies, in fractions of the frame.

    rect / ellipse: the box (x0, y0, x1, y1), softened by `feather` (a fraction of the shorter side).
    ramp: full strength at y0, none at y1 (y0 > y1 ramps upward), so a top title gets `Region("ramp", (0, 0, 1, .6))`.
    `invert` swaps inside and outside."""
    shape: Shape = "rect"
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    feather: float = 0.06
    invert: bool = False

    def __post_init__(self):
        if self.shape not in ("rect", "ellipse", "ramp"):
            raise ValueError(f"unknown region shape '{self.shape}'; known: rect, ellipse, ramp")
        if len(self.box) != 4 or not all(0 <= v <= 1 for v in self.box):
            raise ValueError(f"region box {self.box} must be four fractions of the frame, 0..1")
        if self.shape != "ramp" and not (self.box[0] < self.box[2] and self.box[1] < self.box[3]):
            raise ValueError(f"region box {self.box} is empty")
        if not 0 <= self.feather <= 0.5:
            raise ValueError(f"feather {self.feather} outside 0..0.5")


@dataclass(frozen=True)
class Layer:
    """One effect through one region. `amount` 0 leaves the picture alone; `options` are the effect's own knobs
    (look: preset, or filter = a Filter's fields; halftone/duotone: ink, paper; grain: seed)."""
    effect: str
    region: Region = field(default_factory=Region)
    amount: float = 0.6
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.effect not in EFFECTS:
            raise ValueError(f"unknown effect '{self.effect}'; known: {', '.join(EFFECTS)}")
        if not 0 <= self.amount <= 2:
            raise ValueError(f"amount {self.amount} outside 0..2")
        if self.effect == "look":
            if "filter" in self.options:
                filters.Filter.model_validate(self.options["filter"])  # a hand-made grade: curve, tint, grain...
            elif self.options.get("preset") not in filters.PACK:
                raise ValueError(f"look needs options.preset (one of: {', '.join(filters.PACK)}) or options.filter")


def mask(region: Region, h: int, w: int) -> np.ndarray:
    """float32 HxW in 0..1."""
    x0, y0, x1, y1 = region.box
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    fy, fx_ = yy / max(h - 1, 1), xx / max(w - 1, 1)
    soft = max(region.feather * min(h, w), 1.0)
    if region.shape == "ramp":
        m = np.clip((fy - y1) / (y0 - y1 if y0 != y1 else 1e-6), 0, 1)
        m = m * m * (3 - 2 * m)  # smoothstep, so there is no visible band edge
    elif region.shape == "rect":
        d = np.minimum.reduce([xx - x0 * w, x1 * w - xx, yy - y0 * h, y1 * h - yy])
        m = np.clip(d / soft + 0.5, 0, 1)
    else:
        cx, cy, rx, ry = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h, (x1 - x0) / 2 * w, (y1 - y0) / 2 * h
        d = (1 - np.hypot((xx - cx) / rx, (yy - cy) / ry)) * min(rx, ry)  # signed distance, in pixels, roughly
        m = np.clip(d / soft + 0.5, 0, 1)
    return 1 - m if region.invert else m


def run_effect(im: np.ndarray, layer: Layer, seed: int = 0) -> np.ndarray:
    """The effect on the whole frame (BGR uint8); the caller blends it through the region."""
    a, o = layer.amount, layer.options
    h, w = im.shape[:2]
    if layer.effect == "blur":
        return cv2.GaussianBlur(im, (0, 0), max(0.5, a * max(h, w) / 45))
    if layer.effect == "darken":
        return np.clip(im.astype(np.float32) * max(0.0, 1 - 0.8 * a), 0, 255).astype(np.uint8)
    if layer.effect == "mono":
        g = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(im, 1 - min(a, 1), g, min(a, 1), 0)
    if layer.effect == "bloom":
        return fx.bloom(im, a, o.get("threshold", 0.7))
    if layer.effect == "vignette":
        return fx.vignette(im, a)
    if layer.effect == "grain":
        return fx.grain(im, a, seed=o.get("seed", seed))
    if layer.effect == "halftone":
        return fx.halftone(im, o.get("cell", 0), o.get("ink", "#e8141c"), o.get("paper", "#ffffff"),
                           o.get("angle", 45), min(a, 1))
    if layer.effect == "duotone":
        return fx.duotone(im, o.get("ink", "#1a2fb0"), o.get("paper", "#f3ead8"), min(a, 1), o.get("texture", 0.5), seed)
    f = filters.Filter.model_validate(o["filter"]) if "filter" in o else filters.PACK[o["preset"]][1]
    return filters.apply(im, f, a, seed)


def apply_layers(im: np.ndarray, layers: list[Layer], seed: int = 0) -> np.ndarray:
    """Blend each layer through its region, first to last. Each later layer sees the earlier ones' result."""
    out = im
    for layer in layers:
        if layer.amount <= 0:
            continue
        m = mask(layer.region, *out.shape[:2])[..., None]
        fx_im = run_effect(out, layer, seed).astype(np.float32)
        out = np.clip(out.astype(np.float32) * (1 - m) + fx_im * m + 0.5, 0, 255).astype(np.uint8)
    return out
