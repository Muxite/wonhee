"""Skin undertone: nudge only skin-coloured pixels (the same colour-range rule as skin protection, no mask) toward a
cool, neutral, warm, rose or gold undertone, after the rest of the grade. Sky and foliage are outside the skin hue
wedge, so they are not touched."""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from wonhee.edit.filters import SKIN_DEG, skin_weight

# (da, db) in CIELAB units at strength 1, at the centre of the skin wedge
SHIFT = {"cool": (1.0, -5.0), "warm": (2.0, 6.0), "rose": (6.0, -1.0), "gold": (-2.0, 7.0)}
STRIP_ROWS = 2048


class Undertone(BaseModel):
    tone: Literal["cool", "neutral", "warm", "rose", "gold"] = "neutral"
    strength: float = Field(0.5, ge=0, le=1)


def apply_undertone(im: np.ndarray, u: Undertone) -> np.ndarray:
    """BGR uint8 in, BGR uint8 out. Pixels with no skin weight come back bit-identical."""
    if u.strength <= 0:
        return im
    out = im.copy()
    t = np.deg2rad(SKIN_DEG)
    for y in range(0, im.shape[0], STRIP_ROWS):
        strip = im[y:y + STRIP_ROWS]
        lab = cv2.cvtColor(strip.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        a, b = lab[..., 1], lab[..., 2]
        w = skin_weight(a, b) * u.strength  # weights come from this image, not the original
        live = w > 0.01
        if not live.any():
            continue
        if u.tone == "neutral":  # pull the hue (not the chroma) halfway toward typical skin
            c = np.hypot(a, b)
            da, db = (c * np.cos(t) - a) * 0.5, (c * np.sin(t) - b) * 0.5
        else:
            da, db = SHIFT[u.tone]
        lab[..., 1], lab[..., 2] = a + da * w, b + db * w
        new = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)
        strip_out = out[y:y + STRIP_ROWS]
        strip_out[live] = new[live]
    return out
