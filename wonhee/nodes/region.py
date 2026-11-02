"""region_fx node: filters, looks and print effects on part of a photo, the rest left as shot.

The engine (edit/region.py) knows nothing about the DB; this node just reads the image, runs the layers in order and
stores the JPEG. Bump `version` when an effect's output changes.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field

from wonhee.db import repo
from wonhee.edit import region as rg
from wonhee.media import path_for
from wonhee.nodes import registry
from wonhee.nodes.base import Node


class RegionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shape: Literal["rect", "ellipse", "ramp"] = "rect"
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # x0, y0, x1, y1 as frame fractions (ramp: y0 full, y1 none)
    feather: float = Field(default=0.06, ge=0, le=0.5)
    invert: bool = False


class LayerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effect: Literal[rg.EFFECTS] = "darken"  # type: ignore[valid-type]
    region: RegionParams = Field(default_factory=RegionParams)
    amount: float = Field(default=0.6, ge=0, le=2)
    options: dict = Field(default_factory=dict)  # look: {"preset": ...}; halftone/duotone: ink, paper, cell, angle


class RegionFxParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layers: list[LayerParams] = Field(min_length=1, max_length=8)
    seed: int = 0
    quality: int = Field(default=95, ge=60, le=100)


def region_fx_fn(ctx, inputs, p: RegionFxParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    im = cv2.imread(str(path_for(d, inputs["image"])))
    if im is None:
        raise RuntimeError("cannot read the image")
    layers = [rg.Layer(l.effect, rg.Region(l.region.shape, tuple(l.region.box), l.region.feather, l.region.invert),
                       l.amount, l.options) for l in p.layers]
    out_im = rg.apply_layers(im, layers, p.seed)
    with tempfile.TemporaryDirectory(prefix="wonhee-region-") as tmp:
        out = Path(tmp) / "out.jpg"
        cv2.imwrite(str(out), out_im, [cv2.IMWRITE_JPEG_QUALITY, p.quality])
        sha = repo.save_file(ctx.conn, d, out, "image", "image/jpeg")
    return {"image": sha}


registry.register(Node("region_fx", 1, {"image": "image"}, {"image": "image"}, RegionFxParams, region_fx_fn,
                       "Apply filters, looks and print effects (blur, darken, mono, bloom, halftone, duotone, a look "
                       "preset) to part of a photo through a soft rect / ellipse / ramp mask; the rest is untouched."))
