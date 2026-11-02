"""text_place node: where a text block should sit on a photo (quietest area, clear of the subject, inside the safe
zones), as JSON with the best position and the top candidates. It only decides; `text_image` draws."""
from __future__ import annotations

import json
from typing import Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field

from wonhee.edit import mask as mk
from wonhee.edit import place as pl
from wonhee.edit.text import Preset as TextPreset
from wonhee.edit.text import TextSpec
from wonhee.media import path_for
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import save_text

LOW_SCORE = 0.5  # below this the best spot is still busy or crowds the subject: a reviewer should look


class PlaceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=400)
    preset: TextPreset = "outline"
    size_frac: float | None = Field(default=None, gt=0.01, le=0.2)
    width_frac: float = Field(default=0.84, ge=0.3, le=1.0)
    role: str | None = None
    tracking_em: float = Field(default=0.0, ge=-0.1, le=0.2)
    line_height: float | None = Field(default=None, ge=0.8, le=1.6)
    stroke_em: float | None = Field(default=None, ge=0, le=0.15)
    n: int = Field(default=3, ge=1, le=8)
    mask: Literal["auto", "off"] = "auto"  # auto: try the person mask; off: do not look for a subject


def _spec(p: PlaceParams) -> TextSpec:
    return TextSpec(p.text, p.preset, size_frac=p.size_frac, width_frac=p.width_frac, role=p.role,
                    tracking_em=p.tracking_em, line_height=p.line_height, stroke_em=p.stroke_em)


def text_place_fn(ctx, inputs, p: PlaceParams) -> dict[str, str]:
    im = cv2.imread(str(path_for(ctx.settings.media_dir, inputs["image"])))
    if im is None:
        raise RuntimeError("cannot read the image")
    subject = None
    if p.mask == "off":
        ctx.flag("NO_SUBJECT_MASK", "subject mask switched off; the text was not checked against the face or subject")
    else:
        try:
            subject = mk.subject(im, ctx.settings.home / "models", "person")
        except Exception as e:  # noqa: BLE001 - model missing, offline, onnx error: placement still works without it
            ctx.flag("NO_SUBJECT_MASK", f"no subject mask ({type(e).__name__}); the text was not checked against "
                                        f"the face or subject")
    cands = pl.place(im, _spec(p), subject, p.n)
    if not cands:
        raise RuntimeError("no position found for this text")
    if cands[0].score < LOW_SCORE:
        ctx.flag("NO_QUIET_AREA", f"best spot scores {cands[0].score:.2f}: {cands[0].why}")
    h, w = im.shape[:2]
    report = {"frame": {"width": w, "height": h}, "subject_mask": subject is not None,
              "best": cands[0].to_dict(), "candidates": [c.to_dict() for c in cands]}
    return {"placement": save_text(ctx.conn, ctx.settings.media_dir, json.dumps(report, indent=1))}


registry.register(Node("text_place", 1, {"image": "image"}, {"placement": "text"}, PlaceParams, text_place_fn,
                       "Best position and size for a text block on a photo: quiet area, clear of the subject, inside "
                       "the safe zones; top candidates with scores (JSON)."))
