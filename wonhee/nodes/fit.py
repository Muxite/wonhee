"""photo_fit node: can this photo carry this line of text? Verdict (ok, tight, reject), the measurements, and one
concrete fix per failure, as JSON. It only judges; it writes no DB rows for the post.

Slot rejection is the caller's step, not this node's (engines and nodes do not touch the creative tables). When the
verdict is `reject` (or the caller decides a `tight` one is not usable), the caller sets the slot with the fix as the
reason, e.g. from Python:

    creative.post_slot_set(conn, settings, project_id=pid, slot="slide-2", state="rejected",
                           reason=report["fixes"][0])

and the CLI / MCP twin of that call. `fixes` holds one string per failure; join them with "; " for a longer reason.
"""
from __future__ import annotations

import json
from typing import Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field

from wonhee.edit import fit as ft
from wonhee.edit import mask as mk
from wonhee.edit.text import Preset as TextPreset
from wonhee.edit.text import TextSpec
from wonhee.media import path_for
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import save_text

LOW_SCORE = 0.5  # the same bar text_place uses for "no quiet area"


class FitParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=400)
    preset: TextPreset = "outline"
    size_frac: float | None = Field(default=None, gt=0.01, le=0.2)
    width_frac: float = Field(default=0.84, ge=0.3, le=1.0)
    role: str | None = None
    mask: Literal["auto", "off"] = "auto"  # auto: try the person mask; off: do not look for a subject
    slot: str | None = Field(default=None, max_length=60)  # reporting only: echoed in the JSON, never written anywhere


def photo_fit_fn(ctx, inputs, p: FitParams) -> dict[str, str]:
    im = cv2.imread(str(path_for(ctx.settings.media_dir, inputs["image"])))
    if im is None:
        raise RuntimeError("cannot read the image")
    subject = None
    if p.mask == "off":
        ctx.flag("NO_SUBJECT_MASK", "subject mask switched off; the line was not checked against the face or subject")
    else:
        try:
            subject = mk.subject(im, ctx.settings.home / "models", "person")
        except Exception as e:  # noqa: BLE001 - model missing, offline, onnx error: the verdict still works without it
            ctx.flag("NO_SUBJECT_MASK", f"no subject mask ({type(e).__name__}); the line was not checked against the "
                                        f"face or subject")
    spec = TextSpec(p.text, p.preset, size_frac=p.size_frac, width_frac=p.width_frac, role=p.role)
    f = ft.fit(im, spec, subject)
    if f.score is None or f.score < LOW_SCORE:
        ctx.flag("NO_QUIET_AREA", f"best spot scores {f.score if f.score is not None else 'n/a'}: {f.verdict}")
    h, w = im.shape[:2]
    report = {"slot": p.slot, "frame": {"width": w, "height": h}, "subject_mask": subject is not None, **f.to_dict()}
    return {"fit": save_text(ctx.conn, ctx.settings.media_dir, json.dumps(report, indent=1))}


registry.register(Node("photo_fit", 1, {"image": "image"}, {"fit": "text"}, FitParams, photo_fit_fn,
                       "Can this photo carry this line? verdict ok/tight/reject, measurements and one concrete fix "
                       "per failure (JSON). A reject is set on the post slot by the caller: creative.post_slot_set(..., "
                       "state='rejected', reason=<fix>)."))
