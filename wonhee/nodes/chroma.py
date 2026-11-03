"""chroma_key node: key the backdrop out of a video (green, blue, or a whitish wall) and composite a new one.

Spec: docs/specs/2026-11-01-chroma-key-design.md. The engine is edit/chroma.py; this only reads the stored ref, runs
it, flags what a reviewer should see and stores the result. `background` is optional and may be an image or a video
(a `file` port, like montage's audio); without it the drawn cabin (or a flat colour) is the fill.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from wonhee import media
from wonhee.db import repo
from wonhee.edit import chroma
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import save_text

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_FLAGS = {
    "BACKDROP_UNEVEN": "the wall is not one colour: add light, or keep guard=person",
    "SUBJECT_TOO_SMALL": "the subject is under 3% of the frame: the key probably ate the person",
    "SUBJECT_TOO_BIG": "the subject is over 85% of the frame: the key probably missed the wall",
}


class ChromaParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backdrop: str = Field(default="auto", description="auto, green, blue, white or #rrggbb")
    tolerance: float = Field(default=0.5, ge=0, le=1)
    softness: float = Field(default=0.5, ge=0, le=1)
    guard: str = Field(default="auto", description="auto, person or none")
    fill: str = Field(default="cabin", description="cabin, window, matte or #rrggbb (used when no background is given)")
    spill: float = Field(default=0.7, ge=0, le=1)

    @field_validator("backdrop")
    @classmethod
    def _backdrop(cls, v):
        if v not in ("auto", "green", "blue", "white") and not _HEX.match(v):
            raise ValueError("backdrop must be auto, green, blue, white or #rrggbb")
        return v

    @field_validator("guard")
    @classmethod
    def _guard(cls, v):
        if v not in ("auto", "person", "none"):
            raise ValueError("guard must be auto, person or none")
        return v

    @field_validator("fill")
    @classmethod
    def _fill(cls, v):
        if v not in ("cabin", "window", "matte") and not _HEX.match(v):
            raise ValueError("fill must be cabin, window, matte or #rrggbb")
        return v


def chroma_fn(ctx, inputs, p: ChromaParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    bg = media.path_for(d, inputs["background"]) if inputs.get("background") else None
    with tempfile.TemporaryDirectory(prefix="wonhee-chroma-") as tmp:
        out = Path(tmp) / "keyed.mp4"
        rep = chroma.key_video(media.path_for(d, inputs["video"]), out, p.backdrop, p.tolerance, p.softness, p.guard,
                               p.fill, bg, p.spill, models_dir=ctx.settings.home / "models")
        for code in rep["flags"]:
            ctx.flag(code, _FLAGS[code])
        sha = repo.save_file(ctx.conn, d, out, "video", "video/mp4")
    return {"video": sha, "report": save_text(ctx.conn, d, json.dumps(rep, indent=2, sort_keys=True))}


registry.register(Node("chroma_key", 1, {"video": "video", "background": "file?"}, {"video": "video", "report": "text"},
                       ChromaParams, chroma_fn,
                       "Key out the backdrop (green, blue or a whitish wall, found automatically), guard the person "
                       "with the person mask, and composite an image/video background or a drawn plane cabin.",
                       salt=lambda s: {"mask": "u2net-onnx"}))
