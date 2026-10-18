"""poster node: a photo as a spec-sheet poster (big title, two label boxes, tilted image, mono paragraph)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from wonhee.db import repo
from wonhee.edit import poster as pz
from wonhee.media import path_for
from wonhee.nodes import registry
from wonhee.nodes.base import Node


class PosterParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=40)
    label1: str = Field(default="", max_length=24)
    value1: str = Field(default="", max_length=40)
    label2: str = Field(default="", max_length=24)
    value2: str = Field(default="", max_length=40)
    body_label: str = Field(default="Prompt", max_length=20)
    body: str = Field(default="", max_length=900)
    tilt: float = Field(default=-3.0, ge=-12, le=12)
    rule: str = Field(default="#efb3a3", pattern=r"^#[0-9a-fA-F]{6}$")
    size: str = Field(default="4:5", pattern=r"^(4:5|1:1)$")


def poster_fn(ctx, inputs, p: PosterParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    labels = [(a, b) for a, b in ((p.label1, p.value1), (p.label2, p.value2)) if a or b]
    spec = pz.PosterSpec(title=p.title, labels=labels, body_label=p.body_label, body=p.body, tilt=p.tilt,
                         rule=p.rule, size=(1080, 1350) if p.size == "4:5" else (1080, 1080))
    with Image.open(path_for(d, inputs["image"])) as im:
        card = pz.render(im, spec)
    with tempfile.TemporaryDirectory(prefix="wonhee-poster-") as tmp:
        out = Path(tmp) / "poster.jpg"
        card.save(out, quality=95, subsampling=0)
        return {"image": repo.save_file(ctx.conn, d, out, "image", "image/jpeg")}


registry.register(Node("poster", 1, {"image": "image"}, {"image": "image"}, PosterParams, poster_fn,
                       "A photo as a spec-sheet poster: big title, two label boxes, tilted image, justified mono "
                       "paragraph (a design-study format that frames anything mundane as an exhibit)."))
