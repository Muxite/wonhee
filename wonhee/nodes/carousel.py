"""carousel / content_images nodes: build a graded, captioned slide set; pull a stored post's images."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from wonhee import media
from wonhee.db import repo
from wonhee.edit.carousel import INSTAGRAM_MAX_SLIDES, layout, load, parse_size
from wonhee.edit.looks import Preset as LookPreset
from wonhee.edit.looks import filter_chain
from wonhee.edit.text import Position, TextSpec, draw_on
from wonhee.edit.text import Preset as TextPreset
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.images import Images, Slide, load_images, save_images
from wonhee.procenv import clean_env


class CarouselParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: str = Field(default="4:5", pattern=r"^(1:1|4:5|9:16|\d{2,4}x\d{2,4})$")
    fit: Literal["cover", "contain"] = "cover"
    span: bool = True  # wide photos (panoramas) spread across slides on exact seams
    look: LookPreset | None = None  # one look for the whole set, so it reads as a set
    look_intensity: float = Field(default=1.0, ge=0, le=2)
    grain: int | None = Field(default=None, ge=0, le=60)
    datestamp: str | None = Field(default=None, pattern=r"^[0-9' .:/-]{1,20}$")
    texts: list[str | None] = Field(default_factory=list, max_length=40)  # per input photo; overrides manifest text
    text_preset: TextPreset = "outline"
    text_position: Position = "lower"
    text_slide: Literal["first", "last"] = "last"  # which slide of a spanning photo gets its text


def _grade(slides: list[Path], out_dir: Path, p: CarouselParams, height: int) -> list[Path]:
    """The whole set through one ffmpeg process (an image sequence), not one process per slide."""
    if p.look is None and p.grain is None and not p.datestamp:
        return slides
    stamp = None
    if p.datestamp:
        stamp = out_dir / "stamp.txt"
        stamp.write_text(p.datestamp, encoding="utf-8")
    vf = filter_chain(p.look or "none", p.look_intensity, video=False, grain=p.grain, datestamp_file=stamp,
                      height=height).replace("format=yuvj444p", "format=rgb24")
    proc = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-start_number", "1", "-i",
                           str(slides[0].parent / "s%03d.png"), "-vf", vf, "-start_number", "1",
                           str(out_dir / "g%03d.png")], env=clean_env(), capture_output=True, text=True)
    graded = [out_dir / f"g{i:03d}.png" for i in range(1, len(slides) + 1)]
    if proc.returncode != 0 or not all(g.exists() for g in graded):
        raise RuntimeError(f"ffmpeg look failed: {proc.stderr.strip()[-400:]}")
    return graded


def carousel_fn(ctx, inputs, p: CarouselParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    src = load_images(d, inputs["images"]).items
    if not src:
        raise ValueError("no images to lay out")
    size = parse_size(p.size)
    with tempfile.TemporaryDirectory(prefix="wonhee-carousel-") as tmp:
        tmpd = Path(tmp)
        raw, captions = tmpd / "raw", []
        raw.mkdir()
        n = 0
        for k, item in enumerate(src):
            text = p.texts[k] if k < len(p.texts) else item.text
            cols = layout(load(media.path_for(d, item.sha)), size, p.fit, p.span)
            target = 0 if p.text_slide == "first" else len(cols) - 1
            for i, col in enumerate(cols):
                n += 1
                col.save(raw / f"s{n:03d}.png")
                captions.append(text if (text and i == target) else None)
        slides = _grade([raw / f"s{i:03d}.png" for i in range(1, n + 1)], tmpd, p, size[1])
        if n > INSTAGRAM_MAX_SLIDES:
            ctx.flag("TOO_MANY_SLIDES", f"{n} slides; Instagram takes {INSTAGRAM_MAX_SLIDES}")
        items = []
        for i, (path, caption) in enumerate(zip(slides, captions), 1):
            with Image.open(path) as im:
                im = im.convert("RGB")
                if caption:
                    im = draw_on(im, TextSpec(caption, p.text_preset, p.text_position))
                out = tmpd / f"slide{i:03d}.jpg"
                im.save(out, quality=95, subsampling=0)
            items.append(Slide(sha=repo.save_file(ctx.conn, d, out, "image", "image/jpeg")))
    return {"images": save_images(ctx.conn, d, Images(items=items))}


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def content_images_fn(ctx, inputs, p: NoParams) -> dict[str, str]:
    cid = int(inputs["content"].split(":", 1)[1])
    shas = [r[0] for r in ctx.conn.execute(
        "SELECT media_sha256 FROM content_media WHERE content_id=? AND kind='image' ORDER BY ord", (cid,))]
    if not shas:
        raise ValueError(f"content {cid} has no stored images (grab it first; reels have none)")
    return {"images": save_images(ctx.conn, ctx.settings.media_dir, Images(items=[Slide(sha=s) for s in shas]))}


registry.register(Node("carousel", 1, {"images": "images"}, {"images": "images"}, CarouselParams, carousel_fn,
                       "Lay photos out as Instagram slides (4:5 / 1:1 / 9:16; panoramas span slides), one look for "
                       "the whole set, optional per-photo text."))
registry.register(Node("content_images", 1, {"content": "content"}, {"images": "images"}, NoParams,
                       content_images_fn, "The images of a stored photo or carousel post, in order.",
                       cacheable=False))
