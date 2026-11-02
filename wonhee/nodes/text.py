"""text_image / text_video nodes: one line (or block) of styled text, optionally with a look, in a single encode."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from wonhee import media
from wonhee.db import repo
from wonhee.edit.looks import Preset as LookPreset
from wonhee.edit.looks import filter_chain
from wonhee.edit.text import ROLES, BoxKind, Position, TextSpec, render_layer
from wonhee.edit.text import Preset as TextPreset
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.fsutil import run_timed
from wonhee.procenv import clean_env


class TextStyle(BaseModel):
    """The TextSpec knobs beyond preset and position (docs/specs/2026-10-20-reel-styles-design.md §4, and the sticker
    and placement knobs of the meme spec). Every default is TextSpec's own, so a node with none of them set renders
    exactly as it did before these existed. Shared by text_image / text_video and by carousel's text_style."""
    model_config = ConfigDict(extra="forbid")
    at: tuple[float, float] | None = None  # block centre (x, y) as frame fractions; overrides position, kept in the safe band
    role: str | None = None  # a font role (edit/text.py ROLES) replacing the preset's font
    colours: list[str] | None = Field(default=None, min_length=1, max_length=3)  # fill, stroke (sticker: + outer stroke)
    align: Literal["left", "center", "right"] = "center"  # used by render_block (compose); text_image lines are centred
    tracking_em: float = Field(default=0.0, ge=-0.1, le=0.2)
    line_height: float | None = Field(default=None, ge=0.8, le=1.6)
    box: BoxKind = "none"  # a background per line: box, pill or paper
    box_colour: str | None = None  # None = the box kind's own default
    box_pad_em: float = Field(default=0.25, ge=0, le=1)
    box_radius_em: float | None = Field(default=None, ge=0, le=1)
    stroke_em: float | None = Field(default=None, ge=0, le=0.15)  # outline width; None = the preset's own
    slant: float = Field(default=0.0, ge=-12, le=12)  # sticker only, degrees
    drop: float = Field(default=0.06, ge=0, le=0.5)  # sticker only: shadow offset as a fraction of the glyph size

    @field_validator("at")
    @classmethod
    def _unit(cls, v):
        if v is not None and not all(0 <= c <= 1 for c in v):
            raise ValueError("at must be two fractions of the frame, 0..1")
        return v

    @field_validator("role")
    @classmethod
    def _role(cls, v):
        if v is not None and v not in ROLES:
            raise ValueError(f"unknown font role '{v}'; known: {', '.join(ROLES)}")
        return v


def style_kwargs(s: TextStyle) -> dict:
    """The TextSpec keyword arguments for a TextStyle."""
    return dict(role=s.role, colours=tuple(s.colours) if s.colours else None, align=s.align,
                tracking_em=s.tracking_em, line_height=s.line_height, box=s.box, box_colour=s.box_colour,
                box_pad_em=s.box_pad_em, box_radius_em=s.box_radius_em, stroke_em=s.stroke_em, slant=s.slant,
                drop=s.drop, at=s.at)


class TextParams(TextStyle):
    text: str = Field(min_length=1, max_length=400)
    preset: TextPreset = "outline"  # outline | band (solid bar) | script (cursive) | sticker (pop-art)
    position: Position = "upper"
    size_frac: float | None = Field(default=None, gt=0.01, le=0.2)
    width_frac: float = Field(default=0.84, ge=0.3, le=1.0)
    band: Literal["dark", "light"] = "dark"
    look: LookPreset | None = None  # applied under the text in the same pass
    look_intensity: float = Field(default=1.0, ge=0, le=2)


class VideoTextParams(TextParams):
    start_s: float = Field(default=0, ge=0)
    end_s: float | None = Field(default=None, gt=0)  # None = until the end
    encoder: Literal["auto", "nvenc", "x264"] = "auto"
    quality: int = Field(default=20, ge=12, le=35)


def _spec(p: TextParams) -> TextSpec:
    return TextSpec(p.text, p.preset, p.position, p.size_frac, p.width_frac, p.band, **style_kwargs(p))


def _graph(p: TextParams, video: bool, window: str = "") -> str:
    """[0] base, [1] text layer -> [o]: the look (if any), then the overlay; one pass."""
    look = filter_chain(p.look or "none", p.look_intensity, video=video)
    fmt = "yuv420p" if video else "yuvj444p"
    return f"[0:v]{look}[g];[1:v]format=rgba[t];[g][t]overlay=0:0{window}:format=auto,format={fmt}[o]"


def _ffmpeg(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return run_timed(["ffmpeg", "-loglevel", "error", "-y", *args], cwd=cwd, env=clean_env(),
                          capture_output=True, text=True)


def text_image_fn(ctx, inputs, p: TextParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    src = media.path_for(d, inputs["image"]).resolve()
    info = media.probe(src)
    if not info.width or not info.height:
        raise RuntimeError("cannot read image dimensions")
    with tempfile.TemporaryDirectory(prefix="wonhee-text-") as tmp:
        tmpd = Path(tmp)
        render_layer(_spec(p), (info.width, info.height)).save(tmpd / "text.png")
        out = tmpd / "out.jpg"
        proc = _ffmpeg(["-i", str(src), "-i", "text.png", "-filter_complex", _graph(p, video=False),
                        "-map", "[o]", "-frames:v", "1", "-q:v", "2", str(out)], tmpd)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg text failed: {proc.stderr.strip()[-400:]}")
        sha = repo.save_file(ctx.conn, d, out, "image", "image/jpeg")
    return {"image": sha}


def text_video_fn(ctx, inputs, p: VideoTextParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    src = media.path_for(d, inputs["video"]).resolve()
    info = media.probe(src)
    if not info.width or not info.height:
        raise RuntimeError("cannot read video dimensions")
    window = ""
    if p.start_s > 0 or p.end_s is not None:
        end = p.end_s if p.end_s is not None else 1e9
        window = f":enable='between(t,{p.start_s:g},{end:g})'"
    enc = {"nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(p.quality),
                     "-b:v", "0"],
           "x264": ["-c:v", "libx264", "-preset", "medium", "-crf", str(p.quality)]}
    order = ["nvenc", "x264"] if p.encoder == "auto" else [p.encoder]
    with tempfile.TemporaryDirectory(prefix="wonhee-text-") as tmp:
        tmpd = Path(tmp)
        render_layer(_spec(p), (info.width, info.height)).save(tmpd / "text.png")
        out = tmpd / "out.mp4"
        proc = None
        for name in order:
            proc = _ffmpeg(["-i", str(src), "-i", "text.png", "-filter_complex", _graph(p, True, window),
                            "-map", "[o]", "-map", "0:a?", *enc[name], "-c:a", "copy", "-movflags", "+faststart",
                            str(out)], tmpd)
            if proc.returncode == 0 and out.exists():
                if name != order[0]:
                    ctx.flag("ENCODER_FALLBACK", f"{order[0]} failed, used {name}")
                break
        else:
            raise RuntimeError(f"ffmpeg text failed: {proc.stderr.strip()[-400:]}")
        sha = repo.save_file(ctx.conn, d, out, "video", "video/mp4")
    return {"video": sha}


registry.register(Node("text_image", 2, {"image": "image"}, {"image": "image"}, TextParams, text_image_fn,
                       "Put styled text on a photo (outline / band / script; colour emoji and CJK), optional look; "
                       "one pass."))
registry.register(Node("text_video", 2, {"video": "video"}, {"video": "video"}, VideoTextParams, text_video_fn,
                       "Put styled text on a video for a time window (outline / band / script; colour emoji and "
                       "CJK), optional look; one encode."))
