"""text_image / text_video nodes: one line (or block) of styled text, optionally with a look, in a single encode."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wonhee import media
from wonhee.db import repo
from wonhee.edit.looks import Preset as LookPreset
from wonhee.edit.looks import filter_chain
from wonhee.edit.text import Position, TextSpec, render_layer
from wonhee.edit.text import Preset as TextPreset
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.procenv import clean_env


class TextParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=400)
    preset: TextPreset = "outline"  # outline | band (solid bar) | script (cursive)
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
    return TextSpec(p.text, p.preset, p.position, p.size_frac, p.width_frac, p.band)


def _graph(p: TextParams, video: bool, window: str = "") -> str:
    """[0] base, [1] text layer -> [o]: the look (if any), then the overlay; one pass."""
    look = filter_chain(p.look or "none", p.look_intensity, video=video)
    fmt = "yuv420p" if video else "yuvj444p"
    return f"[0:v]{look}[g];[1:v]format=rgba[t];[g][t]overlay=0:0{window}:format=auto,format={fmt}[o]"


def _ffmpeg(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], cwd=cwd, env=clean_env(),
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


registry.register(Node("text_image", 1, {"image": "image"}, {"image": "image"}, TextParams, text_image_fn,
                       "Put styled text on a photo (outline / band / script; colour emoji and CJK), optional look; "
                       "one pass."))
registry.register(Node("text_video", 1, {"video": "video"}, {"video": "video"}, VideoTextParams, text_video_fn,
                       "Put styled text on a video for a time window (outline / band / script; colour emoji and "
                       "CJK), optional look; one encode."))
