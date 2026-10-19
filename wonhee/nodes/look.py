"""look / look_image nodes: apply a named grade or film texture (moody, grain, digicam, ...) in one encode."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wonhee import media
from wonhee.db import repo
from wonhee.edit.looks import Preset, filter_chain
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.fsutil import run_timed
from wonhee.procenv import clean_env


class LookParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preset: Preset = "moody"
    intensity: float = Field(default=1.0, ge=0, le=2)
    grain: int | None = Field(default=None, ge=0, le=60)  # overrides the preset's grain; 0 = none
    datestamp: str | None = Field(default=None, pattern=r"^[0-9' .:/-]{1,20}$")  # e.g. "'25 10 17"


class VideoLookParams(LookParams):
    encoder: Literal["auto", "nvenc", "x264"] = "auto"  # auto: NVENC if it works here, else x264
    quality: int = Field(default=20, ge=12, le=35)  # x264 CRF / NVENC CQ


def _ffmpeg(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return run_timed(["ffmpeg", "-loglevel", "error", "-y", *args], cwd=cwd, env=clean_env(),
                          capture_output=True, text=True)


def _stamp(tmp: Path, p: LookParams) -> Path | None:
    if not p.datestamp:
        return None
    f = tmp / "stamp.txt"
    f.write_text(p.datestamp, encoding="utf-8")
    return f


def look_fn(ctx, inputs, p: VideoLookParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    src = media.path_for(d, inputs["video"]).resolve()
    info = media.probe(src)
    with tempfile.TemporaryDirectory(prefix="wonhee-look-") as tmp:
        tmpd = Path(tmp)
        vf = filter_chain(p.preset, p.intensity, video=True, grain=p.grain, datestamp_file=_stamp(tmpd, p),
                          height=info.height)
        out = tmpd / "out.mp4"
        enc = {"nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(p.quality),
                         "-b:v", "0"],
               "x264": ["-c:v", "libx264", "-preset", "medium", "-crf", str(p.quality)]}
        order = ["nvenc", "x264"] if p.encoder == "auto" else [p.encoder]
        proc = None
        for name in order:
            proc = _ffmpeg(["-i", str(src), "-vf", vf, *enc[name], "-c:a", "copy", "-movflags", "+faststart",
                            str(out)], tmpd)
            if proc.returncode == 0 and out.exists():
                if name != order[0]:
                    ctx.flag("ENCODER_FALLBACK", f"{order[0]} failed, used {name}")
                break
        else:
            raise RuntimeError(f"ffmpeg look failed: {proc.stderr.strip()[-400:]}")
        sha = repo.save_file(ctx.conn, d, out, "video", "video/mp4")
    return {"video": sha}


def look_image_fn(ctx, inputs, p: LookParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    src = media.path_for(d, inputs["image"]).resolve()
    info = media.probe(src)
    with tempfile.TemporaryDirectory(prefix="wonhee-look-") as tmp:
        tmpd = Path(tmp)
        vf = filter_chain(p.preset, p.intensity, video=False, grain=p.grain, datestamp_file=_stamp(tmpd, p),
                          height=info.height)
        out = tmpd / "out.jpg"
        proc = _ffmpeg(["-i", str(src), "-vf", vf, "-frames:v", "1", "-q:v", "2", str(out)], tmpd)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg look failed: {proc.stderr.strip()[-400:]}")
        sha = repo.save_file(ctx.conn, d, out, "image", "image/jpeg")
    return {"image": sha}


registry.register(Node("look", 1, {"video": "video"}, {"video": "video"}, VideoLookParams, look_fn,
                       "Grade a video with a named look (moody, grain, digicam, faded, bw_grain, warm_pink, "
                       "cool_night), optional digicam date stamp; one encode."))
registry.register(Node("look_image", 1, {"image": "image"}, {"image": "image"}, LookParams, look_image_fn,
                       "Grade a photo with a named look (moody, grain, digicam, faded, bw_grain, warm_pink, "
                       "cool_night), optional digicam date stamp."))
