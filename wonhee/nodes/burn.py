"""burn_subtitles node: plain white-with-black-outline lower-third subtitles burned into the video."""
from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from wonhee import media
from wonhee.db import repo
from wonhee.edit.ass import render_ass
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import load_subs
from wonhee.fsutil import run_timed
from wonhee.procenv import clean_env


class BurnParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    font: str | None = Field(default=None, pattern=r"^[\w .-]{1,48}$")
    size_frac: float = 0.035
    margin_frac: float = 0.22
    max_chars: int = 28
    crf: int = 20


def burn_fn(ctx, inputs, p: BurnParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    subs = load_subs(d, inputs["subtitles"])
    if not any(ln.text.strip() for ln in subs.lines):
        ctx.flag("NO_TEXT", "no subtitle text to burn; video unchanged")
        return {"video": inputs["video"]}
    src = media.path_for(d, inputs["video"])
    info = media.probe(src)
    if not info.width or not info.height:
        raise RuntimeError("cannot read video dimensions")
    with tempfile.TemporaryDirectory(prefix="wonhee-burn-") as tmp:
        tmpd = Path(tmp)
        (tmpd / "subs.ass").write_text(render_ass(subs, info.width, info.height, p.font, p.size_frac,
                                                  p.margin_frac, p.max_chars), encoding="utf-8")
        out = tmpd / "out.mp4"
        proc = run_timed(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src.resolve()), "-vf", "ass=subs.ass",
             "-c:v", "libx264", "-crf", str(p.crf), "-pix_fmt", "yuv420p", "-c:a", "copy",
             "-movflags", "+faststart", str(out)],
            cwd=tmpd, env=clean_env(), capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg burn failed: {proc.stderr.strip()[-400:]}")
        sha = repo.save_file(ctx.conn, d, out, "video", "video/mp4")
    return {"video": sha}


registry.register(Node("burn_subtitles", 1, {"video": "video", "subtitles": "subtitles"}, {"video": "video"},
                       BurnParams, burn_fn,
                       "Burn subtitles into the video as plain white text with a black outline, lower third."))
