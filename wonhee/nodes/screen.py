"""screen_text node: on-screen text across the whole video, sampled by time with perceptual-hash dedupe."""
from __future__ import annotations

import re
import tempfile
from typing import Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from wonhee import media
from wonhee.analyze.frames import extract_every
from wonhee.analyze.phash import phash_hex
from wonhee.llm import data_block
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import Line, Subtitles, load_subs, save_subs
from wonhee.scout.prompts import SYSTEM_SCREEN_TEXT


class FrameText(BaseModel):
    index: int
    text: str | None = None


class ScreenRead(BaseModel):
    frames: list[FrameText]


class ScreenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every_s: float = 2.0
    max_frames: int = 90
    dedupe_hamming: int = 4
    batch: int = 8
    max_side: int = 1280
    scope: Literal["overlay", "all"] = "overlay"  # overlay: only text the editor put on the video


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


_SCOPE = {
    "overlay": "Only report text overlaid on the video in editing (subtitles, captions, titles, stickers). "
               "Ignore text that is part of the filmed scene: phone or device screens, monitors, code, signs, "
               "packaging. A frame whose only text is in the scene is null.",
    "all": "Report all readable text in the frame, including screens and signs in the scene.",
}


def _prompt(times: list[int], speech: str | None, scope: str = "overlay") -> str:
    listing = "\n".join(f"image {i + 1} = t {t / 1000:.1f}s" for i, t in enumerate(times))
    parts = ["Read the on-screen text of each image: the exact text in its original language, null if none, "
             "ignoring watermarks and handles. Return one entry per image; index is the image number "
             "(1 = first image).", _SCOPE[scope], listing]
    if speech:
        parts.append(data_block("speech in this video (use it to spell names and words correctly)", speech))
    return "\n\n".join(parts)


def screen_fn(ctx, inputs, p: ScreenParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    path = media.path_for(d, inputs["video"])
    dur_ms = media.probe(path).duration_ms or 0
    every = max(p.every_s, 0.1)
    max_frames = max(p.max_frames, 1)
    if dur_ms / 1000 / every > max_frames:
        every = dur_ms / 1000 / max_frames
    ctx_subs = load_subs(d, inputs["context"]) if inputs.get("context") else None
    speech = "\n".join(ln.text for ln in ctx_subs.lines) if ctx_subs and ctx_subs.lines else None
    language = ctx_subs.language if ctx_subs else None

    reads: list[tuple[int, Path]] = []
    with tempfile.TemporaryDirectory(prefix="wonhee-screen-") as tmp:
        frames = extract_every(path, every, Path(tmp), p.max_side)
        last: int | None = None
        for ms, f in frames:
            h = int(phash_hex(f), 16)
            if last is not None and (h ^ last).bit_count() <= p.dedupe_hamming:
                continue
            reads.append((ms, f))
            last = h
        ctx.save_json({"every_s_requested": p.every_s, "every_s_used": every, "max_frames": max_frames,
                       "frames_sampled": len(frames), "frames_read": len(reads)}, "api_response")
        texts: dict[int, str] = {}
        size = max(p.batch, 1)
        for s in range(0, len(reads), size):
            chunk = reads[s:s + size]
            reply = ctx.llm(ScreenRead, schema_name="ScreenRead", system=SYSTEM_SCREEN_TEXT,
                            prompt=_prompt([ms for ms, _ in chunk], speech, p.scope), images=[f for _, f in chunk])
            for ft in reply.frames:
                if 1 <= ft.index <= len(chunk) and ft.text and ft.text.strip():
                    texts.setdefault(chunk[ft.index - 1][0], ft.text.strip())

    end = max(dur_ms, reads[-1][0] + 1 if reads else 0)
    lines: list[Line] = []
    for k, (ms, _) in enumerate(reads):
        text = texts.get(ms)
        if not text:
            continue
        stop = reads[k + 1][0] if k + 1 < len(reads) else end
        if lines and lines[-1].end_ms == ms and _norm(lines[-1].text) == _norm(text):
            lines[-1].end_ms = stop
        else:
            lines.append(Line(id=len(lines), start_ms=ms, end_ms=stop, text=text, source="screen"))
    if not lines:
        ctx.flag("NO_TEXT", "no on-screen text found")
    return {"subtitles": save_subs(ctx.conn, d, Subtitles(language=language, lines=lines))}


registry.register(Node("screen_text", 2, {"video": "video", "context": "subtitles?"}, {"subtitles": "subtitles"},
                       ScreenParams, screen_fn,
                       "Read on-screen text across the whole video, sampled by time, skipping near-identical frames.",
                       salt=lambda s: {"llm_model": s.llm_model}))
