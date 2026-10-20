"""Subtitle nodes: speech_subs (transcribe or reuse), merge_subs, to_srt."""
from __future__ import annotations

import difflib
import json
import re
import textwrap
from typing import Literal

from pydantic import BaseModel, ConfigDict

from wonhee import media
from wonhee.db import queries
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import Line, Subtitles, Word, load_subs, save_subs, save_text


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _renumbered(language, segments) -> Subtitles:
    """segments: (start_ms, end_ms, text[, words]) where words is a list of (start_ms, end_ms, text) or None."""
    lines = [Line(id=0, start_ms=s[0], end_ms=s[1], text=s[2].strip(), source="speech",
                  words=[Word(start_ms=a, end_ms=b, text=t) for a, b, t in s[3]] if len(s) > 3 and s[3] else None)
             for s in segments if s[2].strip()]
    for i, ln in enumerate(lines):
        ln.id = i
    return Subtitles(language=language, lines=lines)


def speech_fn(ctx, inputs, p) -> dict[str, str]:
    sha = inputs["video"]
    path = media.path_for(ctx.settings.media_dir, sha)
    row = queries.media_analysis_row(ctx.conn, sha)
    if row is not None and row["transcript_sha256"]:
        data = json.loads(media.path_for(ctx.settings.media_dir, row["transcript_sha256"]).read_text())
        # A stored transcript keeps its word timings if it has them (the analyze stage writes none today); we do
        # not re-run whisper to get them: it costs a GPU pass, and consumers estimate words from line times and
        # flag WORDS_ESTIMATED instead.
        subs = _renumbered(data.get("language"),
                           [(s["start_ms"], s["end_ms"], s["text"],
                             [(w["start_ms"], w["end_ms"], w["text"]) for w in s.get("words") or []])
                            for s in data["segments"]])
    elif not media.has_audio(path):
        subs = Subtitles()
    else:
        t = ctx.deps.transcriber.transcribe(path)
        subs = _renumbered(t.language, [(s.start_ms, s.end_ms, s.text, [(w.start_ms, w.end_ms, w.text) for w in s.words or []])
                                   for s in t.segments])
    if not subs.lines:
        ctx.flag("NO_TEXT", "no speech found")
    return {"subtitles": save_subs(ctx.conn, ctx.settings.media_dir, subs)}


class MergeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prefer: Literal["a"] = "a"
    slack_ms: int = 2000   # a lines this close (either side) are compared against b
    cover: float = 0.8     # drop b when contiguous shared blocks make up this share of its text
    long_chars: int = 20   # a b line with this many normalised characters is an end-card candidate...
    long_slack_ms: int = 15000  # ...and is compared against a lines this close: cards replay the last speech


CJK_RE = re.compile("[぀-ヿ㐀-鿿가-힯]")


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def _dup(b: Line, a_lines: list[Line], p: "MergeParams") -> bool:
    """b repeats a when the a lines around it already say (nearly) all of it. Screen lines are timed by the
    sampling interval, so one may span several speech lines, or half of one: compare text, not spans.
    Only contiguous shared blocks count (>= 3 chars, >= 2 for CJK), so scattered letters of a short title
    never add up to a match."""
    nb = _norm(b.text)
    if not nb:
        return False
    slack = p.long_slack_ms if len(nb) >= p.long_chars else p.slack_ms
    near = [a for a in a_lines if a.start_ms < b.end_ms + slack and a.end_ms > b.start_ms - slack]
    joined = _norm("".join(a.text for a in near))
    min_block = 2 if CJK_RE.search(nb) else 3
    blocks = difflib.SequenceMatcher(None, nb, joined, autojunk=False).get_matching_blocks()
    matched = sum(m.size for m in blocks if m.size >= min_block)
    return matched / len(nb) >= p.cover


def merge_fn(ctx, inputs, p: MergeParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    a, b = load_subs(d, inputs["a"]), load_subs(d, inputs["b"])
    kept = list(a.lines) + [ln for ln in b.lines if not _dup(ln, a.lines, p)]
    kept.sort(key=lambda ln: ln.start_ms)
    lines = [ln.model_copy(update={"id": i}) for i, ln in enumerate(kept)]
    return {"subtitles": save_subs(ctx.conn, d, Subtitles(language=a.language or b.language, lines=lines))}


class SrtParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_chars: int = 42


def _ts(ms: int) -> str:
    h, rem = divmod(max(ms, 0), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def _wrap(text: str, max_chars: int) -> list[str]:
    text = " ".join(text.split())
    if " " in text:
        return textwrap.wrap(text, max(max_chars, 1), break_long_words=False, break_on_hyphens=False) or [text]
    n = max(max_chars // 2, 1)
    return [text[i:i + n] for i in range(0, len(text), n)]


def render_srt(subs: Subtitles, max_chars: int) -> str:
    blocks = []
    for n, ln in enumerate(subs.lines, 1):
        body = "\n".join(_wrap(ln.text, max_chars))
        blocks.append(f"{n}\n{_ts(ln.start_ms)} --> {_ts(max(ln.end_ms, ln.start_ms))}\n{body}\n")
    return "\n".join(blocks)


def srt_fn(ctx, inputs, p: SrtParams) -> dict[str, str]:
    subs = load_subs(ctx.settings.media_dir, inputs["subtitles"])
    return {"file": save_text(ctx.conn, ctx.settings.media_dir, render_srt(subs, p.max_chars))}


registry.register(Node("speech_subs", 2, {"video": "video"}, {"subtitles": "subtitles"}, NoParams, speech_fn,
                       "Subtitles from the spoken audio, with word timings when whisper gave them (reuses a stored transcript).",
                       salt=lambda s: {"whisper_model": s.whisper_model, "compute_type": s.whisper_compute_type},
                       lane="cuda"))
registry.register(Node("merge_subs", 4, {"a": "subtitles", "b": "subtitles"}, {"subtitles": "subtitles"},
                       MergeParams, merge_fn, "Merge two subtitle sets, dropping near-duplicate lines from b."))
registry.register(Node("to_srt", 1, {"subtitles": "subtitles"}, {"file": "file"}, SrtParams, srt_fn,
                       "Render subtitles as an SRT file."))
