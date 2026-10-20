from __future__ import annotations

from pathlib import Path
from typing import Literal

import re

from pydantic import BaseModel, model_serializer

from wonhee.db import repo
from wonhee.media import path_for


class Word(BaseModel):
    """One spoken word with its own time (reel-styles spec §1)."""
    start_ms: int
    end_ms: int
    text: str


class Line(BaseModel):
    id: int
    start_ms: int
    end_ms: int
    text: str
    source: Literal["speech", "screen"]
    words: list[Word] | None = None  # None = line timing only (older caches, screen_text, translations)

    @model_serializer(mode="wrap")
    def _omit_absent_words(self, handler):
        """Word-less lines serialise exactly as before the field existed, so their cached hashes hold."""
        d = handler(self)
        if d.get("words") is None:
            d.pop("words", None)
        return d


class Subtitles(BaseModel):
    language: str | None = None
    lines: list[Line] = []


def load_subs(media_dir: Path, sha: str) -> Subtitles:
    return Subtitles.model_validate_json(path_for(media_dir, sha).read_bytes())


def save_subs(conn, media_dir: Path, subs: Subtitles) -> str:
    return repo.save_bytes(conn, media_dir, subs.model_dump_json().encode(), "application/json")


def save_text(conn, media_dir: Path, text: str) -> str:
    return repo.save_bytes(conn, media_dir, text.encode("utf-8"), "text/plain; charset=utf-8")


_CJK = re.compile("[぀-ヿ㐀-鿿가-힯]")


def line_words(line: Line) -> tuple[list[Word], bool]:
    """The line's words and whether they were estimated (reel-styles spec §1).

    Real words come back as stored, `estimated=False`. A line without them is split on whitespace (a CJK line with
    no spaces is split per character) and the line's time is shared evenly, `estimated=True`, which the consumer
    reports as WORDS_ESTIMATED. A blank line has no words and nothing to flag."""
    if line.words:
        return list(line.words), False
    text = line.text.strip()
    if not text:
        return [], False
    toks = text.split() if (" " in text or "\t" in text or "\n" in text or not _CJK.search(text)) else list(text)
    n, dur = len(toks), max(line.end_ms - line.start_ms, 0)
    return [Word(start_ms=line.start_ms + round(i * dur / n), end_ms=line.start_ms + round((i + 1) * dur / n),
                 text=t) for i, t in enumerate(toks)], True
