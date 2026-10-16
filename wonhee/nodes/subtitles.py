from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from wonhee.db import repo
from wonhee.media import path_for


class Line(BaseModel):
    id: int
    start_ms: int
    end_ms: int
    text: str
    source: Literal["speech", "screen"]


class Subtitles(BaseModel):
    language: str | None = None
    lines: list[Line] = []


def load_subs(media_dir: Path, sha: str) -> Subtitles:
    return Subtitles.model_validate_json(path_for(media_dir, sha).read_bytes())


def save_subs(conn, media_dir: Path, subs: Subtitles) -> str:
    return repo.save_bytes(conn, media_dir, subs.model_dump_json().encode(), "application/json")


def save_text(conn, media_dir: Path, text: str) -> str:
    return repo.save_bytes(conn, media_dir, text.encode("utf-8"), "text/plain; charset=utf-8")
