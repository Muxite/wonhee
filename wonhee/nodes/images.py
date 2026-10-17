"""The `images` port: an ordered set of stored images (a carousel / photo dump) as a JSON manifest blob."""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from wonhee.db import repo
from wonhee.media import path_for

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}


class Slide(BaseModel):
    sha: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str | None = None  # optional per-slide line, drawn by nodes that take text


class Images(BaseModel):
    items: list[Slide] = []


def load_images(media_dir: Path, sha: str) -> Images:
    return Images.model_validate_json(path_for(media_dir, sha).read_bytes())


def save_images(conn, media_dir: Path, images: Images) -> str:
    return repo.save_bytes(conn, media_dir, images.model_dump_json().encode(), "application/json")


def _natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def collect(path: Path) -> list[tuple[Path, str | None]]:
    """Local input for an images port: a directory (natural sort), a JSON list of paths or {path, text}
    objects (relative to the JSON file), or a single image file."""
    if path.is_dir():
        return [(p, None) for p in sorted((p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT), key=_natural)]
    if path.suffix.lower() == ".json":
        out = []
        for e in json.loads(path.read_text()):
            p, text = (e, None) if isinstance(e, str) else (e["path"], e.get("text"))
            p = Path(p).expanduser()
            out.append((p if p.is_absolute() else path.parent / p, text))
        return out
    return [(path, None)]


def store_local(conn, media_dir: Path, path: Path) -> str:
    items = []
    for p, text in collect(path):
        if not p.is_file():
            raise ValueError(f"{p}: no such image")
        items.append(Slide(sha=repo.save_file(conn, media_dir, p, "image", "image/jpeg"), text=text))
    if not items:
        raise ValueError(f"{path}: no images found")
    return save_images(conn, media_dir, Images(items=items))
