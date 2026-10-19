"""publish_muksite (image, dark: image? -> text): put a finished still on the owner's portfolio.

Uploads to muksite's Supabase Storage (wonhee/muksite.py) and returns the YAML `photos:` entry for the project
file. Never cached: publishing is a side effect, and a cache hit would report a URL without uploading anything.
Refuses to replace a live image unless overwrite=true.
"""
from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from wonhee import muksite
from wonhee.config import secret
from wonhee.media import path_for
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import save_text
from wonhee.pipeline.context import StageFailed

MAGIC = ((b"\x89PNG\r\n\x1a\n", ".png", "image/png"), (b"\xff\xd8\xff", ".jpg", "image/jpeg"))


class PublishParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str = Field(description="muksite project id, e.g. awork (the image goes under projects/<slug>/)")
    name: str = Field(description="file name without extension, e.g. dependency-graph")
    caption: str = Field(max_length=200)
    alt: str = Field(max_length=400, description="what the image shows, for screen readers")
    size: Literal["hero", "large", "medium", "small", "wide", "tall"] = "hero"
    fit: Literal["partial", "full"] = "partial"
    overwrite: bool = False


def _kind(data: bytes) -> tuple[str, str]:
    for magic, ext, mime in MAGIC:
        if data.startswith(magic):
            return ext, mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    raise StageFailed("PUBLISH_FAILED", "muksite takes png, jpg or webp")


def publish_fn(ctx, inputs, p: PublishParams) -> dict[str, str]:
    url, key = secret("SUPABASE_URL"), secret("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise StageFailed("PUBLISH_FAILED", "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set (wonhee doctor)")
    d = ctx.settings.media_dir
    uploads = []
    try:
        with httpx.Client(timeout=60) as client:
            for port, dark in (("image", False), ("dark", True)):
                if not inputs.get(port):
                    continue
                data = path_for(d, inputs[port]).read_bytes()
                ext, mime = _kind(data)
                path = muksite.object_path(p.slug, p.name, ext, dark=dark)
                uploads.append(muksite.upload(client, url, key, path, data, mime, overwrite=p.overwrite))
                ctx.flag("PUBLISHED", uploads[-1].url)
    except (muksite.PublishError, httpx.HTTPError) as e:
        raise StageFailed("PUBLISH_FAILED", str(e)) from e
    light = uploads[0]
    dark = uploads[1] if len(uploads) > 1 else None
    return {"entry": save_text(ctx.conn, d, muksite.photo_entry(light, dark, p.caption, p.alt, p.size, p.fit))}


registry.register(Node("publish_muksite", 1, {"image": "image", "dark": "image?"}, {"entry": "text"}, PublishParams,
                       publish_fn, "Upload a finished still to muksite's image storage; returns its YAML photo entry.",
                       cacheable=False))
