"""Publish a finished image to muksite (the owner's portfolio): upload to its Supabase Storage bucket and return the
YAML `photos:` entry to paste into a project file.

Same bucket and URL shape as muksite's own scripts/upload_images.py, so muksite's build_image_variants.py picks the
image up like any other and makes its WebP variants; wonhee does not build variants itself, so that stays in one
place. Without `overwrite`, the upload omits `x-upsert` and Supabase refuses an existing path, which makes the
"never silently replace a live image" rule atomic instead of a check-then-write race.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

BUCKET = "portfolio-images"
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class PublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class Upload:
    path: str  # object path inside the bucket
    url: str   # public URL


def object_path(slug: str, name: str, ext: str, dark: bool = False) -> str:
    for label, v in (("slug", slug), ("name", name)):
        if not SLUG.match(v):
            raise PublishError(f"{label} must be lowercase letters, digits and dashes: {v!r}")
    return f"projects/{slug}/{name}{'-dark' if dark else ''}{ext}"


def public_url(supabase_url: str, path: str) -> str:
    return f"{supabase_url.rstrip('/')}/storage/v1/object/public/{BUCKET}/{path}"


def upload(client: httpx.Client, supabase_url: str, key: str, path: str, data: bytes, mime: str,
           overwrite: bool = False) -> Upload:
    headers = {"Authorization": f"Bearer {key}", "apikey": key, "Content-Type": mime}
    if overwrite:
        headers["x-upsert"] = "true"
    r = client.post(f"{supabase_url.rstrip('/')}/storage/v1/object/{BUCKET}/{path}", content=data, headers=headers)
    if r.status_code in (400, 409) and "exist" in r.text.lower():
        raise PublishError(f"{path} is already published; pass overwrite=true to replace it")
    if r.status_code >= 300:
        raise PublishError(f"upload {path}: HTTP {r.status_code} {r.text[:200]}")
    return Upload(path, public_url(supabase_url, path))


def photo_entry(light: Upload, dark: Upload | None, caption: str, alt: str, size: str = "hero",
                fit: str = "partial") -> str:
    """The `photos:` list item for a muksite project YAML (same keys and order as the existing entries)."""
    entry: dict[str, str] = {"url": light.url}
    if dark:
        entry["darkUrl"] = dark.url
    entry.update(caption=caption, alt=alt, size=size, fit=fit)
    body = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=1000)
    return "photos:\n" + "".join(f"  {line}\n" for line in body.splitlines())


def mime_for(path: Path) -> str:
    try:
        return MIME[path.suffix.lower()]
    except KeyError:
        raise PublishError(f"{path.name}: muksite takes png, jpg or webp") from None
