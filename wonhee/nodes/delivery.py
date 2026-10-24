"""A CPU preview node keeps transport derivatives traceable to their authoritative render."""
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from wonhee import media
from wonhee.db import repo
from wonhee.edit.preview import preview
from wonhee.nodes import registry
from wonhee.nodes.base import Node


class PreviewParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_seconds: int = Field(default=60, ge=1, le=60)
    width: int = Field(default=720, ge=128, le=720)
    max_bytes: int = Field(default=45_000_000, ge=1_000_000, le=45_000_000)


def _preview(ctx, inputs, params):
    with tempfile.TemporaryDirectory(prefix='wonhee-preview-') as td:
        dest = Path(td) / 'preview.mp4'
        preview(media.path_for(ctx.settings.media_dir, inputs['video']), dest, **params.model_dump())
        return {'video': repo.save_file(ctx.conn, ctx.settings.media_dir, dest, 'video', 'video/mp4')}


registry.register(Node('delivery_preview', 1, {'video': 'video'}, {'video': 'video'}, PreviewParams, _preview,
                       'A bounded 60-second/720px/45MB CPU delivery preview; keep the original output.'))
