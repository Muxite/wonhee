"""CPU footage inspection supplies real visual context without running a vision model implicitly."""
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from wonhee import media
from wonhee.analyze.contact import contact_sheet
from wonhee.db import repo
from wonhee.nodes import registry
from wonhee.nodes.base import Node


class ContactParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    frames: int = Field(default=6, ge=1, le=12)


def _contact(ctx, inputs, params):
    with tempfile.TemporaryDirectory(prefix='wonhee-contact-') as td:
        path = Path(td) / 'contact.jpg'
        contact_sheet(media.path_for(ctx.settings.media_dir, inputs['video']), path, **params.model_dump())
        return {'image': repo.save_file(ctx.conn, ctx.settings.media_dir, path, 'image', 'image/jpeg')}


registry.register(Node('contact_sheet', 1, {'video': 'video'}, {'image': 'image'}, ContactParams, _contact,
                       'Timestamped real frames for inspecting a take; CPU only, no model.'))
