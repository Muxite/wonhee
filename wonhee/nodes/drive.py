"""Drive intake is an explicit uncached step with a stored manifest for provenance."""
import json
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from wonhee import drive
from wonhee.db import repo
from wonhee.nodes import registry
from wonhee.nodes.base import Node


class DownloadParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str
    dest: str


def _download(ctx, inputs, params):
    try:
        return _transfer(ctx, params)
    except drive.DownloadError as exc:
        ctx.save_json({'kind': 'drive_error', 'message': str(exc), 'retryable': exc.retryable}, 'error')
        raise


def _transfer(ctx, params):
    result = drive.download(params.url, Path(params.dest), max_bytes=ctx.settings.intake_max_bytes,
                            reserve_bytes=ctx.settings.intake_reserve_bytes)
    for file in result['files']:
        ctx.settings.media_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(ctx.settings.media_dir).free < file['bytes'] + ctx.settings.intake_reserve_bytes:
            raise drive.DownloadError('not enough free disk space to store this download', retryable=False)
        kind = file['mime'].split('/')[0]
        file['ref'] = repo.save_file(ctx.conn, ctx.settings.media_dir, Path(file['path']),
                                     kind if kind in {'image', 'video', 'audio'} else 'file', file['mime'])
    ref = repo.save_bytes(ctx.conn, ctx.settings.media_dir, json.dumps(result).encode(), 'application/json')
    return {'manifest': ref}


registry.register(Node('drive_download', 1, {}, {'manifest': 'file'}, DownloadParams, _download,
                       'Download a public Google Drive file or folder into dest, preserving subfolders.', cacheable=False))
