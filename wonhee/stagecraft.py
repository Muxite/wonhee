"""Delegate staged prop creation to Stagecraft, retain trusted delivery ownership here."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from wonhee.clock import now_iso
from wonhee.db.connect import tx


def _call(command: str, data: dict | None = None, job_id: str | None = None) -> dict:
    repo = Path(os.environ.get('WONHEE_STAGECRAFT_REPO', str(Path(__file__).resolve().parents[2] / 'stagecraft')))
    argv = [str(repo / '.venv/bin/stagecraft'), command]
    if job_id:
        if not re.fullmatch(r'[0-9a-f]{32}', job_id):
            raise ValueError('invalid Stagecraft job ID')
        argv.append(job_id)
    env = {k:v for k,v in os.environ.items() if k in {'PATH', 'HOME', 'STAGECRAFT_HOME'}}
    env.setdefault('STAGECRAFT_HOME', '/mnt/arteta/stagecraft')
    result = subprocess.run(argv, cwd=repo, env=env, input=json.dumps(data) if data is not None else None,
                            capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise ValueError('Stagecraft rejected the operation: ' + result.stderr.strip()[-1500:])
    return json.loads(result.stdout)


def stagecraft_schema(conn, settings) -> dict:
    """Read production spec/schema and optional custom React app contract before authoring a demo."""
    return _call('schema')


def stagecraft_submit(conn, settings, *, spec: dict, project_id: int, request_key: str,
                      route: dict | None = None) -> dict:
    """Create one durable containerized demo job. Author title/premise, initial state and timed actions;
    optional component/css implements a custom React scene. Set spec.coding={task, primitive:'stdio_mesh'}
    to dispatch Codex to author a real stdio program and dashboard; independent Docker checks gate capture.
    Capture returns through the existing outbox.
    Reuse request_key on timeout; a changed idea is a new request. Queued does not mean delivered."""
    from wonhee import creative
    creative.project_show(conn, settings, project_id=project_id)
    return _call('submit', {'spec':spec,'project_id':project_id,'request_key':request_key,
                            'route':route or {},'wonhee_home':str(settings.home.resolve())})


def stagecraft_show(conn, settings, *, job_id: str, project_id: int | None = None) -> dict:
    """Inspect one Stagecraft job: phase, capture result, failure and staged delivery ID. Do not poll in a loop."""
    result = _call('show', job_id=job_id)
    if project_id is not None and result['project_id'] != project_id:
        raise ValueError('Stagecraft job belongs to another creative project')
    return result


def stagecraft_retry(conn, settings, *, job_id: str, project_id: int | None = None) -> dict:
    """Retry a failed Stagecraft job only when requested. Successful capture is reused for delivery recovery."""
    stagecraft_show(conn, settings, job_id=job_id, project_id=project_id)
    return _call('retry', job_id=job_id)


def stagecraft_complete(conn, settings, *, job_id: str, source: str, project_id: int, request_key: str,
                        manifest: dict, title: str, route: dict | None = None) -> dict:
    """Host-only completion receipt. Atomically deduplicate outbox staging; never claims a transport send."""
    from wonhee import creative
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('invalid Stagecraft job ID')
    path = Path(source).resolve()
    if not path.is_file() or path.suffix != '.mp4':
        raise ValueError('Stagecraft completion needs a real MP4')
    capture = manifest.get('capture', {})
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if manifest.get('capture_status') != 'mechanical_checks_passed' or capture.get('sha256') != digest:
        raise ValueError('Stagecraft capture manifest/hash is invalid')
    route_json = json.dumps(route or {}, sort_keys=True)
    creative.project_show(conn, settings, project_id=project_id)
    with tx(conn):
        old = conn.execute('SELECT * FROM stagecraft_delivery WHERE job_id=?', (job_id,)).fetchone()
        if old:
            if (old['ref'],old['project_id'],old['request_key'],old['route_json']) != (digest,project_id,request_key,route_json):
                raise ValueError('Stagecraft completion key conflicts with the original output/route')
            return creative.delivery_show(conn, settings, delivery_id=old['delivery_id'])
        asset = creative.inbox_add(conn, settings, source=str(path), project_id=project_id,
            source_key=f'stagecraft:{job_id}', role='output', metadata={'producer':'stagecraft','manifest':manifest,
                                                                      'request_key':request_key})
        delivery = creative.outbox_add(conn, settings, ref=asset['ref'], filename=f'stagecraft-{job_id}.mp4',
            project_id=project_id, route=route, content=f'{title[:160]} — staged demo ({manifest.get("mode", "simulated_story")})',
            as_document=True)
        conn.execute('INSERT INTO stagecraft_delivery VALUES(?,?,?,?,?,?,?,?)',
                     (job_id, project_id, request_key, digest, route_json, delivery['id'],
                      json.dumps(manifest), now_iso()))
        return delivery
