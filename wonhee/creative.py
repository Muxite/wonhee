"""Creative workspace services: canonical project state and source-linked memory, independent of chat history."""
from __future__ import annotations

import json
import hashlib
import fcntl
import functools
import mimetypes
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from typing import Literal

from wonhee import media
from wonhee.clock import now_iso
from wonhee.db import repo
from wonhee.db.connect import tx
from wonhee.db.repo import creative as writes
from wonhee.fsutil import write_atomic


MemoryKind = Literal['preference', 'constraint', 'decision', 'feedback', 'question', 'idea', 'brief', 'transcript', 'output']
MemoryState = Literal['confirmed', 'proposed', 'superseded', 'forgotten']
MemoryScope = Literal['auto', 'project', 'creator', 'all']
QueryOperation = Literal['search', 'list', 'count', 'timeline']
IdeaState = Literal['draft', 'accepted', 'producing', 'done', 'rejected']


_MEMORY_STATES = {'confirmed', 'proposed', 'superseded', 'forgotten'}
_IDEA_STATES = {'draft', 'accepted', 'producing', 'done', 'rejected'}


def _record(row) -> dict:
    if row is None:
        raise KeyError('creative record does not exist')
    out = dict(row)
    for k in list(out):
        if k.endswith('_json'):
            out[k[:-5]] = json.loads(out.pop(k))
    return out


def _get(conn, table: str, record_id: int) -> dict:
    if table not in {'creative_project', 'creative_intake', 'creative_memory', 'creative_idea',
                     'creative_submission', 'creative_delivery'}:
        raise ValueError('unknown creative table')
    return _record(conn.execute(f'SELECT * FROM {table} WHERE id=?', (record_id,)).fetchone())


def _text(value: str, name='content', limit=20000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'{name} must contain 1..{limit} characters')
    return value.strip()


def project_create(conn, settings, *, name: str) -> dict:
    return project_show(conn, settings, project_id=writes.project_create(conn, _text(name, 'name', 200)))


def project_select(conn, settings, *, project_id: int | None, route_key: str) -> dict:
    route_key = _text(route_key, 'route_key', 500)
    if project_id is not None:
        project_show(conn, settings, project_id=project_id)
    writes.project_select(conn, project_id, route_key)
    return project_show(conn, settings, route_key=route_key)


def project_show(conn, settings, *, project_id: int | None = None, route_key: str | None = None) -> dict:
    if route_key:
        row = conn.execute('SELECT project_id FROM creative_route WHERE route_key=?', (route_key,)).fetchone()
        if row is None:
            return {'id': None}
        project_id = row[0]
    if project_id is None:
        return {'projects': [_record(r) for r in conn.execute('SELECT * FROM creative_project ORDER BY id DESC')]}
    row = _get(conn, 'creative_project', project_id)
    brief = conn.execute('SELECT * FROM creative_brief WHERE project_id=? ORDER BY revision DESC LIMIT 1',
                         (project_id,)).fetchone()
    row['brief'] = _record(brief) if brief else None
    from wonhee.intent.creator import load_creator
    row['creator'] = load_creator(settings.creator_path).model_dump(mode='json')
    return row


def _serialized_intake(fn):
    @functools.wraps(fn)
    def call(conn, settings, **kwargs):
        key = kwargs.get('source_key') or uuid4().hex
        kwargs['source_key'] = _text(key, 'source_key', 1000)
        root = settings.home / 'intake-locks'
        root.mkdir(parents=True, exist_ok=True)
        with (root / hashlib.sha256(key.encode()).hexdigest()).open('a') as lock:
            # A timed-out client can reconnect while the server still downloads; the same receipt has one writer.
            fcntl.flock(lock, fcntl.LOCK_EX)
            return fn(conn, settings, **kwargs)
    return call


@_serialized_intake
def inbox_add(conn, settings, *, source: str, project_id: int | None = None, source_key: str | None = None,
              role: str = 'raw', metadata: dict | None = None) -> dict:
    from wonhee.intake import download_drive, drive_identity
    if project_id is not None:
        project_show(conn, settings, project_id=project_id)
    if role not in {'raw', 'reference', 'audio', 'output'}:
        raise ValueError('asset role must be raw, reference, audio or output')
    source = _text(source, 'source', 8000)
    remote = source.startswith(('http:', 'https:'))
    if remote:
        drive_identity(source)  # Reject unsupported links before creating a receipt or doing network I/O.
    key = source_key or uuid4().hex
    old = conn.execute('SELECT * FROM creative_intake WHERE source_key=?', (key,)).fetchone()
    if old and (old['project_id'] != project_id or old['source'] != source):
        raise ValueError('source_key already belongs to another source/project')
    if old and old['status'] == 'stored':
        return media_info(conn, settings, asset_id=old['id'])
    intake_id = old['id'] if old else writes.insert(conn, 'creative_intake', project_id=project_id, source_key=key,
        source=source, role=role, status='received', metadata_json=json.dumps(metadata or {}, ensure_ascii=False))
    try:
        writes.update(conn, 'creative_intake', intake_id, status='downloading', error=None)
        settings.home.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='intake-', dir=settings.home) as td:
            if remote:
                path = Path(td) / 'download'
                detail = download_drive(source, path, max_bytes=settings.intake_max_bytes,
                                        reserve_bytes=settings.intake_reserve_bytes)
                filename, mime = detail['filename'], detail['mime']
            else:
                path = Path(source).expanduser().resolve()
                if not path.is_file():
                    raise ValueError('input is not an existing file')
                filename = Path((metadata or {}).get('original_name') or path.name).name
                mime = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            if path.stat().st_size > settings.intake_max_bytes:
                raise ValueError('file exceeds the configured intake size limit')
            if shutil.disk_usage(settings.home).free < path.stat().st_size + settings.intake_reserve_bytes:
                raise ValueError('not enough free space to store this file')
            if mime in {'text/html', 'application/xhtml+xml'}:
                raise ValueError('HTML is not supported as a media upload')
            kind = mime.split('/')[0] if mime.split('/')[0] in {'video', 'audio', 'image'} else 'file'
            # A valid binary response with a poor filename still gets actual video metadata where possible.
            if kind == 'file':
                try:
                    p = media.probe(path)
                    if p.width and p.duration_ms:
                        kind, mime = 'video', 'video/mp4'
                        if not Path(filename).suffix:
                            filename += '.mp4'
                except RuntimeError:
                    pass
            ref = repo.save_file(conn, settings.media_dir, path, kind, mime)
            writes.update(conn, 'creative_intake', intake_id, status='stored', ref=ref,
                          filename=Path(filename).name, mime=mime)
        return media_info(conn, settings, asset_id=intake_id)
    except Exception as exc:
        writes.update(conn, 'creative_intake', intake_id, status='failed', error=f'{type(exc).__name__}: {exc}'[:500])
        raise


def inbox_list(conn, settings, *, project_id: int | None = None) -> dict:
    rows = conn.execute('SELECT * FROM creative_intake' + (' WHERE project_id=?' if project_id is not None else '')
                        + ' ORDER BY id DESC LIMIT 200', (project_id,) if project_id is not None else ())
    return {'assets': [_record(r) for r in rows]}


def media_info(conn, settings, *, asset_id: int, analyze: bool = False) -> dict:
    row = _get(conn, 'creative_intake', asset_id)
    row['sha256'] = row['ref']
    if row['ref']:
        row['path'] = str(media.path_for(settings.media_dir, row['ref']))
        probe = conn.execute('SELECT m.*, b.bytes FROM blob b LEFT JOIN media m ON m.blob_sha256=b.sha256 '
                             'WHERE b.sha256=?', (row['ref'],)).fetchone()
        row['probe'] = dict(probe) if probe else None
        if analyze and row['mime'].startswith('video/'):
            from wonhee import service
            from wonhee.pipeline.deps import Deps
            run_id, _ = service.node_run(conn, settings, Deps.live(settings), 'contact_sheet', {'video': row['ref']})
            outputs = service._step_out(conn, run_id, 'contact_sheet')
            if outputs:
                row['contact_sheet'] = {'ref': outputs['image'], 'run_id': run_id,
                                        'path': str(media.path_for(settings.media_dir, outputs['image']))}
        row['transcripts'] = [_record(r) for r in conn.execute(
            "SELECT * FROM creative_memory WHERE kind='transcript' AND status='confirmed' "
            "AND json_extract(source_json,'$.asset_id')=? ORDER BY id DESC", (asset_id,))]
    return row


class _CPUEmbedder:
    """Explicit device avoids letting sentence-transformers choose the shared GPU."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def available(self):
        from importlib.util import find_spec
        return find_spec('sentence_transformers') is not None

    def embed(self, texts):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device='cpu', local_files_only=True)
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=False)
        return [row.astype(np.float32) for row in vectors]


@lru_cache(maxsize=8)
def _open_store(path: str, model: str):
    from moeka.memory import DocStore
    return DocStore(path, embedder=_CPUEmbedder(model))


def _docstore(settings):
    if not settings.memory_semantic:
        return None
    return _open_store(str(settings.home / 'creative-memory' / 'search.db'), settings.memory_embedding_model)


def _serialized_index(fn):
    @functools.wraps(fn)
    def call(conn, settings):
        root = settings.home / 'creative-memory'
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'index.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return fn(conn, settings)
    return call


@_serialized_index
def _index_pending(conn, settings) -> str:
    try:
        store = _docstore(settings)
        if store is None or not store.available:
            return 'keyword'
        for r in conn.execute('SELECT * FROM creative_memory WHERE indexed=0 ORDER BY id').fetchall():
            source = f"memory:{r['id']}"
            store.clear(collection='wonhee', source=source)
            if r['status'] not in {'superseded', 'forgotten'}:
                n = store.add(r['content'], source=source, collection='wonhee',
                              tags=[f"project:{r['project_id']}", f"kind:{r['kind']}", f"status:{r['status']}"])
                if not n:
                    raise RuntimeError('semantic indexing returned no chunks')
            writes.update(conn, 'creative_memory', r['id'], indexed=1)
        writes.maintenance(conn, 'semantic', None)
        return 'hybrid'
    except Exception as exc:
        writes.maintenance(conn, 'semantic', f'{type(exc).__name__}: {exc}'[:500])
        return 'keyword'


def _export_after_write(conn, settings):
    try:
        memory_export(conn, settings)
    except OSError as exc:
        writes.maintenance(conn, 'export', str(exc)[:500])


def memory_remember(conn, settings, *, content: str, project_id: int | None = None, kind: MemoryKind = 'preference',
                    source: dict | None = None, status: Literal['confirmed', 'proposed'] = 'confirmed',
                    scope: Literal['auto', 'creator', 'project'] = 'auto') -> dict:
    content = _text(content)
    if scope not in {'auto', 'creator', 'project'}:
        raise ValueError('invalid write scope')
    if scope == 'creator':
        project_id = None
    if scope == 'project' and project_id is None:
        raise ValueError('project scope requires project_id')
    if project_id is not None:
        project_show(conn, settings, project_id=project_id)
    if status not in {'confirmed', 'proposed'}:
        raise ValueError('new memory status must be confirmed or proposed')
    if kind not in {'preference', 'constraint', 'decision', 'feedback', 'question', 'idea', 'brief', 'transcript', 'output'}:
        raise ValueError('unknown memory kind')
    source = source or {}
    # A retried tool call from the same source must not manufacture another memory.
    previous = conn.execute('SELECT * FROM creative_memory WHERE content=? AND project_id IS ? AND kind=? '
                            'AND source_json=? AND status=? LIMIT 1',
                            (content, project_id, kind, json.dumps(source, sort_keys=True), status)).fetchone()
    if previous:
        return _record(previous)
    record_id = writes.insert(conn, 'creative_memory', project_id=project_id, kind=kind, content=content,
                              status=status, source_json=json.dumps(source, sort_keys=True))
    _export_after_write(conn, settings)
    return _get(conn, 'creative_memory', record_id)


def memory_correct(conn, settings, *, memory_id: int, content: str, source: dict | None = None) -> dict:
    old = _get(conn, 'creative_memory', memory_id)
    record_id = writes.correct(conn, memory_id, _text(content), source or old['source'])
    _export_after_write(conn, settings)
    return _get(conn, 'creative_memory', record_id)


def memory_forget(conn, settings, *, memory_id: int, source: dict | None = None) -> dict:
    old = _get(conn, 'creative_memory', memory_id)
    provenance = {**old['source'], **({'forgotten_by': source} if source else {})}
    writes.update(conn, 'creative_memory', memory_id, status='forgotten', indexed=0, source_json=json.dumps(provenance))
    _export_after_write(conn, settings)
    return _get(conn, 'creative_memory', memory_id)


def memory_query(conn, settings, *, query: str = '', project_id: int | None = None, scope: MemoryScope = 'auto',
                 kind: MemoryKind | None = None, status: MemoryState | Literal['all'] | None = 'confirmed', since: str | None = None,
                 until: str | None = None, operation: QueryOperation = 'search', limit: int = 10) -> dict:
    if scope not in {'auto', 'project', 'creator', 'all'} or operation not in {'search', 'list', 'count', 'timeline'}:
        raise ValueError('invalid memory scope or operation')
    if not 1 <= limit <= 100 or len(query) > 4000:
        raise ValueError('limit must be 1..100 and query at most 4000 characters')
    where, args = ["m.status NOT IN ('superseded','forgotten')"], []
    if status in {'superseded', 'forgotten'}:
        where = ['m.status=?']
        args = [status]
    elif status and status != 'all':
        if status not in _MEMORY_STATES:
            raise ValueError('invalid memory status')
        where += ['m.status=?']
        args += [status]
    if scope == 'auto':
        where += ['(m.project_id IS NULL OR m.project_id=?)']
        args += [project_id]
    elif scope == 'project':
        if project_id is None:
            raise ValueError('project scope requires project_id')
        where += ['m.project_id=?']
        args += [project_id]
    elif scope == 'creator':
        where += ['m.project_id IS NULL']
    if kind:
        where += ['m.kind=?']
        args += [kind]
    if since:
        where += ['m.created_at>=?']
        args += [since]
    if until:
        where += ['m.created_at<=?']
        args += [until + 'T23:59:59+00:00' if re.fullmatch(r'\d{4}-\d{2}-\d{2}', until) else until]
    base = 'SELECT m.*,p.name AS project_name FROM creative_memory m LEFT JOIN creative_project p ON p.id=m.project_id WHERE '
    allowed = {r['id']: _record(r) for r in conn.execute(base + ' AND '.join(where), args)}
    words = re.findall(r'\w+', query, flags=re.UNICODE)
    # Bound and quote individual tokens; punctuation never becomes FTS query syntax.
    expression = ' OR '.join('"' + w.replace('"', '""') + '"' for w in words[:60])
    rank = {}
    if expression:
        hits = conn.execute('SELECT rowid FROM creative_memory_fts WHERE creative_memory_fts MATCH ? '
                            'ORDER BY bm25(creative_memory_fts)', (expression,))
        rank = {r[0]: i for i, r in enumerate(hits) if r[0] in allowed}
    else:
        rank = {k: i for i, k in enumerate(reversed(allowed))} if not query.strip() else {}
    backend = 'keyword'
    if operation == 'search' and query.strip():
        backend = _index_pending(conn, settings)
        if backend == 'hybrid':
            try:
                store = _docstore(settings)
                scopes = ([None, project_id] if scope == 'auto' and project_id is not None else
                          [None] if scope in {'auto', 'creator'} else [project_id] if scope == 'project' else ['*'])
                hits = []
                for selected in scopes:
                    tags = [] if selected == '*' else [f'project:{selected}']
                    if kind:
                        tags.append(f'kind:{kind}')
                    if status in {'confirmed', 'proposed'}:
                        tags.append(f'status:{status}')
                    hits.extend(store.search(query, k=min(200, max(40, limit * 8)), collection='wonhee',
                                             mode='hybrid', tags=tags))
                semantic = {}
                for i, hit in enumerate(hits):
                    if hit.source and hit.source.startswith('memory:'):
                        record_id = int(hit.source.split(':')[1])
                        if record_id in allowed and record_id not in semantic:
                            semantic[record_id] = i
                scores = {k: (1 / (60 + rank[k]) if k in rank else 0) +
                             (1 / (60 + semantic[k]) if k in semantic else 0) for k in rank.keys() | semantic.keys()}
                rank = {k: i for i, k in enumerate(sorted(scores, key=lambda x: -scores[x]))}
            except Exception as exc:
                writes.maintenance(conn, 'semantic', str(exc)[:500])
                backend = 'keyword'
    ids = list(rank)
    if operation == 'timeline':
        ids.sort(key=lambda k: (allowed[k]['created_at'], k))
    elif operation == 'search':
        ids.sort(key=lambda k: rank[k])
    else:
        ids.sort(reverse=True)
    return {'results': [allowed[k] for k in ids[:limit]] if operation != 'count' else [],
            'count': len(ids), 'backend': backend, 'operation': operation}


def memory_export(conn, settings) -> dict:
    root = settings.home / 'creative-memory'
    rows = [_record(r) for r in conn.execute('SELECT * FROM creative_memory ORDER BY id')]
    projects = [_record(r) for r in conn.execute('SELECT * FROM creative_project ORDER BY id')]
    ideas = [_record(r) for r in conn.execute('SELECT * FROM creative_idea ORDER BY id')]
    payload = {'version': 1, 'exported_at': now_iso(), 'projects': projects, 'memories': rows, 'ideas': ideas}
    write_atomic(root / 'records.json', json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Wonhee creative memory', '', 'Canonical SQLite records exported for reading and recovery.', '']
    for r in rows:
        lines += [f"## {r['id']} {r['kind']} ({r['status']})", '', r['content'], '',
                  f"Project: {r['project_id']}; source: {json.dumps(r['source'], ensure_ascii=False)}", '']
    write_atomic(root / 'MEMORY.md', '\n'.join(lines) + '\n')
    writes.maintenance(conn, 'export', None)
    return {'path': str(root / 'records.json'), 'markdown': str(root / 'MEMORY.md'), 'count': len(rows)}


def memory_reindex(conn, settings) -> dict:
    for row in conn.execute('SELECT id FROM creative_memory').fetchall():
        writes.update(conn, 'creative_memory', row[0], indexed=0)
    backend = _index_pending(conn, settings)
    memory_export(conn, settings)
    return {'backend': backend, 'pending': conn.execute('SELECT count(*) FROM creative_memory WHERE indexed=0').fetchone()[0]}


def idea_save(conn, settings, *, content: str, project_id: int | None = None, source: dict | None = None) -> dict:
    memory = memory_remember(conn, settings, content=content, project_id=project_id, kind='idea',
                             source=source, status='proposed')
    old = conn.execute('SELECT * FROM creative_idea WHERE memory_id=?', (memory['id'],)).fetchone()
    if old:
        return _record(old)
    record_id = writes.insert(conn, 'creative_idea', project_id=project_id, content=_text(content),
                              source_json=json.dumps(source or {}), memory_id=memory['id'])
    _export_after_write(conn, settings)
    return _get(conn, 'creative_idea', record_id)


def idea_list(conn, settings, *, project_id: int | None = None, status: str | None = None) -> dict:
    where, args = ['1'], []
    if project_id is not None:
        where += ['project_id=?']
        args += [project_id]
    if status:
        where += ['status=?']
        args += [status]
    return {'ideas': [_record(r) for r in conn.execute('SELECT * FROM creative_idea WHERE '+ ' AND '.join(where)
                                                      + ' ORDER BY id DESC LIMIT 200', args)]}


def idea_update(conn, settings, *, idea_id: int, status: IdeaState, source: dict | None = None) -> dict:
    if status not in _IDEA_STATES:
        raise ValueError('invalid idea state')
    idea = _get(conn, 'creative_idea', idea_id)
    provenance = {**idea['source'], **({'status_changed_by': source} if source else {})}
    writes.update(conn, 'creative_idea', idea_id, status=status, source_json=json.dumps(provenance))
    writes.update(conn, 'creative_memory', idea['memory_id'], status=('confirmed' if status in {'accepted','producing','done'}
                                                                  else 'proposed'), indexed=0, source_json=json.dumps(provenance))
    _export_after_write(conn, settings)
    return _get(conn, 'creative_idea', idea_id)


def make_submit(conn, settings, *, project_id: int, asset_id: int, request_key: str, style: str = 'legend',
                params: dict | None = None, route: dict | None = None, brief: str | None = None) -> dict:
    """Queue a reel once per logical request. A request is an instruction to make, not merely an upload."""
    from wonhee import service
    from wonhee.creative_worker import MakeOptions
    project_show(conn, settings, project_id=project_id)
    asset = media_info(conn, settings, asset_id=asset_id)
    if asset['status'] != 'stored' or asset['project_id'] != project_id or not asset['mime'].startswith('video/'):
        raise ValueError('choose a stored video asset in this creative project')
    service._reel_template(style)
    options = MakeOptions.model_validate(params or {}).model_dump(mode='json')
    key = _text(request_key, 'request_key', 500)
    old = conn.execute('SELECT * FROM creative_submission WHERE request_key=?', (key,)).fetchone()
    if old:
        previous_brief = conn.execute('SELECT content FROM creative_brief WHERE project_id=? AND revision=?',
                                      (project_id, old['brief_revision'])).fetchone()
        if brief is not None and previous_brief and previous_brief[0] != brief:
            raise ValueError('request_key already belongs to a different brief')
        if old['project_id'] != project_id or old['asset_id'] != asset_id or old['style'] != style or json.loads(old['params_json']) != options:
            raise ValueError('request_key already belongs to a different creation request')
        return make_show(conn, settings, submission_id=old['id'])
    with tx(conn):
        revision = writes.brief(conn, project_id, brief or json.dumps({'style': style, **options}, ensure_ascii=False),
                                {'request_key': key, **(route or {})})
        record_id = writes.insert(conn, 'creative_submission', request_key=key, project_id=project_id, asset_id=asset_id,
                                  style=style, params_json=json.dumps(options, sort_keys=True),
                                  route_json=json.dumps(route or {}), brief_revision=revision)
    memory_remember(conn, settings, content=brief or f'Make a {style} reel from asset {asset_id}', project_id=project_id,
                    kind='brief', source={'submission_id': record_id, 'request_key': key})
    return make_show(conn, settings, submission_id=record_id)


def make_show(conn, settings, *, submission_id: int | None = None, request_key: str | None = None) -> dict:
    if request_key:
        row = conn.execute('SELECT id FROM creative_submission WHERE request_key=?', (request_key,)).fetchone()
        if row is None:
            raise KeyError('no submission for this request')
        submission_id = row[0]
    if submission_id is None:
        return {'submissions': [_record(r) for r in conn.execute('SELECT * FROM creative_submission ORDER BY id DESC LIMIT 100')]}
    row = _get(conn, 'creative_submission', submission_id)
    row['runs'] = [dict(r) for r in conn.execute('SELECT phase,run_id FROM creative_run WHERE submission_id=?', (submission_id,))]
    row['deliveries'] = [_delivery_record(conn, settings, r) for r in conn.execute('SELECT * FROM creative_delivery WHERE submission_id=?',
                                                                               (submission_id,))]
    row['full_check'] = None
    return row


def _download_endpoint(settings) -> str:
    if settings.creative_download_url:
        return settings.creative_download_url.rstrip('/')
    try:
        return json.loads((settings.home / 'creative-outbox' / 'endpoint.json').read_text())['url'].rstrip('/')
    except (OSError, ValueError, KeyError):
        return ''


def _delivery_record(conn, settings, row) -> dict:
    row = _record(row)
    row['path'] = str(settings.home / 'creative-outbox' / row['token'] / row['filename'])
    base = _download_endpoint(settings)
    row['url'] = f"{base}/files/{row['token']}" if base else None
    row['bytes'] = conn.execute('SELECT bytes FROM blob WHERE sha256=?', (row['ref'],)).fetchone()[0]
    return row


def outbox_add(conn, settings, *, ref: str, filename: str = 'output.mp4', route: dict | None = None,
               content: str = '', project_id: int | None = None, submission_id: int | None = None,
               as_document: bool = False) -> dict:
    if not re.fullmatch(r'[0-9a-f]{64}', ref):
        raise ValueError('outbox expects a stored media sha256 ref')
    blob = conn.execute('SELECT * FROM blob WHERE sha256=?', (ref,)).fetchone()
    if blob is None or not media.path_for(settings.media_dir, ref).is_file():
        raise ValueError('output ref is missing from the store')
    filename = Path(_text(filename, 'filename', 200)).name
    if filename in {'.', '..'}:
        raise ValueError('invalid filename')
    if not Path(filename).suffix:
        filename += mimetypes.guess_extension(blob['mime'].split(';')[0]) or '.bin'
    record_id = writes.delivery(conn, ref=ref, filename=filename, mime=blob['mime'], route=route or {}, content=content,
                               project_id=project_id, submission_id=submission_id, as_document=as_document)
    row = _get(conn, 'creative_delivery', record_id)
    path = settings.home / 'creative-outbox' / row['token'] / row['filename']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        # Staging keeps immutable store data independent of a transport's filename handling.
        fd, temp = tempfile.mkstemp(dir=path.parent)
        import os
        os.close(fd)
        try:
            shutil.copyfile(media.path_for(settings.media_dir, ref), temp)
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)
    return _delivery_record(conn, settings, conn.execute('SELECT * FROM creative_delivery WHERE id=?', (record_id,)).fetchone())


def outbox_list(conn, settings, *, project_id: int | None = None) -> dict:
    sql = 'SELECT * FROM creative_delivery' + (' WHERE project_id=?' if project_id is not None else '')
    return {'outputs': [_delivery_record(conn, settings, r) for r in conn.execute(sql + ' ORDER BY id DESC LIMIT 100',
                                                                            (project_id,) if project_id is not None else ())]}


def delivery_show(conn, settings, *, delivery_id: int | None = None, transition: str | None = None,
                  message_ids: list[int] | None = None, error: str | None = None) -> dict:
    """Read deliveries, or acknowledge an actual transport send. Used by the trusted bridge."""
    if transition is not None:
        if delivery_id is None or transition not in {'sending', 'sent', 'failed', 'uncertain'}:
            raise ValueError('a valid delivery_id and transition are required')
        old = _get(conn, 'creative_delivery', delivery_id)
        valid = {'queued': {'sending'}, 'sending': {'sent','failed','uncertain'},
                 'failed': set(), 'uncertain': set(), 'sent': set()}
        if transition != old['status'] and transition not in valid[old['status']]:
            raise ValueError(f"cannot move delivery from {old['status']} to {transition}")
        if transition == 'sent' and not message_ids:
            raise ValueError('sent delivery requires transport message ids')
        writes.update(conn, 'creative_delivery', delivery_id, status=transition,
                      message_ids_json=json.dumps(message_ids or old['message_ids']), error=error,
                      attempts=old['attempts'] + int(transition == 'sending'))
    if delivery_id is not None:
        return _delivery_record(conn, settings, conn.execute('SELECT * FROM creative_delivery WHERE id=?', (delivery_id,)).fetchone())
    return {'deliveries': [_delivery_record(conn, settings, r) for r in conn.execute(
        "SELECT * FROM creative_delivery WHERE status='queued' ORDER BY id LIMIT 20")],
        'failures': [_record(r) for r in conn.execute(
            "SELECT id,project_id,route_json,error FROM creative_submission WHERE status='failed' ORDER BY id DESC LIMIT 100")]}


def delivery_retry(conn, settings, *, delivery_id: int) -> dict:
    row = _get(conn, 'creative_delivery', delivery_id)
    if row['status'] not in {'failed', 'uncertain'}:
        raise ValueError('only failed or uncertain deliveries can be explicitly retried')
    writes.update(conn, 'creative_delivery', delivery_id, status='queued', error=None)
    return delivery_show(conn, settings, delivery_id=delivery_id)


def health(conn, settings) -> dict:
    from importlib.util import find_spec
    maintenance = {r['name']: {'error': r['error'], 'updated_at': r['updated_at']}
                   for r in conn.execute('SELECT * FROM creative_maintenance')}
    return {'home': str(settings.home), 'semantic_enabled': settings.memory_semantic,
            'semantic_dependencies': all(find_spec(n) is not None for n in ('sentence_transformers', 'sqlite_vec')),
            'maintenance': maintenance, 'pending_index': conn.execute('SELECT count(*) FROM creative_memory WHERE indexed=0').fetchone()[0],
            'queued_submissions': conn.execute("SELECT count(*) FROM creative_submission WHERE status='queued'").fetchone()[0],
            'queued_deliveries': conn.execute("SELECT count(*) FROM creative_delivery WHERE status='queued'").fetchone()[0],
            'uncertain_deliveries': conn.execute("SELECT count(*) FROM creative_delivery WHERE status='uncertain'").fetchone()[0],
            'download_url': _download_endpoint(settings) or None}


ACTIONS = {name: globals()[name] for name in (
    'project_create', 'project_select', 'project_show', 'inbox_add', 'inbox_list', 'media_info',
    'idea_save', 'idea_list', 'idea_update', 'memory_remember', 'memory_query', 'memory_correct', 'memory_forget',
    'memory_export', 'memory_reindex', 'make_submit', 'make_show', 'outbox_add', 'outbox_list',
    'delivery_show', 'delivery_retry', 'health')}
