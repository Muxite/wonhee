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
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from typing import Literal

from wonhee import media
from wonhee.caption_plan import CaptionCue
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


PROJECT_STATUSES = ('active', 'stashed', 'done')


def project_status(conn, settings, *, project_id: int, status: Literal['active', 'stashed', 'done']) -> dict:
    """Mark a project active, stashed (set aside, to come back to) or done. Nothing else about it changes."""
    if status not in PROJECT_STATUSES:
        raise ValueError(f'unknown project status {status!r}; expected one of: {", ".join(PROJECT_STATUSES)}')
    project_show(conn, settings, project_id=project_id)  # unknown ids fail here, with the usual message
    writes.update(conn, 'creative_project', project_id, status=status)
    return project_show(conn, settings, project_id=project_id)


def project_brief_save(conn, settings, *, project_id: int, content: str, source: dict | None = None) -> dict:
    """Save the creator's stated brief for the selected project, with source provenance."""
    content=_text(content,'brief',16000)
    project_show(conn,settings,project_id=project_id)
    previous=conn.execute('SELECT content,source_json FROM creative_brief WHERE project_id=? ORDER BY revision DESC LIMIT 1',(project_id,)).fetchone()
    if previous is None or previous['content']!=content or json.loads(previous['source_json'])!=(source or {}):
        writes.brief(conn,project_id,content,source or {})
    return project_show(conn,settings,project_id=project_id)


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
    saved=conn.execute("SELECT content,source_json FROM creative_memory WHERE project_id=? AND kind='brief' AND status='confirmed' ORDER BY id DESC LIMIT 1",(project_id,)).fetchone()
    row['saved_brief']={'content':saved['content'],'source':json.loads(saved['source_json'])} if saved else None
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


# Which post slots a received file can fill. A file that fits no empty slot, or two, is asked about, not placed.
KIND_SLOTS = {
    'image': ('photos', 'selfie'),
    'video': ('selfie', 'take', 'screen_record'),
    'audio': ('typing_audio', 'voice'),
}
# Folder intake stores each child through inbox_add; children are not slotted one by one.
_FOLDER_CHILD = ContextVar('wonhee_intake_folder_child', default=False)
_SLOT_SOURCE_LIMIT = 500  # post_slot provenance limit; longer intake sources are cut for the slot label only.


def _post_slot_states(conn, project_id: int) -> dict[str, str]:
    slots = _post_format(_post_meta(conn, project_id)['format'])
    have = {r['name']: r['state'] for r in conn.execute('SELECT name, state FROM post_slot WHERE project_id=?',
                                                       (project_id,))}
    return {slot: have.get(slot, 'empty') for slot in slots}


def _check_explicit_slot(conn, project_id: int | None, slot: str) -> None:
    """Refuse a slot before any write: it needs a post project and a slot that the post's format has."""
    if project_id is None or conn.execute('SELECT 1 FROM post_meta WHERE project_id=?', (project_id,)).fetchone() is None:
        raise ValueError('slot needs a post project; create it with post_create and pass its project_id')
    slots = _post_slot_states(conn, project_id)
    if slot not in slots:
        raise ValueError(f'unknown slot {slot!r} for this post; expected one of: {", ".join(slots)}')


def _place_intake(conn, settings, info: dict, *, project_id: int | None, slot: str | None, kind: str) -> dict:
    """After the intake is stored, set the post slot it fills. Plain projects and folder children are left as they are."""
    if _FOLDER_CHILD.get() or project_id is None or conn.execute(
            'SELECT 1 FROM post_meta WHERE project_id=?', (project_id,)).fetchone() is None:
        return info
    held = conn.execute("SELECT name FROM post_slot WHERE project_id=? AND ref=? AND state='have'",
                        (project_id, info['ref'])).fetchall()
    if slot is None and held:  # a replay of an intake that already filled its slot
        return {**info, 'slot': held[0]['name']}
    if slot is None:
        states = _post_slot_states(conn, project_id)
        fits = [s for s in KIND_SLOTS.get(kind, ()) if states.get(s) == 'empty']
        if len(fits) != 1:
            empty = [s for s, state in states.items() if state == 'empty']
            candidates = fits or empty
            text = (f'Which slot should this {kind} go in? Empty: {", ".join(candidates)}' if candidates
                    else 'This post has no empty slot; name one with slot to replace what it holds.')
            return {**info, 'slot_question': {'text': text, 'candidates': candidates}}
        slot = fits[0]
    post_slot_set(conn, settings, project_id=project_id, slot=slot, state='have', ref=info['ref'],
                  source=str(info.get('source') or '')[:_SLOT_SOURCE_LIMIT] or None)
    return {**info, 'slot': slot}


def _kind_of(mime: str | None) -> str:
    prefix = (mime or '').split('/')[0]
    return prefix if prefix in {'video', 'audio', 'image'} else 'file'


@_serialized_intake
def inbox_add(conn, settings, *, source: str, project_id: int | None = None, source_key: str | None = None,
              role: str = 'raw', metadata: dict | None = None, dest: str | None = None,
              slot: str | None = None) -> dict:
    """Record a file or Drive link in the project. In a post project the file also fills a slot: the named one, or the
    only empty slot that fits its kind. When it fits none or several, the result has slot_question and the intake stays."""
    from wonhee.intake import download_drive, drive_identity
    if project_id is not None:
        project_show(conn, settings, project_id=project_id)
    if role not in {'raw', 'reference', 'audio', 'output'}:
        raise ValueError('asset role must be raw, reference, audio or output')
    source = _text(source, 'source', 8000)
    remote = source.startswith(('http:', 'https:'))
    from wonhee.drive import folder_identity
    folder = folder_identity(source) if remote else None
    if folder:
        if slot is not None:
            raise ValueError('slot fills from one file; a folder intake fills no slot')
        return _inbox_folder(conn, settings, source=source, project_id=project_id, source_key=source_key,
                             role=role, metadata=metadata, dest=dest)
    if dest is not None:
        raise ValueError('dest is supported for folder intake; use download for a single file destination')
    if slot is not None and not _FOLDER_CHILD.get():
        _check_explicit_slot(conn, project_id, slot)  # Validate before any receipt or download is written.
    if remote:
        drive_identity(source)  # Reject unsupported links before creating a receipt or doing network I/O.
    key = source_key or uuid4().hex
    old = conn.execute('SELECT * FROM creative_intake WHERE source_key=?', (key,)).fetchone()
    if old and (old['project_id'] != project_id or old['source'] != source):
        raise ValueError('source_key already belongs to another source/project')
    if old and old['status'] == 'stored':
        info = media_info(conn, settings, asset_id=old['id'])
        return _place_intake(conn, settings, info, project_id=project_id, slot=slot, kind=_kind_of(old['mime']))
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
        info = media_info(conn, settings, asset_id=intake_id)
    except Exception as exc:
        writes.update(conn, 'creative_intake', intake_id, status='failed', error=f'{type(exc).__name__}: {exc}'[:500])
        raise
    return _place_intake(conn, settings, info, project_id=project_id, slot=slot, kind=kind)



def _inbox_folder(conn, settings, *, source, project_id, source_key, role, metadata, dest):
    from wonhee import service
    from wonhee.drive import folder_identity
    from wonhee.pipeline.deps import Deps
    destination = Path(dest).expanduser().resolve() if dest else settings.home / 'drive' / folder_identity(source)[0]
    old = conn.execute('SELECT * FROM creative_intake WHERE source_key=?', (source_key,)).fetchone()
    if old and (old['source'] != source or old['project_id'] != project_id):
        raise ValueError('source_key already belongs to another source/project')
    state = json.loads(old['metadata_json']) if old else {
        **{k: v for k, v in (metadata or {}).items() if k not in {'folder_result', 'folder_destination'}},
        'folder_destination': str(destination)}
    if state.get('folder_destination') != str(destination):
        raise ValueError('source_key already belongs to another destination')
    parent = old['id'] if old else writes.insert(conn, 'creative_intake', source_key=source_key, source=source,
        project_id=project_id, role=role, status='received', metadata_json=json.dumps(state))
    result = state.get('folder_result')
    if result is None:
        writes.update(conn, 'creative_intake', parent, status='downloading', error=None)
        try:
            result = service.drive_download(conn, settings, Deps.live(settings), source, destination)
        except Exception as exc:
            writes.update(conn, 'creative_intake', parent, status='failed', error=str(exc)[:500])
            raise
        state['folder_result'] = result
        writes.update(conn, 'creative_intake', parent, metadata_json=json.dumps(state))
    assets = []
    token = _FOLDER_CHILD.set(True)
    try:
        for file in result['files']:
            key = hashlib.sha256((source_key + ':' + file['id'] + ':' + file['relative_path']).encode()).hexdigest()
            asset = inbox_add(conn, settings, source=file['path'], project_id=project_id, source_key='drive:' + key,
                              role=role, metadata={**(metadata or {}), 'original_name': Path(file['relative_path']).name,
                                                   'relative_path': file['relative_path'], 'drive_id': file['id'],
                                                   'drive_folder_id': folder_identity(source)[0]})
            assets.append(asset)
    finally:
        _FOLDER_CHILD.reset(token)
    writes.update(conn, 'creative_intake', parent, status='stored', ref=result['manifest_ref'],
                  filename='drive-folder.json', mime='application/json', error=None)
    return {'assets': assets, 'destination': str(destination), 'run_id': result['run_id']}


def inbox_list(conn, settings, *, project_id: int | None = None, limit: int = 200, offset: int = 0) -> dict:
    if not 1 <= limit <= 1000 or offset < 0:
        raise ValueError('limit must be 1..1000 and offset must be nonnegative')
    rows = conn.execute('SELECT * FROM creative_intake' + (' WHERE project_id=?' if project_id is not None else '')
                        + ' ORDER BY id DESC LIMIT ? OFFSET ?',
                        ((project_id,) if project_id is not None else ()) + (limit, offset))
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


def _desktop_source(conn, settings, asset: dict, project_id: int, allow_portrait=False) -> dict:
    """Follow only recorded, project-owned finishing lineage; never reframe a portrait finish."""
    seen = set()
    for _ in range(12):
        if asset['ref'] in seen:
            raise ValueError('cyclic desktop source lineage; choose the original desktop asset')
        seen.add(asset['ref'])
        post = asset.get('metadata', {}).get('manifest', {}).get('postproduction', {})
        if not post:
            # Telegram re-imports preserve bytes but may not preserve intake metadata.
            receipt = conn.execute('SELECT manifest_json FROM stagecraft_delivery WHERE project_id=? AND ref=?',
                                   (project_id, asset['ref'])).fetchone()
            if receipt:
                post = json.loads(receipt[0]).get('postproduction', {})
        if not post:
            prior = conn.execute("SELECT metadata_json FROM creative_intake WHERE project_id=? AND ref=? "
                                 "AND json_extract(metadata_json,'$.manifest.postproduction.source_sha256') IS NOT NULL "
                                 'ORDER BY id DESC LIMIT 1', (project_id, asset['ref'])).fetchone()
            if prior:
                post = json.loads(prior[0]).get('manifest',{}).get('postproduction',{})
        source_ref = post.get('source_sha256') if post.get('template') in {'desktop_caption','desktop_reel'} else None
        if source_ref:
            row = conn.execute("SELECT id FROM creative_intake WHERE project_id=? AND ref=? AND status='stored' "
                               'ORDER BY id LIMIT 1', (project_id, source_ref)).fetchone()
            if not row:
                raise ValueError('original desktop is missing from this project; choose a stored original desktop')
            asset = media_info(conn, settings, asset_id=row[0])
            continue
        previous = conn.execute("SELECT asset_id FROM creative_submission WHERE project_id=? AND result_ref=? "
                                "AND style IN ('desktop_caption','desktop_reel') ORDER BY id DESC LIMIT 1",
                                (project_id, asset['ref'])).fetchone()
        if previous:
            asset = media_info(conn, settings, asset_id=previous[0])
            continue
        probe = asset.get('probe') or {}
        portrait_source = (allow_portrait and asset.get('metadata',{}).get('producer') == 'stagecraft'
                           and asset.get('metadata',{}).get('manifest',{}).get('capture_status') == 'mechanical_checks_passed'
                           and asset.get('metadata',{}).get('manifest',{}).get('capture',{}).get('sha256') == asset['ref'])
        if not probe.get('width') or not probe.get('height') or (probe['width'] < probe['height'] and not portrait_source):
            raise ValueError('desktop_caption needs the original landscape desktop; portrait source has no recoverable lineage')
        if asset['project_id'] != project_id or asset['status'] != 'stored':
            raise ValueError('original desktop must be stored in this creative project')
        return asset
    raise ValueError('desktop source lineage is too deep; choose the original desktop asset')


def _caption_options(style, params, caption, caption_lines, caption_cues=None, duration_ms=None):
    from wonhee import service
    from wonhee.creative_worker import MakeOptions
    from wonhee.edit import compose
    options = MakeOptions.model_validate(params or {}).model_dump(mode='json')
    if caption_cues is not None:
        if not caption_cues or options['caption_cues']:
            raise ValueError('supply caption_cues once, with at least one cue')
        options['caption_cues'] = [CaptionCue.model_validate(c).model_dump() for c in caption_cues]
    timed = bool(options['caption_cues'])
    if timed:
        if style != 'desktop_reel':
            raise ValueError('caption_cues require style=desktop_reel')
        if caption is not None or caption_lines is not None or options['title'] is not None or any(
                k in options['slots'] for k in ('title','metric','proof','caveat')):
            raise ValueError('timed cues cannot mix with other caption copy')
        if duration_ms is None:
            raise ValueError('caption cues need a known source duration')
        from wonhee.caption_plan import render_layout
        render_layout(options['slots'], options['caption_cues'], duration_ms)
    if caption is not None or caption_lines is not None:
        if style not in {'desktop_caption','desktop_reel'}:
            raise ValueError('caption / caption_lines require style=desktop_caption')
        if caption is not None and caption_lines is not None:
            raise ValueError('supply caption or caption_lines, not both')
        if options['title'] is not None or any(k in options['slots'] for k in ('title','metric','proof','caveat')):
            raise ValueError('caption copy conflicts with params.title or params.slots; choose one copy input')
        if caption is not None:
            options['title'] = _text(caption, 'caption', 1000)
        else:
            if not isinstance(caption_lines, list) or not 1 <= len(caption_lines) <= 4:
                raise ValueError('caption_lines needs one to four strings: hook, metric, proof, caveat')
            options['slots'].update({k:_text(v, 'caption line', 1000)
                for k,v in zip(('title','metric','proof','caveat'), caption_lines)})
    service._reel_template(style)
    if style in {'desktop_caption','desktop_reel'}:
        if options['cta'] or options['items'] or options['tighten'] is not False or options['look']:
            raise ValueError('desktop_caption preserves the desktop; use caption/caption_lines, without cta/items/tighten/look')
        given = dict(options['slots'])
        if options['title'] is not None:
            if 'title' in given and given['title'] != options['title']:
                raise ValueError('conflicting caption title values')
            given['title'] = options['title']
        for key in ('title','metric','proof','caveat'):
            if key in given and not isinstance(given[key], str):
                raise ValueError('caption slots must be strings')
        if not timed and not str(given.get('title', '')).strip():
            raise ValueError('desktop_caption needs actual copy: supply caption="your text" or '
                             'caption_lines=["hook", "metric", "proof", "caveat"]. '
                             'The brief is saved context and does not render caption text.')
        compose.load_template(style, given)  # slots and minimum readable fit checked before queueing
    return options


def make_submit(conn, settings, *, project_id: int, asset_id: int, request_key: str, style: str = 'legend',
                params: dict | None = None, route: dict | None = None, brief: str | None = None,
                caption: str | None = None, caption_lines: list[str] | None = None,
                caption_cues: list[CaptionCue] | None = None, variant: str | None = None) -> dict:
    """Queue a reel once per logical request. A request is an instruction to make, not merely an upload."""
    from wonhee import service
    project_show(conn, settings, project_id=project_id)
    asset = media_info(conn, settings, asset_id=asset_id)
    if asset['status'] != 'stored' or asset['project_id'] != project_id or not asset['mime'].startswith('video/'):
        raise ValueError('choose a stored video asset in this creative project')
    if style == 'desktop_reel' and (caption_cues is not None or (params or {}).get('caption_cues')):
        asset = _desktop_source(conn, settings, asset, project_id, allow_portrait=True)
        asset_id = asset['id']
    options = _caption_options(style, params, caption, caption_lines, caption_cues, (asset.get('probe') or {}).get('duration_ms'))
    if style in {'desktop_caption','desktop_reel'}:
        asset = _desktop_source(conn, settings, asset, project_id, allow_portrait=style=='desktop_reel')
        asset_id = asset['id']
    key = _text(request_key, 'request_key', 500)
    if variant is not None:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}', variant):
            raise ValueError('variant must be a stable 1..40 character label: letters, digits, _ or -')
        key += ':variant:' + variant
    old = conn.execute('SELECT * FROM creative_submission WHERE request_key=?', (key,)).fetchone()
    if old:
        previous_brief = conn.execute('SELECT content FROM creative_brief WHERE project_id=? AND revision=?',
                                      (project_id, old['brief_revision'])).fetchone()
        if brief is not None and previous_brief and previous_brief[0] != brief:
            raise ValueError('request_key already belongs to a different brief')
        from wonhee.creative_worker import MakeOptions
        old_options = MakeOptions.model_validate_json(old['params_json']).model_dump(mode='json')
        if old['project_id'] != project_id or old['asset_id'] != asset_id or old['style'] != style or old_options != options:
            raise ValueError('request_key already belongs to a different creation request')
        return make_show(conn, settings, submission_id=old['id'])
    with tx(conn):
        revision = writes.brief(conn, project_id, brief or json.dumps({'style': style, **options}, ensure_ascii=False),
                                {'request_key': key, **(route or {})})
        record_id = writes.insert(conn, 'creative_submission', request_key=key, project_id=project_id, asset_id=asset_id,
                                  style=style, phase=('speech' if service._reel_template(style)['reel'].get('words', True) else 'render'),
                                  params_json=json.dumps(options, sort_keys=True),
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
    row['variant'] = row['request_key'].rsplit(':variant:',1)[1] if ':variant:' in row['request_key'] else None
    row['full_check'] = None
    if row['style'] in {'desktop_caption','desktop_reel'}:
        opts = row['params']
        row['effective_caption'] = {k:opts['slots'].get(k, opts.get('title') if k == 'title' else '')
                                    for k in ('title','metric','proof','caveat')}
        row['source_asset_id'] = row['asset_id']
        row['effective_caption_cues'] = opts.get('caption_cues',[])
        row['caption_persistence'] = 'contiguous source-time cues' if opts.get('caption_cues') else 'entire video'
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
        # 'uncertain' -> 'sent' is allowed: recovery marks every in-flight send uncertain, even one a
        # live bridge is still completing, and a transport ack with message ids is real evidence.
        valid = {'queued': {'sending'}, 'sending': {'sent','failed','uncertain'},
                 'failed': set(), 'uncertain': {'sent'}, 'sent': set()}
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
            'sending_deliveries': conn.execute("SELECT count(*) FROM creative_delivery WHERE status='sending'").fetchone()[0],
            'stuck_deliveries': _stuck_deliveries(conn),
            'active_jobs': _active_jobs(conn),
            'download_url': _download_endpoint(settings) or None}


STALL_RUNNING_S = 600     # a running submission with no heartbeat for this long is stalled
STALL_QUEUED_S = 1800     # a queued submission nobody has claimed for this long is stalled


def _age_seconds(iso: str | None) -> float | None:
    from datetime import UTC, datetime
    try:
        return max(0.0, (datetime.now(UTC) - datetime.fromisoformat(str(iso))).total_seconds())
    except (TypeError, ValueError):
        return None


def _stuck_deliveries(conn, limit: int = 10) -> list[dict]:
    """Deliveries that need a human or an explicit retry: in flight, uncertain or failed."""
    rows = conn.execute(
        "SELECT id,project_id,status,filename,route_json,error,updated_at FROM creative_delivery "
        "WHERE status IN ('sending','uncertain','failed') ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        rec = _record(r)
        rec['age_s'] = _age_seconds(r['updated_at'])
        out.append(rec)
    return out


def _active_jobs(conn, limit: int = 10) -> list[dict]:
    """Queued or running reel submissions, flagged `stalled` when nothing is moving."""
    import time
    now = time.time()
    out = []
    for r in conn.execute("SELECT id,project_id,status,phase,heartbeat_at,updated_at,created_at,route_json FROM creative_submission "
                          "WHERE status IN ('queued','running') ORDER BY id LIMIT ?", (limit,)):
        if r['status'] == 'running' and r['heartbeat_at']:
            idle = now - r['heartbeat_at']
        else:
            idle = _age_seconds(r['updated_at']) or 0.0
        limit_s = STALL_RUNNING_S if r['status'] == 'running' else STALL_QUEUED_S
        out.append({'id': r['id'], 'project_id': r['project_id'], 'status': r['status'], 'phase': r['phase'],
                    'idle_s': round(idle), 'stalled': idle > limit_s, 'route': json.loads(r['route_json'] or '{}')})
    return out


# The response keys the trusted bridge (moeka) reads. A wonhee test checks every entry against the real
# responses, and moeka compares this at startup, so a rename fails loudly instead of silently dropping work.
CONTRACT_VERSION = 1
BRIDGE_SHAPES = {
    'project_create': ['id', 'name'],
    'project_select': [],
    'project_show': ['id', 'name'],
    'inbox_add': [],
    'inbox_list': ['assets'],
    'outbox_list': ['outputs'],
    'delivery_show': ['deliveries', 'failures'],
    'delivery_retry': ['id', 'status'],
    'health': ['queued_submissions', 'queued_deliveries', 'uncertain_deliveries', 'sending_deliveries',
               'stuck_deliveries', 'active_jobs'],
    'contract': ['contract_version', 'actions', 'shapes'],
}


@lru_cache(maxsize=1)
def _revision() -> str | None:
    import subprocess
    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(['git', '-C', str(root), 'describe', '--always', '--dirty', '--abbrev=10'],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def contract(conn, settings) -> dict:
    """What this wonhee offers the trusted bridge: version, source revision, actions and response keys."""
    return {'contract_version': CONTRACT_VERSION, 'revision': _revision(),
            'actions': sorted(ACTIONS), 'shapes': BRIDGE_SHAPES,
            'signal': writes.SIGNAL_FILE}  # marker in the home dir the bridge stats instead of polling



# A post is a small project with a format and named slots; each slot is required or optional for that format.
# Required slots must be `have` before a build; a rejected slot blocks whatever its requirement (with its reason).
POST_FORMATS = {
    'talking_head': {'take': True, 'lines': False},
    'image_post': {'lines': True, 'photos': True},
    'mixed_reel': {'lines': True, 'selfie': True, 'typing_audio': True, 'screen_record': False, 'voice': False},
}
PostSlotState = Literal['empty', 'have', 'rejected']
_POST_SLOT_STATES = {'empty', 'have', 'rejected'}


def _optional(value, name: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f'{name} must be at most {limit} characters')
    return value.strip() or None


def _post_meta(conn, project_id: int) -> dict:
    row = conn.execute('SELECT * FROM post_meta WHERE project_id=?', (project_id,)).fetchone()
    if row is None:
        raise ValueError(f'project {project_id} is not a post project; create it with post_create')
    return dict(row)


def _post_format(format: str) -> dict:
    if format not in POST_FORMATS:
        raise ValueError(f'unknown post format {format!r}; expected one of: {", ".join(POST_FORMATS)}')
    return POST_FORMATS[format]


def post_create(conn, settings, *, name: str, format: str, goal: str) -> dict:
    """Create a post project (or return the existing one with the same format and goal) with its slots seeded empty."""
    slots = _post_format(format)
    name, goal = _text(name, 'name', 200), _text(goal, 'goal', 2000)
    existing = conn.execute('SELECT p.id, m.format, m.goal FROM creative_project p '
                            'LEFT JOIN post_meta m ON m.project_id=p.id WHERE p.name=?', (name,)).fetchone()
    if existing is not None and existing['format'] is not None:
        if (existing['format'], existing['goal']) != (format, goal):
            raise ValueError(f'post {name!r} already exists with a different format or goal')
        project_id = existing['id']
    else:
        project_id = writes.post_project(conn, name, format, goal, list(slots))
    return {'id': project_id, 'name': name, 'format': format, 'goal': goal}


def post_slot_set(conn, settings, *, project_id: int, slot: str, state: PostSlotState, ref: str | None = None,
                  text: str | None = None, source: str | None = None, reason: str | None = None) -> dict:
    """Set one slot to empty, have (a ref or text, with its source) or rejected (with the reason it is not usable)."""
    slots = _post_format(_post_meta(conn, project_id)['format'])
    if slot not in slots:
        raise ValueError(f'unknown slot {slot!r} for this post; expected one of: {", ".join(slots)}')
    if state not in _POST_SLOT_STATES:
        raise ValueError(f'unknown slot state {state!r}; expected one of: empty, have, rejected')
    ref = _optional(ref, 'ref', 2000)
    text = _optional(text, 'text', 20000)
    source = _optional(source, 'source', 500)
    reason = _optional(reason, 'reason', 2000)
    if state == 'have' and not (ref or text):
        raise ValueError('a have slot needs a ref (file) or text')
    if state == 'rejected' and not reason:
        raise ValueError('a rejected slot needs a reason')
    if state == 'empty':
        ref = text = source = None
    if state != 'rejected':
        reason = None
    writes.post_slot(conn, project_id, slot, state=state, ref=ref, text=text, source=source, reason=reason)
    row = conn.execute('SELECT * FROM post_slot WHERE project_id=? AND name=?', (project_id, slot)).fetchone()
    return _record(row)


def post_status(conn, settings, *, project_id: int) -> str:
    """What the post has, what blocks the build, what is skipped, and the one next step."""
    meta = _post_meta(conn, project_id)
    slots = _post_format(meta['format'])
    name = _get(conn, 'creative_project', project_id)['name']
    rows = {r['name']: dict(r) for r in conn.execute('SELECT * FROM post_slot WHERE project_id=?', (project_id,))}
    have, rejected, missing, skipped = [], [], [], []
    for slot, required in slots.items():
        row = rows.get(slot, {'state': 'empty'})
        if row['state'] == 'have':
            have.append(f"{slot} ({row['ref'] or 'text'}{', from ' + row['source'] if row['source'] else ''})")
        elif row['state'] == 'rejected':
            rejected.append(f"{slot}: {row['reason']}")
        elif required:
            missing.append(slot)
        else:
            skipped.append(slot)
    allowed = not rejected and not missing
    lines = [f"Post {name!r} ({meta['format']}): {meta['goal']}",
             f"Have: {', '.join(have) or 'nothing yet'}"]
    if rejected:
        lines.append(f"Rejected: {'; '.join(rejected)}")
    if missing:
        lines.append(f"Missing required: {', '.join(missing)}")
    if skipped:
        lines.append(f"Skipped optional: {', '.join(skipped)} (a preview can build without them)")
    lines.append(f"Build: {'allowed' if allowed else 'blocked'}")
    if rejected:
        slot, _, reason = rejected[0].partition(': ')
        lines.append(f"Next: replace {slot} ({reason})")
    elif missing:
        lines.append(f"Next: fill {missing[0]}")
    elif skipped:
        lines.append(f"Next: build the {meta['format']}; {', '.join(skipped)} can come later")
    else:
        lines.append(f"Next: build the {meta['format']}")
    return '\n'.join(lines)


from wonhee.stagecraft import (stagecraft_schema, stagecraft_benchmarks, stagecraft_dataset, stagecraft_dataset_complete, stagecraft_list, stagecraft_build, stagecraft_submit, stagecraft_show, stagecraft_review, stagecraft_retry,
                              stagecraft_complete)


ACTIONS = {name: globals()[name] for name in (
    'project_create', 'project_select', 'project_show', 'project_status', 'project_brief_save', 'inbox_add', 'inbox_list', 'media_info',
    'idea_save', 'idea_list', 'idea_update', 'memory_remember', 'memory_query', 'memory_correct', 'memory_forget',
    'memory_export', 'memory_reindex', 'make_submit', 'make_show', 'outbox_add', 'outbox_list',
    'delivery_show', 'delivery_retry', 'health', 'contract', 'post_create', 'post_slot_set', 'post_status', 'stagecraft_schema', 'stagecraft_submit', 'stagecraft_show',
    'stagecraft_review', 'stagecraft_retry', 'stagecraft_benchmarks', 'stagecraft_dataset', 'stagecraft_dataset_complete', 'stagecraft_list', 'stagecraft_build', 'stagecraft_complete')}
