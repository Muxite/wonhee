"""Write boundary for creative records, durable submissions and derived-index checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

from wonhee.clock import now_iso
from wonhee.db.connect import tx

# Identifiers are internal constants; callers never supply SQL identifiers from a wire payload.
_TABLES = {'creative_project', 'creative_intake', 'creative_memory', 'creative_idea', 'creative_submission', 'creative_delivery',
           'post_meta', 'post_slot'}


def insert(conn, table: str, **fields) -> int:
    if table not in _TABLES:
        raise ValueError('unknown creative record table')
    columns = {r['name'] for r in conn.execute(f'PRAGMA table_info({table})')}
    if not fields.keys() <= columns:
        raise ValueError('unknown creative field')
    fields = {**fields, 'created_at': now_iso(), 'updated_at': now_iso()}
    with tx(conn):
        return conn.execute(f"INSERT INTO {table}({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                            tuple(fields.values())).lastrowid


# The trusted bridge watches this marker (its mtime) instead of polling by spawning a process. It is
# touched only when something the bridge may need to act on appears: a queued, failed or uncertain
# delivery or submission. Heartbeats and ordinary progress deliberately do not touch it.
SIGNAL_FILE = '.bridge-signal'
_SIGNAL_STATUSES = frozenset({'queued', 'failed', 'uncertain'})


def signal_bridge(conn) -> None:
    """Best effort: a missing signal only means the bridge falls back to its slow safety poll."""
    try:
        db = next(r[2] for r in conn.execute('PRAGMA database_list') if r[1] == 'main')
        if db:
            (Path(db).parent / SIGNAL_FILE).touch()
    except (OSError, StopIteration):
        pass


def update(conn, table: str, record_id: int, **fields) -> None:
    if table not in _TABLES:
        raise ValueError('unknown creative record table')
    columns = {r['name'] for r in conn.execute(f'PRAGMA table_info({table})')}
    if not fields.keys() <= columns:
        raise ValueError('unknown creative field')
    fields = {**fields, 'updated_at': now_iso()}
    with tx(conn):
        conn.execute(f"UPDATE {table} SET {','.join(k+'=?' for k in fields)} WHERE id=?",
                     (*fields.values(), record_id))
    if table in {'creative_delivery', 'creative_submission'} and fields.get('status') in _SIGNAL_STATUSES:
        signal_bridge(conn)  # after the commit, so the bridge's poll sees the change


def project_create(conn, name: str) -> int:
    with tx(conn):
        conn.execute('INSERT INTO creative_project(name,created_at,updated_at) VALUES(?,?,?) '
                     'ON CONFLICT(name) DO NOTHING', (name, now_iso(), now_iso()))
        return conn.execute('SELECT id FROM creative_project WHERE name=?', (name,)).fetchone()[0]


def project_select(conn, project_id: int | None, route_key: str) -> None:
    with tx(conn):
        if project_id is None:
            conn.execute('DELETE FROM creative_route WHERE route_key=?', (route_key,))
        else:
            conn.execute('INSERT INTO creative_route VALUES(?,?,?) ON CONFLICT(route_key) DO UPDATE SET '
                         'project_id=excluded.project_id,updated_at=excluded.updated_at',
                         (route_key, project_id, now_iso()))


def brief(conn, project_id: int, content: str, source: dict) -> int:
    with tx(conn):
        n = conn.execute('SELECT coalesce(max(revision),0)+1 FROM creative_brief WHERE project_id=?',
                         (project_id,)).fetchone()[0]
        conn.execute('INSERT INTO creative_brief(project_id,revision,content,source_json,created_at) VALUES(?,?,?,?,?)',
                     (project_id, n, content, json.dumps(source), now_iso()))
        return n


def correct(conn, memory_id: int, content: str, source: dict) -> int:
    with tx(conn):
        old = conn.execute('SELECT * FROM creative_memory WHERE id=?', (memory_id,)).fetchone()
        if old is None or old['status'] not in ('confirmed', 'proposed'):
            raise ValueError('memory is missing or no longer active')
        update(conn, 'creative_memory', memory_id, status='superseded', indexed=0)
        return insert(conn, 'creative_memory', project_id=old['project_id'], kind=old['kind'], content=content,
                      status=old['status'], source_json=json.dumps(source), supersedes=memory_id)


def post_project(conn, name: str, format: str, goal: str, slots: list[str]) -> int:
    """Create (or reuse) the project, record its format and goal, and seed every slot as empty."""
    with tx(conn):
        project_id = project_create(conn, name)
        conn.execute('INSERT INTO post_meta(project_id,format,goal) VALUES(?,?,?)', (project_id, format, goal))
        conn.executemany('INSERT OR IGNORE INTO post_slot(project_id,name,state,updated_at) VALUES(?,?,?,?)',
                         [(project_id, slot, 'empty', now_iso()) for slot in slots])
        return project_id


def post_slot(conn, project_id: int, name: str, **fields) -> None:
    """Replace a slot's state and provenance; the slot key is (project, name), so this is not the id-keyed update."""
    columns = {'state', 'ref', 'text', 'source', 'reason'}
    if not fields.keys() <= columns:
        raise ValueError('unknown post slot field')
    fields = {**fields, 'updated_at': now_iso()}
    with tx(conn):
        conn.execute(f"UPDATE post_slot SET {','.join(k+'=?' for k in fields)} WHERE project_id=? AND name=?",
                     (*fields.values(), project_id, name))


def maintenance(conn, name: str, error: str | None) -> None:
    with tx(conn):
        conn.execute('INSERT INTO creative_maintenance VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET '
                     'error=excluded.error,updated_at=excluded.updated_at', (name, error, now_iso()))


def link_run(conn, submission_id: int, phase: str, run_id: int) -> None:
    conn.execute('INSERT INTO creative_run VALUES(?,?,?)', (submission_id, phase, run_id))


def claim_submission(conn):
    with tx(conn):
        # The daemon holds an exclusive process lock; abandoned work can therefore be reconciled safely.
        row = conn.execute("SELECT * FROM creative_submission WHERE status IN ('queued','running') ORDER BY id LIMIT 1").fetchone()
        if row:
            update(conn, 'creative_submission', row['id'], status='running', heartbeat_at=time.time(),
                   attempts=row['attempts'] + 1)
        return row


def recover_deliveries(conn) -> None:
    with tx(conn):
        conn.execute("UPDATE creative_delivery SET status='uncertain',error='delivery interrupted before acknowledgement',"
                     "updated_at=? WHERE status='sending'", (now_iso(),))
    signal_bridge(conn)


def delivery(conn, *, ref, filename, mime, route, content='', project_id=None, submission_id=None, as_document=False) -> int:
    with tx(conn):
        if submission_id is not None:
            row = conn.execute('SELECT id FROM creative_delivery WHERE submission_id=? AND ref=?',
                               (submission_id, ref)).fetchone()
            if row:
                return row[0]
        new_id = insert(conn, 'creative_delivery', ref=ref, filename=filename, mime=mime,
                        token=uuid4().hex, route_json=json.dumps(route), content=content,
                        project_id=project_id, submission_id=submission_id, as_document=int(as_document))
    signal_bridge(conn)
    return new_id
