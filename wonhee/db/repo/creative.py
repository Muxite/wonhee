"""Write boundary for creative records, durable submissions and derived-index checkpoints."""
from __future__ import annotations

import json
import time
from uuid import uuid4

from wonhee.clock import now_iso
from wonhee.db.connect import tx

# Identifiers are internal constants; callers never supply SQL identifiers from a wire payload.
_TABLES = {'creative_intake', 'creative_memory', 'creative_idea', 'creative_submission', 'creative_delivery'}


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


def delivery(conn, *, ref, filename, mime, route, content='', project_id=None, submission_id=None, as_document=False) -> int:
    with tx(conn):
        if submission_id is not None:
            row = conn.execute('SELECT id FROM creative_delivery WHERE submission_id=? AND ref=?',
                               (submission_id, ref)).fetchone()
            if row:
                return row[0]
        return insert(conn, 'creative_delivery', ref=ref, filename=filename, mime=mime,
                      token=uuid4().hex, route_json=json.dumps(route), content=content,
                      project_id=project_id, submission_id=submission_id, as_document=int(as_document))
