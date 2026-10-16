from __future__ import annotations

import sqlite3

from wonhee.db.connect import tx


def set_step_outputs(conn: sqlite3.Connection, step_id: int, outs: dict[str, tuple[str, str]]) -> None:
    with tx(conn):
        for port, (type_, ref) in outs.items():
            conn.execute("INSERT OR REPLACE INTO step_output(step_id, port, type, ref) VALUES (?,?,?,?)",
                         (step_id, port, type_, ref))


def put_cache(conn: sqlite3.Connection, key: str, node: str, version: int, step_id: int,
              outs: dict[str, tuple[str, str]]) -> None:
    with tx(conn):
        for port, (type_, ref) in outs.items():
            conn.execute(
                "INSERT OR REPLACE INTO node_cache(key_sha256, port, type, ref, node, node_version, created_by_step)"
                " VALUES (?,?,?,?,?,?,?)", (key, port, type_, ref, node, version, step_id))
