"""One way to open the database, so every connection has the same pragmas and functions."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


def _hamming(a: str | None, b: str | None) -> int | None:
    if a is None or b is None:
        return None
    return (int(a, 16) ^ int(b, 16)).bit_count()


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.create_function("hamming", 2, _hamming, deterministic=True)
    return conn


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE … COMMIT; rolls back on any exception. Nested use joins the outer tx."""
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException as e:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error as rb:
            log.warning("rollback failed after %r: %s", e, rb)  # the original error matters more
        raise
    conn.execute("COMMIT")


def open_db(settings) -> sqlite3.Connection:
    from wonhee.db.migrate import migrate

    settings.home.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    migrate(conn)
    return conn
