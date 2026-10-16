"""Apply wonhee/db/migrations/NNNN_name.sql in order, each once, each in its own transaction."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from wonhee.clock import now_iso

MIGRATIONS = Path(__file__).parent / "migrations"
_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
FK_OFF = "-- wonhee:foreign-keys-off"
# Fails the migration (inside its transaction) if a table rebuild left dangling references.
_FK_GUARD = ("CREATE TEMP TABLE _fk_guard (violations INTEGER CHECK (violations = 0));\n"
             "INSERT INTO _fk_guard SELECT count(*) FROM pragma_foreign_key_check;\n"
             "DROP TABLE _fk_guard;")


def migrate(conn: sqlite3.Connection, directory: Path = MIGRATIONS) -> list[int]:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migration (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL) STRICT")
    done = {r[0] for r in conn.execute("SELECT version FROM schema_migration")}
    applied: list[int] = []
    for f in sorted(directory.iterdir()):
        m = _NAME.match(f.name)
        if not m:
            continue
        v = int(m.group(1))
        if v in done:
            continue
        sql = f.read_text()
        fk_off = sql.startswith(FK_OFF)  # table rebuilds: FKs off for the copy, checked before COMMIT
        guard = f"\n{_FK_GUARD}" if fk_off else ""
        try:
            if fk_off:
                conn.execute("PRAGMA foreign_keys = OFF")
            conn.executescript(f"BEGIN;\n{sql}{guard}\nINSERT INTO schema_migration VALUES ({v}, '{now_iso()}');\nCOMMIT;")
        except sqlite3.Error as e:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise RuntimeError(f"migration {f.name} failed: {e}") from e
        finally:
            if fk_off:
                conn.execute("PRAGMA foreign_keys = ON")
        applied.append(v)
    return applied
