from __future__ import annotations

import json
import sqlite3

from wonhee.clock import now_iso
from wonhee.db.connect import tx


def start_run(conn: sqlite3.Connection, project_id: int, cost_cap_usd: float, orchestrator: str = "fixed") -> int:
    cur = conn.execute(
        "INSERT INTO run(project_id, orchestrator, status, cost_cap_usd, started_at) VALUES (?,?, 'running', ?, ?)",
        (project_id, orchestrator, cost_cap_usd, now_iso()))
    return cur.lastrowid


def set_run_cap(conn: sqlite3.Connection, run_id: int, usd: float) -> None:
    if not usd > 0:
        raise ValueError(f"cost cap must be above 0, got {usd}")
    cur = conn.execute("UPDATE run SET cost_cap_usd=? WHERE id=?", (float(usd), run_id))
    if cur.rowcount != 1:
        raise KeyError(f"no run {run_id}")


def reopen_run(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute("UPDATE run SET status='running', ended_at=NULL WHERE id=?", (run_id,))


def finish_run(conn: sqlite3.Connection, run_id: int, status: str) -> None:
    conn.execute("UPDATE run SET status=?, ended_at=? WHERE id=?", (status, now_iso(), run_id))


def start_step(conn: sqlite3.Connection, run_id: int, stage: str, params: dict, supersedes: int | None) -> int:
    with tx(conn):
        (prev,) = conn.execute("SELECT coalesce(max(attempt), 0) FROM step WHERE run_id=? AND stage=?",
                               (run_id, stage)).fetchone()
        cur = conn.execute(
            "INSERT INTO step(run_id, stage, attempt, params_json, supersedes_step, started_at) VALUES (?,?,?,?,?,?)",
            (run_id, stage, prev + 1, json.dumps(params, sort_keys=True), supersedes, now_iso()))
        return cur.lastrowid


def finish_step(conn: sqlite3.Connection, step_id: int, status: str) -> None:
    conn.execute("UPDATE step SET status=?, ended_at=? WHERE id=?", (status, now_iso(), step_id))


def add_cost(conn: sqlite3.Connection, step_id: int, usd: float) -> None:
    if usd:
        conn.execute("UPDATE step SET cost_usd = cost_usd + ? WHERE id=?", (float(usd), step_id))


def add_flag(conn: sqlite3.Connection, step_id: int, code: str, detail: str = "") -> None:
    conn.execute(
        """INSERT INTO step_flag(step_id, code, detail) VALUES (?,?,?)
           ON CONFLICT(step_id, code) DO UPDATE SET
             detail = CASE WHEN step_flag.detail = '' THEN excluded.detail
                           WHEN excluded.detail = '' THEN step_flag.detail
                           ELSE step_flag.detail || '; ' || excluded.detail END""",
        (step_id, code, detail))


def link_input(conn: sqlite3.Connection, step_id: int, input_step_id: int) -> None:
    conn.execute("INSERT OR IGNORE INTO step_input(step_id, input_step_id) VALUES (?,?)", (step_id, input_step_id))


def add_artifact(conn: sqlite3.Connection, step_id: int, sha: str, role: str) -> None:
    conn.execute("INSERT OR IGNORE INTO step_artifact(step_id, blob_sha256, role) VALUES (?,?,?)", (step_id, sha, role))


def record_llm_call(conn: sqlite3.Connection, *, step_id: int | None = None, intent_id: int | None = None,
                    model: str, schema_name: str, tokens_in: int, tokens_out: int,
                    cost_usd: float, valid: bool) -> int:
    cur = conn.execute(
        """INSERT INTO llm_call(step_id, intent_id, model, schema_name, tokens_in, tokens_out, cost_usd, valid, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (step_id, intent_id, model, schema_name, tokens_in, tokens_out, cost_usd, int(valid), now_iso()))
    return cur.lastrowid
