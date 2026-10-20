from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from wonhee.clock import now_iso
from wonhee.db.connect import tx


def create_intent(conn: sqlite3.Connection, raw_text: str, schema_name: str, schema_version: int, compiled: dict) -> int:
    return conn.execute(
        "INSERT INTO intent(raw_text, schema_name, schema_version, compiled_json, created_at) VALUES (?,?,?,?,?)",
        (raw_text, schema_name, schema_version, json.dumps(compiled, default=str), now_iso())).lastrowid


def update_intent(conn: sqlite3.Connection, intent_id: int, compiled: dict) -> None:
    cur = conn.execute("UPDATE intent SET compiled_json=? WHERE id=? AND confirmed_at IS NULL",
                       (json.dumps(compiled, default=str), intent_id))
    if cur.rowcount != 1:
        raise ValueError(f"intent {intent_id} is unknown or already confirmed")


def add_intent_turn(conn: sqlite3.Connection, intent_id: int, role: str, text: str, gaps: list | None) -> int:
    with tx(conn):
        (n,) = conn.execute("SELECT coalesce(max(ord), 0) + 1 FROM intent_turn WHERE intent_id=?", (intent_id,)).fetchone()
        conn.execute("INSERT INTO intent_turn(intent_id, ord, role, text, gaps_json) VALUES (?,?,?,?,?)",
                     (intent_id, n, role, text, None if gaps is None else json.dumps(gaps)))
        return n


def confirm_intent(conn: sqlite3.Connection, intent_id: int) -> None:
    cur = conn.execute("UPDATE intent SET confirmed_at=? WHERE id=? AND confirmed_at IS NULL", (now_iso(), intent_id))
    if cur.rowcount != 1:
        raise ValueError(f"intent {intent_id} is unknown or already confirmed")


def create_project(conn: sqlite3.Connection, title: str, kind: str, intent_id: int | None) -> int:
    return conn.execute("INSERT INTO project(title, kind, intent_id, created_at) VALUES (?,?,?,?)",
                        (title, kind, intent_id, now_iso())).lastrowid


def set_project_status(conn: sqlite3.Connection, project_id: int, status: str) -> None:
    conn.execute("UPDATE project SET status=? WHERE id=?", (status, project_id))


def set_project_trends(conn: sqlite3.Connection, project_id: int, primary: int, secondary: list[int]) -> None:
    with tx(conn):
        conn.execute("DELETE FROM project_trend WHERE project_id=?", (project_id,))
        conn.execute("INSERT INTO project_trend VALUES (?,?,'primary')", (project_id, primary))
        for t in secondary:
            if t != primary:
                conn.execute("INSERT OR IGNORE INTO project_trend VALUES (?,?,'secondary')", (project_id, t))


def add_reference(conn: sqlite3.Connection, project_id: int, content_id: int, role: str) -> None:
    conn.execute("INSERT OR IGNORE INTO project_reference VALUES (?,?,?)", (project_id, content_id, role))


def add_output(conn: sqlite3.Connection, project_id: int, content_id: int) -> None:
    conn.execute("INSERT OR IGNORE INTO project_output VALUES (?,?)", (project_id, content_id))


def create_trend(conn: sqlite3.Connection, *, name: str, summary: str | None, window_start: str | None,
                 window_end: str | None, step_id: int, example_ids: list[int], source_ids: list[int]) -> int:
    with tx(conn):
        tid = conn.execute(
            "INSERT INTO trend(name, summary, window_start, window_end, created_by_step) VALUES (?,?,?,?,?)",
            (name, summary, window_start, window_end, step_id)).lastrowid
        for c in example_ids:
            conn.execute("INSERT OR IGNORE INTO trend_example VALUES (?,?)", (tid, c))
        for s in source_ids:
            conn.execute("INSERT OR IGNORE INTO trend_source VALUES (?,?)", (tid, s))
        return tid


def set_trend_card(conn: sqlite3.Connection, trend_id: int, summary: str,
                   beats: list[tuple[str, str]], rules: list[tuple[str, str]]) -> None:
    with tx(conn):
        conn.execute("UPDATE trend SET summary=? WHERE id=?", (summary, trend_id))
        conn.execute("DELETE FROM trend_beat WHERE trend_id=?", (trend_id,))
        conn.execute("DELETE FROM trend_rule WHERE trend_id=?", (trend_id,))
        for i, (role, desc) in enumerate(beats, 1):
            conn.execute("INSERT INTO trend_beat VALUES (?,?,?,?)", (trend_id, i, role, desc))
        for i, (kind, text) in enumerate(rules, 1):
            conn.execute("INSERT INTO trend_rule VALUES (?,?,?,?)", (trend_id, i, kind, text))


@dataclass
class ShotIn:
    duration_ms: int
    framing: str
    action: str
    on_screen_text: str | None
    audio_cue: str | None
    content_ids: list[int] = field(default_factory=list)
    web_source_ids: list[int] = field(default_factory=list)


def create_shotlist(conn: sqlite3.Connection, project_id: int, step_id: int, shots: list[ShotIn]) -> int:
    with tx(conn):
        (v,) = conn.execute("SELECT coalesce(max(version), 0) + 1 FROM shotlist WHERE project_id=?", (project_id,)).fetchone()
        sid = conn.execute("INSERT INTO shotlist(project_id, version, created_by_step) VALUES (?,?,?)",
                           (project_id, v, step_id)).lastrowid
        for ord_, s in enumerate(shots, 1):
            conn.execute("INSERT INTO shot VALUES (?,?,?,?,?,?,?)",
                         (sid, ord_, s.duration_ms, s.framing, s.action, s.on_screen_text, s.audio_cue))
            for c in dict.fromkeys(s.content_ids):
                conn.execute("INSERT INTO shot_citation(shotlist_id, shot_ord, content_id) VALUES (?,?,?)", (sid, ord_, c))
            for w in dict.fromkeys(s.web_source_ids):
                conn.execute("INSERT INTO shot_citation(shotlist_id, shot_ord, web_source_id) VALUES (?,?,?)", (sid, ord_, w))
        return sid


@dataclass
class ShotAnalysis:
    start_ms: int
    end_ms: int
    keyframe_sha: str | None
    on_screen_text: str | None = None
    visual: str | None = None


def save_image_analyses(conn: sqlite3.Connection, *, analyzer_version: int, step_id: int,
                        rows: list[tuple[str, str | None, str | None]]) -> None:
    """One transaction of (media_sha, on_screen_text, visual) rows for stills."""
    with tx(conn):
        for sha, text, visual in rows:
            conn.execute("INSERT INTO image_analysis VALUES (?,?,?,?,?)", (sha, analyzer_version, step_id, text, visual))


def save_analysis(conn: sqlite3.Connection, *, sha: str, analyzer_version: int, step_id: int,
                  first_word_ms: int | None, language: str | None, transcript_sha: str | None,
                  shots: list[ShotAnalysis]) -> None:
    with tx(conn):
        conn.execute("INSERT INTO media_analysis VALUES (?,?,?,?,?,?)",
                     (sha, analyzer_version, step_id, first_word_ms, language, transcript_sha))
        for i, s in enumerate(shots, 1):
            conn.execute("INSERT INTO media_shot VALUES (?,?,?,?,?,?,?,?)",
                         (sha, analyzer_version, i, s.start_ms, s.end_ms, s.keyframe_sha, s.on_screen_text, s.visual))
