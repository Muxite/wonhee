"""Read helpers. Reads may happen anywhere; writes only in wonhee.db.repo."""
from __future__ import annotations

import sqlite3


def current_step(conn: sqlite3.Connection, run_id: int, stage: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM v_current_steps WHERE run_id=? AND stage=? ORDER BY attempt DESC LIMIT 1",
        (run_id, stage)).fetchone()


def run_cost(conn: sqlite3.Connection, run_id: int) -> float:
    return conn.execute("SELECT coalesce(sum(cost_usd), 0) FROM step WHERE run_id=?", (run_id,)).fetchone()[0]


def run_row(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"no run {run_id}")
    return row


def run_steps(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM step WHERE run_id=? ORDER BY id", (run_id,)).fetchall()


def step_flags(conn: sqlite3.Connection, step_id: int) -> list[sqlite3.Row]:
    return conn.execute("SELECT code, detail FROM step_flag WHERE step_id=? ORDER BY code", (step_id,)).fetchall()


def step_artifacts(conn: sqlite3.Connection, step_id: int) -> list[sqlite3.Row]:
    return conn.execute("SELECT blob_sha256, role FROM step_artifact WHERE step_id=?", (step_id,)).fetchall()


def trends_of_step(conn, step_id: int):
    return conn.execute(
        """SELECT t.id AS trend_id, t.name, t.summary, t.window_start, t.window_end,
                  p.example_count, p.median_plays
           FROM trend t JOIN v_trend_popularity p ON p.trend_id = t.id
           WHERE t.created_by_step = ? ORDER BY t.id""", (step_id,)).fetchall()


def primary_trend(conn, project_id: int):
    return conn.execute(
        """SELECT t.id AS trend_id, t.name, t.summary, t.window_start, t.window_end
           FROM project_trend pt JOIN trend t ON t.id = pt.trend_id
           WHERE pt.project_id = ? AND pt.role = 'primary'""", (project_id,)).fetchone()


def trend_examples(conn, trend_id: int):
    return conn.execute(
        """SELECT l.*, c.media_sha256, c.caption
           FROM trend_example te
           JOIN v_content_latest l ON l.content_id = te.content_id
           JOIN content_item c ON c.id = te.content_id
           WHERE te.trend_id = ?
           ORDER BY l.plays IS NULL, l.plays DESC, l.content_id""", (trend_id,)).fetchall()


def has_analysis(conn, sha: str, version: int) -> bool:
    return conn.execute("SELECT 1 FROM media_analysis WHERE media_sha256=? AND analyzer_version=?",
                        (sha, version)).fetchone() is not None


def latest_shotlist(conn, project_id: int):
    return conn.execute("SELECT * FROM shotlist WHERE project_id=? ORDER BY version DESC LIMIT 1", (project_id,)).fetchone()


def shots(conn, shotlist_id: int):
    return conn.execute("SELECT * FROM shot WHERE shotlist_id=? ORDER BY ord", (shotlist_id,)).fetchall()


def citations(conn, shotlist_id: int):
    return conn.execute("SELECT * FROM v_shot_lineage WHERE shotlist_id=? ORDER BY shot_ord", (shotlist_id,)).fetchall()


def project_of_run(conn, run_id: int):
    return conn.execute("SELECT p.* FROM project p JOIN run r ON r.project_id = p.id WHERE r.id=?", (run_id,)).fetchone()


def intent_row(conn, intent_id: int):
    row = conn.execute("SELECT * FROM intent WHERE id=?", (intent_id,)).fetchone()
    if row is None:
        raise KeyError(f"no intent {intent_id}")
    return row


def intent_turns(conn, intent_id: int):
    return conn.execute("SELECT * FROM intent_turn WHERE intent_id=? ORDER BY ord", (intent_id,)).fetchall()


def project_references(conn, project_id: int):
    """Reference reels of a project with their audio (external id) and posting time."""
    return conn.execute(
        """SELECT c.id AS content_id, c.external_id, c.posted_at, c.caption, l.plays, a.external_id AS audio_id
           FROM project_reference pr JOIN content_item c ON c.id = pr.content_id
           JOIN v_content_latest l ON l.content_id = c.id
           LEFT JOIN audio_track a ON a.id = c.audio_id
           WHERE pr.project_id = ? ORDER BY c.id""", (project_id,)).fetchall()


def top_hashtags(conn, content_ids: list[int], n: int) -> list[str]:
    if not content_ids:
        return []
    marks = ",".join("?" * len(content_ids))
    rows = conn.execute(
        f"""SELECT h.tag FROM content_hashtag ch JOIN hashtag h ON h.id = ch.hashtag_id
            WHERE ch.content_id IN ({marks}) GROUP BY h.tag ORDER BY count(*) DESC, h.tag LIMIT ?""",
        (*content_ids, n)).fetchall()
    return [r[0] for r in rows]


def trend_beats(conn, trend_id: int):
    return conn.execute("SELECT ord, role, description FROM trend_beat WHERE trend_id=? ORDER BY ord", (trend_id,)).fetchall()


def trend_rules(conn, trend_id: int):
    return conn.execute("SELECT ord, kind, text FROM trend_rule WHERE trend_id=? ORDER BY ord", (trend_id,)).fetchall()


def trend_source_ids(conn, trend_id: int) -> list[int]:
    return [r[0] for r in conn.execute(
        "SELECT web_source_id FROM trend_source WHERE trend_id=? ORDER BY web_source_id", (trend_id,))]


def media_style(conn, sha: str):
    return conn.execute("SELECT * FROM v_media_style WHERE media_sha256=? ORDER BY analyzer_version DESC LIMIT 1",
                        (sha,)).fetchone()


def media_analysis_row(conn, sha: str):
    return conn.execute("SELECT * FROM media_analysis WHERE media_sha256=? ORDER BY analyzer_version DESC LIMIT 1",
                        (sha,)).fetchone()


def media_shots(conn, sha: str, analyzer_version: int):
    return conn.execute("SELECT * FROM media_shot WHERE media_sha256=? AND analyzer_version=? ORDER BY ord",
                        (sha, analyzer_version)).fetchall()


def flag_meaning(conn, code: str) -> str:
    row = conn.execute("SELECT meaning FROM flag_code WHERE code=?", (code,)).fetchone()
    return row[0] if row else code
