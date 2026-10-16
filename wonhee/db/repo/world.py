from __future__ import annotations

import sqlite3
from typing import Iterable

from wonhee.db.connect import tx


def _nn(v):  # IG reports hidden counts as -1
    return None if v is None or v < 0 else int(v)


def upsert_account(conn: sqlite3.Connection, platform: str, external_id: str, handle: str,
                   relation: str | None = None) -> int:
    conn.execute(
        """INSERT INTO account(platform, external_id, handle, relation) VALUES (?,?,?, coalesce(?, 'external'))
           ON CONFLICT(platform, external_id) DO UPDATE SET
             handle = excluded.handle,
             relation = coalesce(?, account.relation)""",
        (platform, external_id, handle, relation, relation))
    return conn.execute("SELECT id FROM account WHERE platform=? AND external_id=?", (platform, external_id)).fetchone()[0]


def set_relation(conn: sqlite3.Connection, platform: str, handle: str, relation: str) -> int:
    row = conn.execute("SELECT id FROM account WHERE platform=? AND handle=?", (platform, handle)).fetchone()
    if row is None:
        raise KeyError(f"no {platform} account with handle {handle!r} seen yet")
    conn.execute("UPDATE account SET relation=? WHERE id=?", (relation, row[0]))
    return row[0]


def upsert_audio(conn: sqlite3.Connection, platform: str, external_id: str, title: str | None, artist: str | None) -> int:
    conn.execute(
        """INSERT INTO audio_track(platform, external_id, title, artist) VALUES (?,?,?,?)
           ON CONFLICT(platform, external_id) DO UPDATE SET
             title = coalesce(excluded.title, audio_track.title),
             artist = coalesce(excluded.artist, audio_track.artist)""",
        (platform, external_id, title, artist))
    return conn.execute("SELECT id FROM audio_track WHERE platform=? AND external_id=?", (platform, external_id)).fetchone()[0]


def upsert_content(conn: sqlite3.Connection, *, platform: str, external_id: str, url: str, account_id: int,
                   posted_at: str | None = None, caption: str | None = None, audio_id: int | None = None) -> int:
    conn.execute(
        """INSERT INTO content_item(platform, external_id, url, account_id, posted_at, caption, audio_id)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(platform, external_id) DO UPDATE SET
             url = excluded.url,
             posted_at = coalesce(content_item.posted_at, excluded.posted_at),
             caption = coalesce(excluded.caption, content_item.caption),
             audio_id = coalesce(excluded.audio_id, content_item.audio_id)""",
        (platform, external_id, url, account_id, posted_at, caption, audio_id))
    return conn.execute("SELECT id FROM content_item WHERE platform=? AND external_id=?", (platform, external_id)).fetchone()[0]


def set_content_media(conn: sqlite3.Connection, content_id: int, sha: str) -> None:
    conn.execute("UPDATE content_item SET media_sha256=? WHERE id=?", (sha, content_id))


def add_hashtags(conn: sqlite3.Connection, content_id: int, tags: Iterable[str]) -> None:
    with tx(conn):
        for raw in tags:
            tag = (raw or "").strip().lstrip("#").strip().lower()
            if not tag:
                continue
            conn.execute("INSERT OR IGNORE INTO hashtag(tag) VALUES (?)", (tag,))
            (hid,) = conn.execute("SELECT id FROM hashtag WHERE tag=?", (tag,)).fetchone()
            conn.execute("INSERT OR IGNORE INTO content_hashtag(content_id, hashtag_id) VALUES (?,?)", (content_id, hid))


def add_snapshot(conn: sqlite3.Connection, content_id: int, step_id: int, observed_at: str,
                 plays, likes, comments, shares) -> None:
    conn.execute(
        """INSERT INTO metric_snapshot(content_id, observed_at, plays, likes, comments, shares, step_id)
           VALUES (?,?,?,?,?,?,?) ON CONFLICT(content_id, observed_at) DO NOTHING""",
        (content_id, observed_at, _nn(plays), _nn(likes), _nn(comments), _nn(shares), step_id))


def upsert_web_source(conn: sqlite3.Connection, url: str, title: str | None, retrieved_at: str,
                      snapshot_sha: str | None) -> int:
    conn.execute(
        """INSERT INTO web_source(url, title, retrieved_at, snapshot_sha256) VALUES (?,?,?,?)
           ON CONFLICT(url) DO NOTHING""",
        (url, title, retrieved_at, snapshot_sha))
    return conn.execute("SELECT id FROM web_source WHERE url=?", (url,)).fetchone()[0]
