from __future__ import annotations

import sqlite3
from pathlib import Path

from wonhee import media as files
from wonhee.clock import now_iso
from wonhee.db.connect import tx


def _put_blob(conn: sqlite3.Connection, sha: str, size: int, mime: str) -> None:
    conn.execute("INSERT OR IGNORE INTO blob(sha256, bytes, mime, stored_at) VALUES (?,?,?,?)",
                 (sha, size, mime, now_iso()))


def save_bytes(conn: sqlite3.Connection, media_dir: Path, data: bytes, mime: str) -> str:
    sha = files.store_bytes(media_dir, data)
    with tx(conn):
        _put_blob(conn, sha, len(data), mime)
    return sha


def save_file(conn: sqlite3.Connection, media_dir: Path, path: Path, kind: str, mime: str) -> str:
    sha = files.store_file(media_dir, path)
    pr = files.probe(path) if kind in ("video", "audio", "image") else files.Probe(None, None, None, None)
    with tx(conn):
        _put_blob(conn, sha, Path(path).stat().st_size, mime)
        conn.execute(
            "INSERT OR IGNORE INTO media(blob_sha256, kind, duration_ms, width, height, fps) VALUES (?,?,?,?,?,?)",
            (sha, kind, pr.duration_ms, pr.width, pr.height, pr.fps))
    return sha


def save_media_bytes(conn: sqlite3.Connection, media_dir: Path, data: bytes, kind: str, mime: str) -> str:
    """For non-probed media such as transcripts."""
    sha = files.store_bytes(media_dir, data)
    with tx(conn):
        _put_blob(conn, sha, len(data), mime)
        conn.execute("INSERT OR IGNORE INTO media(blob_sha256, kind) VALUES (?,?)", (sha, kind))
    return sha


def add_phash(conn: sqlite3.Connection, sha: str, frame_ms: int, phash_hex: str) -> None:
    conn.execute("INSERT OR IGNORE INTO media_phash(blob_sha256, frame_ms, phash) VALUES (?,?,?)",
                 (sha, frame_ms, phash_hex))


def read_bytes(media_dir: Path, sha: str) -> bytes:
    return files.path_for(media_dir, sha).read_bytes()
