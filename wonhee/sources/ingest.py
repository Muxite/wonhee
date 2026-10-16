from __future__ import annotations

from wonhee.db import repo
from wonhee.db.connect import tx
from wonhee.sources.apify import ReelRecord


def ingest_records(conn, step_id: int, records: list[ReelRecord], observed_at: str) -> dict[str, int]:
    """Every record seen gets a content row and a metric snapshot (G6, R14), selected or not."""
    ids: dict[str, int] = {}
    with tx(conn):
        for r in records:
            acc = repo.upsert_account(conn, "instagram", r.owner_id, r.owner_handle)
            audio = repo.upsert_audio(conn, "instagram", r.audio_id, r.audio_title, r.audio_artist) if r.audio_id else None
            cid = repo.upsert_content(conn, platform="instagram", external_id=r.external_id, url=r.url,
                                      account_id=acc, posted_at=r.posted_at, caption=r.caption, audio_id=audio)
            repo.add_hashtags(conn, cid, r.hashtags)
            repo.add_snapshot(conn, cid, step_id, observed_at, r.plays, r.likes, r.comments, r.shares)
            ids[r.external_id] = cid
    return ids
