"""Grab reels by URL into the library: metadata, a metric snapshot, the video file, and its analysis.

Used for our own posts (relation 'ours'), friends' posts and anyone else's. The URLs travel in the
intent's ScoutRequest.reference_urls; the relation travels as the run parameter 'relation'."""
from __future__ import annotations

import tempfile
from pathlib import Path

from wonhee import llm
from wonhee.clock import now_iso
from wonhee.db import queries, repo
from wonhee.db.connect import tx
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.scout.stages.analyze import ANALYZER_VERSION, analyze_video
from wonhee.sources.apify import ACTOR_SCRAPER, parse_item, urls_input
from wonhee.sources.fetch import DownloadError
from wonhee.sources.ingest import ingest_records


def project_content_ids(conn, project_id: int) -> list[int]:
    """Content a library project holds: our posts (project_output) and others' (project_reference)."""
    return [r[0] for r in conn.execute(
        """SELECT content_id FROM project_output WHERE project_id=:p
           UNION SELECT content_id FROM project_reference WHERE project_id=:p ORDER BY 1""", {"p": project_id})]


def grab(ctx: Ctx) -> str:
    urls = list(ctx.request.reference_urls)
    ids = grab_urls(ctx, urls, ctx.param("relation", "external"), ctx.param("refetch", False))
    return "weak" if ids.failed else "ok"


class GrabResult(dict):
    """external_id -> content_id, plus how many downloads failed in this call."""
    failed: int = 0


def grab_urls(ctx: Ctx, urls: list[str], relation: str, refetch: bool) -> GrabResult:
    """Fetch, ingest and download the given reels under ctx.project_id; returns external_id -> content_id."""
    if not urls:
        raise StageFailed("NO_CANDIDATES", "no URLs to grab")
    run_input = urls_input(urls)
    items = ctx.apify(ACTOR_SCRAPER, run_input)
    ctx.save_json({"actor": ACTOR_SCRAPER, "input": run_input, "items": items}, "api_response")
    recs = [r for r in map(parse_item, items) if r]
    if not recs:
        raise StageFailed("NO_CANDIDATES", f"reel(s) not found: {', '.join(urls)}")
    ids = ingest_records(ctx.conn, ctx.step_id, recs, now_iso())
    with tx(ctx.conn):
        for r in recs:
            if relation != "external":  # never downgrade an account already marked ours/friend
                repo.upsert_account(ctx.conn, "instagram", r.owner_id, r.owner_handle, relation)
            cid = ids[r.external_id]
            if relation == "ours":
                repo.add_output(ctx.conn, ctx.project_id, cid)
            else:
                repo.add_reference(ctx.conn, ctx.project_id, cid, "inspiration")

    # Download now: Apify's videoUrl is a signed link that expires.
    got = 0
    for r in recs:
        cid = ids[r.external_id]
        row = ctx.conn.execute("SELECT media_sha256 FROM content_item WHERE id=?", (cid,)).fetchone()
        if row["media_sha256"] and not refetch:
            got += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / f"{r.external_id}.mp4"
            try:
                ctx.deps.downloader.download(r.video_url, r.url, dest)
                sha = repo.save_file(ctx.conn, ctx.settings.media_dir, dest, "video", "video/mp4")
            except (DownloadError, RuntimeError) as e:
                ctx.flag("DOWNLOAD_FAILED", f"{r.url}: {e}"[:300])
                continue
        repo.set_content_media(ctx.conn, cid, sha)
        got += 1
    if got == 0:
        raise StageFailed("DOWNLOAD_FAILED", "no video could be downloaded (see DOWNLOAD_FAILED flags)")
    result = GrabResult(ids)
    result.failed = len(recs) - got
    return result


def analyze(ctx: Ctx) -> str:
    cids = project_content_ids(ctx.conn, ctx.project_id)
    shas = [r[0] for r in ctx.conn.execute(
        f"SELECT DISTINCT media_sha256 FROM content_item WHERE media_sha256 IS NOT NULL AND id IN "
        f"({','.join('?' * len(cids))})", cids)] if cids else []
    todo = [s for s in shas if not queries.has_analysis(ctx.conn, s, ANALYZER_VERSION)]
    if todo:
        ensure = getattr(ctx.deps.transcriber, "ensure_ready", None)
        if ensure:
            ensure()  # a broken GPU setup fails once, up front
    failed = 0
    for sha in todo:
        try:
            analyze_video(ctx, sha)
        except llm.LLMError:
            raise
        except Exception as e:  # noqa: BLE001 - one bad video must not sink the others
            if isinstance(e, RuntimeError) and "CUDA" in str(e):
                raise
            ctx.flag("ANALYZE_FAILED", f"{sha[:12]}: {e}"[:300])
            failed += 1
    if shas and failed == len(shas):
        raise StageFailed("ANALYZE_FAILED", "every video failed analysis (see flags)")
    return "weak" if failed else "ok"
