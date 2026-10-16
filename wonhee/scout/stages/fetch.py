"""Download the primary trend's example reels (fresh re-scrape first, so videoUrls are not expired)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from wonhee.clock import now_iso
from wonhee.db import queries, repo
from wonhee.db.connect import tx
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.sources.apify import ACTOR_SCRAPER, parse_item, urls_input
from wonhee.sources.fetch import DownloadError
from wonhee.sources.ingest import ingest_records

MIN_SAMPLE = 3


def _link(ctx: Ctx, content_id: int) -> None:
    with tx(ctx.conn):
        repo.add_reference(ctx.conn, ctx.project_id, content_id, "style_ref")


def run(ctx: Ctx) -> str:
    primary = queries.primary_trend(ctx.conn, ctx.project_id)
    if primary is None:
        raise StageFailed("EXCEPTION", "no primary trend chosen")
    count = int(ctx.param("count", ctx.settings.reels_per_trend))
    examples = queries.trend_examples(ctx.conn, primary["trend_id"])[:count]
    have = 0
    todo = []
    for ex in examples:
        if ex["media_sha256"]:
            have += 1
            _link(ctx, ex["content_id"])
        else:
            todo.append(ex)

    if todo:
        # re-scrape right before downloading: signed videoUrls expire
        run_input = urls_input([ex["url"] for ex in todo])
        items = ctx.apify(ACTOR_SCRAPER, run_input)
        ctx.save_json({"actor": ACTOR_SCRAPER, "input": run_input, "items": items}, "api_response")
        recs = [r for r in map(parse_item, items) if r]
        ingest_records(ctx.conn, ctx.step_id, recs, now_iso())
        fresh = {r.external_id: r for r in recs}
        with tempfile.TemporaryDirectory(prefix="wonhee-fetch-") as tmp:
            for ex in todo:
                rec = fresh.get(ex["external_id"])
                if rec is None:
                    ctx.flag("DOWNLOAD_FAILED", f"{ex['url']}: not returned by re-scrape")
                    continue
                dest = Path(tmp) / f"{ex['content_id']}.mp4"
                try:
                    ctx.deps.downloader.download(rec.video_url, ex["url"], dest)
                    sha = repo.save_file(ctx.conn, ctx.settings.media_dir, dest, "video", "video/mp4")
                except (DownloadError, OSError, RuntimeError) as e:
                    ctx.flag("DOWNLOAD_FAILED", f"{ex['url']}: {e}")
                    continue
                with tx(ctx.conn):
                    repo.set_content_media(ctx.conn, ex["content_id"], sha)
                    repo.add_reference(ctx.conn, ctx.project_id, ex["content_id"], "style_ref")
                have += 1
                dest.unlink(missing_ok=True)

    if have == 0:
        raise StageFailed("DOWNLOAD_FAILED", f"none of {len(examples)} example reel(s) could be downloaded")
    if have < MIN_SAMPLE:
        ctx.flag("LOW_SAMPLE", f"only {have} usable example video(s)")
        return "weak"
    return "ok"
