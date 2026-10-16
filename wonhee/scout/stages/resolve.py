"""Resolve the user's reference URLs into stored content rows (a no-op without URLs)."""
from __future__ import annotations

from wonhee.clock import now_iso
from wonhee.db import repo
from wonhee.db.connect import tx
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.sources.apify import ACTOR_SCRAPER, parse_item, urls_input
from wonhee.sources.ingest import ingest_records


def run(ctx: Ctx) -> str:
    urls = list(ctx.request.reference_urls)
    if not urls:
        return "ok"
    run_input = urls_input(urls)
    items = ctx.apify(ACTOR_SCRAPER, run_input)
    ctx.save_json({"actor": ACTOR_SCRAPER, "input": run_input, "items": items}, "api_response")
    recs = [r for r in map(parse_item, items) if r]
    if not recs:
        raise StageFailed("NO_CANDIDATES", f"reference reel(s) not found: {', '.join(urls)}")
    ids = ingest_records(ctx.conn, ctx.step_id, recs, now_iso())
    with tx(ctx.conn):
        for cid in ids.values():
            repo.add_reference(ctx.conn, ctx.project_id, cid, "inspiration")
    return "ok"
