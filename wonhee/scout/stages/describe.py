"""Turn the primary trend's analysed examples into a trend card: summary, beats, invariants, variations."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from wonhee.db import queries, repo
from wonhee.llm import data_block
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.scout import prompts


class Beat(BaseModel):
    role: Literal["hook", "setup", "turn", "payoff", "outro"]
    description: str
    evidence_content_ids: list[int] = Field(default_factory=list)


class TrendCard(BaseModel):
    summary: str
    beats: list[Beat]
    invariants: list[str] = Field(default_factory=list)
    variations: list[str] = Field(default_factory=list)


def _transcript_text(ctx: Ctx, sha: str | None) -> str:
    if not sha:
        return ""
    try:
        data = json.loads(repo.read_bytes(ctx.settings.media_dir, sha))
    except (OSError, ValueError):
        ctx.flag("LOW_SAMPLE", f"transcript unreadable for {sha[:12]}")
        return ""
    return " ".join(s.get("text", "") for s in data.get("segments", []))


def _example_block(ctx: Ctx, ex) -> str:
    cid = ex["content_id"]
    lines = [f"example content_id={cid} plays={ex['plays']}"]
    sha = ex["media_sha256"]
    ana = queries.media_analysis_row(ctx.conn, sha) if sha else None
    if ana is None:
        lines.append("  (not analysed)")
        return "\n".join(lines)
    style = queries.media_style(ctx.conn, sha)
    if style is not None:
        lines.append("  style: " + ", ".join(f"{k}={style[k]}" for k in style.keys()
                                             if k not in ("media_sha256", "analyzer_version")))
    lines.append("  " + data_block(f"transcript {cid}", _transcript_text(ctx, ana["transcript_sha256"])))
    for s in queries.media_shots(ctx.conn, sha, ana["analyzer_version"]):
        lines.append(f"  shot {s['ord']} ({s['start_ms']}-{s['end_ms']} ms): "
                     + data_block(f"on-screen text {cid}.{s['ord']}", s["on_screen_text"] or "") + " "
                     + data_block(f"visual {cid}.{s['ord']}", s["visual"] or ""))
    return "\n".join(lines)


def run(ctx: Ctx) -> str:
    trend = queries.primary_trend(ctx.conn, ctx.project_id)
    if trend is None:
        raise StageFailed("EXCEPTION", "describe needs a primary trend")
    examples = queries.trend_examples(ctx.conn, trend["trend_id"])
    allowed = {ex["content_id"] for ex in examples}
    prompt = (f"Trend: {data_block('trend name', trend['name'])}\n"
              f"Summary so far: {data_block('trend summary', trend['summary'] or '')}\n\n"
              "Examples (cite them by content_id in each beat's evidence_content_ids):\n"
              + "\n".join(_example_block(ctx, ex) for ex in examples))
    card = ctx.llm(TrendCard, schema_name="TrendCard", system=prompts.SYSTEM_DESCRIBE, prompt=prompt)

    if not card.beats:
        raise StageFailed("SCHEMA_FAIL", "trend card has no beats")
    weak = False
    beats = []
    for b in card.beats:
        b.evidence_content_ids = [c for c in dict.fromkeys(b.evidence_content_ids) if c in allowed]
        if not b.evidence_content_ids:
            ctx.flag("UNCITED_CLAIM", b.role)
            weak = True
        beats.append((b.role, b.description))
    rules = [("invariant", t) for t in card.invariants] + [("variation", t) for t in card.variations]
    repo.set_trend_card(ctx.conn, trend["trend_id"], card.summary, beats, rules)
    return "weak" if weak else "ok"
