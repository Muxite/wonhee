"""Choose the primary (and secondary) trend from this run's current discover candidates."""
from __future__ import annotations

from pydantic import BaseModel, Field

from wonhee.db import queries, repo
from wonhee.llm import data_block
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.scout import prompts


class PickChoice(BaseModel):
    trend_id: int
    reason: str
    secondary_ids: list[int] = Field(default_factory=list)


def _line(t) -> str:
    return (f"[{t['trend_id']}] {data_block('name', t['name'])} {data_block('summary', t['summary'] or '')} "
            f"example_count={t['example_count']} median_plays={t['median_plays']}")


def run(ctx: Ctx) -> str:
    cands = queries.trends_of_step(ctx.conn, ctx.step_of("discover"))
    if not cands:
        raise StageFailed("NO_CANDIDATES", "discover produced no trends to pick from")
    ids = {t["trend_id"] for t in cands}
    weak = False
    reason, by = None, "model"
    if len(cands) == 1:
        primary, secondary = cands[0]["trend_id"], []
        by = "only_candidate"
    else:
        prompt = (f"Creator profile: {ctx.creator.model_dump_json()}\n"
                  f"Request: {ctx.request.model_dump_json()}\n\nCandidates:\n"
                  + "\n".join(_line(t) for t in cands))
        choice = ctx.llm(PickChoice, schema_name="PickChoice", system=prompts.SYSTEM_PICK, prompt=prompt)
        if choice.trend_id in ids:
            primary = choice.trend_id
            secondary = [i for i in dict.fromkeys(choice.secondary_ids) if i in ids and i != primary]
            reason = choice.reason
        else:
            primary = max(cands, key=lambda t: (t["median_plays"] or 0, -t["trend_id"]))["trend_id"]
            ctx.flag("INVALID_PICK", f"model chose trend {choice.trend_id}, not a candidate; used most popular")
            secondary = []
            weak = True
            by, reason = "fallback_most_popular", f"model chose {choice.trend_id}, not a candidate: {choice.reason}"
    # Provenance (DESIGN: pick records why it chose): the model's reason next to the final, validated choice. The raw
    # reply is already a step artifact; this is the readable decision. The schema has no per-trend "why passed" field.
    ctx.save_json({"decision": "pick", "primary": primary, "secondary": secondary, "by": by, "reason": reason,
                   "considered": sorted(ids)}, "llm_reply")
    repo.set_project_trends(ctx.conn, ctx.project_id, primary, secondary)
    return "weak" if weak else "ok"
