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
    if len(cands) == 1:
        primary, secondary = cands[0]["trend_id"], []
    else:
        prompt = (f"Creator profile: {ctx.creator.model_dump_json()}\n"
                  f"Request: {ctx.request.model_dump_json()}\n\nCandidates:\n"
                  + "\n".join(_line(t) for t in cands))
        choice = ctx.llm(PickChoice, schema_name="PickChoice", system=prompts.SYSTEM_PICK, prompt=prompt)
        if choice.trend_id in ids:
            primary = choice.trend_id
            secondary = [i for i in dict.fromkeys(choice.secondary_ids) if i in ids and i != primary]
        else:
            primary = max(cands, key=lambda t: (t["median_plays"] or 0, -t["trend_id"]))["trend_id"]
            ctx.flag("INVALID_PICK", f"model chose trend {choice.trend_id}, not a candidate; used most popular")
            secondary = []
            weak = True
    repo.set_project_trends(ctx.conn, ctx.project_id, primary, secondary)
    return "weak" if weak else "ok"
