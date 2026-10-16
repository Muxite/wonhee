"""Turn the trend card and the creator's constraints into the final, cited shot list."""
from __future__ import annotations

from pydantic import BaseModel, Field

from wonhee.db import queries, repo
from wonhee.db.repo import ShotIn
from wonhee.llm import data_block
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.scout import prompts


class PlannedShot(BaseModel):
    duration_s: float = Field(gt=0)
    framing: str
    action: str
    on_screen_text: str | None = None
    audio_cue: str | None = None
    cite_content_ids: list[int] = Field(default_factory=list)
    cite_source_ids: list[int] = Field(default_factory=list)


class ShotPlan(BaseModel):
    shots: list[PlannedShot]


def run(ctx: Ctx) -> str:
    trend = queries.primary_trend(ctx.conn, ctx.project_id)
    if trend is None:
        raise StageFailed("EXCEPTION", "shotlist needs a primary trend")
    tid = trend["trend_id"]
    content_ids = sorted(ex["content_id"] for ex in queries.trend_examples(ctx.conn, tid))
    source_ids = queries.trend_source_ids(ctx.conn, tid)
    max_s = ctx.request.constraints.max_length_s or ctx.creator.max_length_s
    on_cam = ctx.request.constraints.on_camera
    if on_cam is None:
        on_cam = ctx.creator.on_camera

    beats = "\n".join(f"{b['ord']}. {b['role']}: {data_block('beat', b['description'])}"
                      for b in queries.trend_beats(ctx.conn, tid))
    rules = "\n".join(f"- {r['kind']}: {data_block('rule', r['text'])}" for r in queries.trend_rules(ctx.conn, tid))
    prompt = (f"Trend: {data_block('trend name', trend['name'])}\n"
              f"Summary: {data_block('trend summary', trend['summary'] or '')}\n"
              f"Beats:\n{beats}\nRules:\n{rules}\n\n"
              f"Creator profile: {ctx.creator.model_dump_json()}\n"
              f"Constraints: max length {max_s}s total, {'on-camera' if on_cam else 'off-camera (no creator on screen)'}, "
              f"tone {ctx.request.constraints.tone or 'creator default'}\n"
              f"Allowed cite_content_ids: {content_ids}\nAllowed cite_source_ids: {source_ids}\n"
              "Every shot must cite at least one allowed id.")
    plan = ctx.llm(ShotPlan, schema_name="ShotPlan", system=prompts.SYSTEM_SHOTLIST, prompt=prompt)
    if not plan.shots:
        raise StageFailed("SCHEMA_FAIL", "shot plan has no shots")

    ok_c, ok_s = set(content_ids), set(source_ids)
    weak, shots = False, []
    for n, s in enumerate(plan.shots, 1):
        cc = [c for c in dict.fromkeys(s.cite_content_ids) if c in ok_c]
        ss = [w for w in dict.fromkeys(s.cite_source_ids) if w in ok_s]
        if not cc and not ss:
            ctx.flag("UNCITED_CLAIM", f"shot {n}")
            weak = True
        shots.append(ShotIn(max(1, round(s.duration_s * 1000)), s.framing, s.action, s.on_screen_text,
                            s.audio_cue, cc, ss))
    total_s = sum(s.duration_ms for s in shots) / 1000
    if total_s > max_s:
        ctx.flag("LENGTH_OVER", f"{total_s:.1f}s planned, max {max_s}s")
        weak = True
    repo.create_shotlist(ctx.conn, ctx.project_id, ctx.step_id, shots)
    return "weak" if weak else "ok"
