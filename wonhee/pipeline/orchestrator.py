"""Fixed orchestrator (design B): walk the stage list in order. An agent orchestrator (C)
can later call the same Stage functions through run_stages(start_at=…)."""
from __future__ import annotations

import json
import traceback
from datetime import date

from wonhee.db import queries, repo
from wonhee.pipeline.context import Ctx, Stage, StageFailed


def load_request(conn, run_id: int):
    from wonhee.intent.schemas import ScoutRequest

    project = queries.project_of_run(conn, run_id)
    intent = queries.intent_row(conn, project["intent_id"])
    return project, ScoutRequest.model_validate_json(intent["compiled_json"])


def run_stages(conn, settings, deps, run_id: int, stages: list[Stage], *, start_at: str | None = None,
               overrides: dict | None = None, today: date | None = None, progress=None) -> str:
    from wonhee.intent.creator import load_creator

    names = [s.name for s in stages]
    if start_at is not None and start_at not in names:
        raise ValueError(f"unknown stage {start_at!r}; stages are {names}")
    begin = names.index(start_at) if start_at else 0
    for s in stages[:begin]:
        row = queries.current_step(conn, run_id, s.name)
        if row is None or row["status"] not in ("ok", "weak"):
            raise ValueError(f"cannot resume at {start_at!r}: no completed step for {s.name!r}")

    project, request = load_request(conn, run_id)
    creator = load_creator(settings.creator_path)
    cap = queries.run_row(conn, run_id)["cost_cap_usd"]
    overrides = dict(overrides or {})
    if today is None:
        today = date.fromisoformat(overrides["today"]) if overrides.get("today") else date.today()
    repo.reopen_run(conn, run_id)

    for i, stage in enumerate(stages[begin:], start=begin):
        prev = queries.current_step(conn, run_id, stage.name)
        step_id = repo.start_step(conn, run_id, stage.name, overrides, prev["id"] if prev else None)
        for need in stage.needs:
            dep = queries.current_step(conn, run_id, need)
            if dep is not None:
                repo.link_input(conn, step_id, dep["id"])
        ctx = Ctx(conn, settings, deps, run_id, project["id"], request, creator, step_id, overrides, today)
        if progress:
            progress(f"stage {i + 1}/{len(stages)} {stage.name}…")
        try:
            status = stage.fn(ctx)
            if status not in ("ok", "weak"):
                raise StageFailed("EXCEPTION", f"stage {stage.name} returned {status!r}")
        except StageFailed as e:
            repo.add_flag(conn, step_id, e.code, e.detail)
            status = "failed"
        except Exception:
            tb = traceback.format_exc()
            sha = repo.save_bytes(conn, settings.media_dir, tb.encode(), "text/plain")
            repo.add_artifact(conn, step_id, sha, "error")
            repo.add_flag(conn, step_id, "EXCEPTION", tb.strip().splitlines()[-1][:300])
            status = "failed"
        except BaseException:  # Ctrl-C and friends: leave the run resumable, not stuck 'running'
            repo.add_flag(conn, step_id, "EXCEPTION", "interrupted")
            repo.finish_step(conn, step_id, "failed")
            repo.finish_run(conn, run_id, "stopped")
            raise
        repo.finish_step(conn, step_id, status)
        if progress:
            progress(f"  {stage.name}: {status}")
        if status == "failed":
            repo.finish_run(conn, run_id, "failed")
            return "failed"
        spent = queries.run_cost(conn, run_id)
        if spent >= cap and i < len(stages) - 1:
            repo.add_flag(conn, step_id, "COST_CAP", f"spent ${spent:.2f} of ${cap:.2f}")
            repo.finish_run(conn, run_id, "stopped")
            return "stopped"

    statuses = [queries.current_step(conn, run_id, s.name)["status"] for s in stages]
    final = "weak" if "weak" in statuses else "done"
    repo.finish_run(conn, run_id, final)
    return final
