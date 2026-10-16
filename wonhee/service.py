"""What the CLI and the MCP server both call. Returns plain text; raises KeyError/ValueError for expected errors."""
from __future__ import annotations

from pathlib import Path

from wonhee.db import queries, repo
from wonhee.intent.compiler import CompileResult, compile_message
from wonhee.intent.schemas import ScoutRequest
from wonhee.pipeline.orchestrator import run_stages
from wonhee.pipeline.registry import GRAB_STAGES, SCOUT_STAGES, stages_for
from wonhee.render import render_library, render_run, render_shotlist

RELATIONS = ("ours", "friend", "external")


def ask(conn, settings, text: str, intent_id: int | None = None) -> CompileResult:
    return compile_message(conn, settings, text, intent_id=intent_id)


def _title(request: ScoutRequest) -> str:
    return request.niche or request.trend_name or (request.reference_urls[0] if request.reference_urls else "scout")


def _stages_of_run(conn, run_id: int):
    return stages_for(queries.project_of_run(conn, run_id)["kind"])


def _next_stage(conn, run_id: int) -> str:
    stages = _stages_of_run(conn, run_id)
    for s in stages:
        row = queries.current_step(conn, run_id, s.name)
        if row is None or row["status"] not in ("ok", "weak"):
            return s.name
    return stages[-1].name


def _outcome(conn, settings, run_id: int, status: str) -> str:
    project = queries.project_of_run(conn, run_id)
    if status in ("done", "weak"):
        if project["kind"] == "library":
            return render_library(conn, project["id"], settings.media_dir)
        return render_shotlist(conn, project["id"])
    text = render_run(conn, run_id, settings.media_dir)
    stage = _next_stage(conn, run_id)
    current = conn.execute("SELECT * FROM v_current_steps WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
    failed = [st for st in current if st["status"] == "failed"]
    nothing_found = any(f["code"] == "NO_CANDIDATES" for st in failed for f in queries.step_flags(conn, st["id"]))
    if status == "stopped" and not failed:
        cap = queries.run_row(conn, run_id)["cost_cap_usd"]
        hint = (f"Stopped at the cost cap (${cap:.2f}). To continue with a higher cap:\n"
                f"  wonhee resume {run_id} --from {stage} --set cost_cap_usd=<higher>")
    else:
        lines = []
        for st in failed:
            for f in queries.step_flags(conn, st["id"]):
                lines.append(f"  {st['stage']}: {queries.flag_meaning(conn, f['code'])}")
        if lines:
            lines.insert(0, "What to change:")
        if nothing_found:
            lines.append("Nothing usable turned up, so retrying the same stage will not help. Start a new scout "
                         "with a wider window or a different niche:\n  wonhee scout \"<niche>, 6 months ago\"\n"
                         f"Details: wonhee show run {run_id}")
        else:
            lines.append(f"Fix that, then retry the failed stage:\n  wonhee resume {run_id} --from {stage}\n"
                         f"Details: wonhee show run {run_id}")
        hint = "\n".join(lines)
    return f"{text}\n\n{hint}"


def _preflight(deps) -> None:
    from wonhee.pipeline.deps import Missing
    if isinstance(deps.apify, Missing):
        raise ValueError("APIFY_API_KEY is not set, and the scout cannot read Instagram without it. "
                         "Put it in .env or export it, then check with: wonhee doctor")


def go(conn, settings, deps, intent_id: int, progress=None) -> tuple[int, str]:
    row = queries.intent_row(conn, intent_id)
    if not row["confirmed_at"]:
        raise ValueError(f"intent {intent_id} is not confirmed yet — answer its questions or reply go")
    request = ScoutRequest.model_validate_json(row["compiled_json"])
    existing = conn.execute("SELECT id FROM project WHERE intent_id=? AND kind='scout' ORDER BY id LIMIT 1",
                            (intent_id,)).fetchone()
    if existing is not None:
        last = conn.execute("SELECT id, status FROM run WHERE project_id=? ORDER BY id DESC LIMIT 1",
                            (existing["id"],)).fetchone()
        if last is not None and last["status"] == "running":
            raise ValueError(f"intent {intent_id} already has run {last['id']} in progress (status running). "
                             f"Wait for it to finish, or look at it with: wonhee show run {last['id']}")
        if last is not None:
            note = f"intent {intent_id} already ran as project {existing['id']}, run {last['id']}; showing that outcome."
            return last["id"], f"{note}\n{_outcome(conn, settings, last['id'], last['status'])}"
        _preflight(deps)
        project_id = existing["id"]
    else:
        _preflight(deps)
        project_id = repo.create_project(conn, _title(request), "scout", intent_id)
    run_id = repo.start_run(conn, project_id, settings.cost_cap_usd)
    status = run_stages(conn, settings, deps, run_id, SCOUT_STAGES, progress=progress)
    return run_id, _outcome(conn, settings, run_id, status)


OVERRIDE_KEYS = ("count", "today", "cost_cap_usd", "force", "relation", "refetch")


def grab(conn, settings, deps, urls: list[str], relation: str = "external", progress=None,
         refetch: bool = False) -> tuple[int, str]:
    """Store reels by URL (ours, a friend's, anyone's): metadata, a metric snapshot, the video, its analysis."""
    import json

    from wonhee.intent.gaps import IG_URL
    from wonhee.intent.schemas import SCHEMA_NAME, SCHEMA_VERSION

    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {', '.join(RELATIONS)}")
    clean = [u.strip().split("?")[0] for u in urls]
    bad = [u for u in clean if not IG_URL.match(u)]
    if bad or not clean:
        raise ValueError(f"not an Instagram reel/post link: {bad[0] if bad else '(none given)'}")
    _preflight(deps)
    request = ScoutRequest(mode="from_reference", reference_urls=clean)
    intent_id = repo.create_intent(conn, "grab " + " ".join(clean), SCHEMA_NAME, SCHEMA_VERSION,
                                   json.loads(request.model_dump_json()))
    repo.confirm_intent(conn, intent_id)
    code = clean[0].rstrip("/").rsplit("/", 1)[-1]
    project_id = repo.create_project(conn, f"grab {code}" + (f" +{len(clean) - 1}" if len(clean) > 1 else ""),
                                     "library", intent_id)
    run_id = repo.start_run(conn, project_id, settings.cost_cap_usd)
    status = run_stages(conn, settings, deps, run_id, GRAB_STAGES, overrides={"relation": relation, "refetch": refetch},
                        progress=progress)
    return run_id, _outcome(conn, settings, run_id, status)


def resume(conn, settings, deps, run_id: int, from_stage: str, overrides: dict | None, progress=None) -> str:
    row = queries.run_row(conn, run_id)  # KeyError: no run N
    overrides = dict(overrides or {})
    force = bool(overrides.pop("force", False))
    if row["status"] == "running" and not force:
        raise ValueError(f"run {run_id} is in progress (status running). Wait for it to finish, or look at it "
                         f"with: wonhee show run {run_id}. If its process died, take it over with: "
                         f"wonhee resume {run_id} --from <stage> --set force=true")
    stages = _stages_of_run(conn, run_id)
    names = [s.name for s in stages]
    if from_stage not in names:
        raise ValueError(f"unknown stage {from_stage!r}; stages are {names}")
    unknown = sorted(set(overrides) - set(OVERRIDE_KEYS))
    if unknown:
        raise ValueError(f"unknown --set key(s) {', '.join(unknown)}; valid keys: {', '.join(OVERRIDE_KEYS)}")
    if "cost_cap_usd" in overrides:
        raw = overrides.pop("cost_cap_usd")
        try:
            cap = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"cost_cap_usd must be a number, got {raw!r}") from None
        repo.set_run_cap(conn, run_id, cap)
    status = run_stages(conn, settings, deps, run_id, stages, start_at=from_stage, overrides=overrides,
                        progress=progress)
    return _outcome(conn, settings, run_id, status)


def show_run(conn, run_id: int, media_dir: Path | None = None) -> str:
    return render_run(conn, run_id, media_dir or Path("media"))


def show_project(conn, project_id: int, media_dir: Path | None = None) -> str:
    row = conn.execute("SELECT kind FROM project WHERE id=?", (project_id,)).fetchone()
    if row is None:
        raise KeyError(f"no project {project_id}")
    if row["kind"] == "library":
        return render_library(conn, project_id, media_dir or Path("media"))
    return render_shotlist(conn, project_id)


def mark_account(conn, handle: str, relation: str) -> str:
    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {', '.join(RELATIONS)}")
    handle = handle.strip().lstrip("@")
    repo.set_relation(conn, "instagram", handle, relation)
    return f"@{handle} marked {relation}"
