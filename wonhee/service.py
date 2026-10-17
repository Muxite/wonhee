"""What the CLI and the MCP server both call. Returns plain text; raises KeyError/ValueError for expected errors."""
from __future__ import annotations

from pathlib import Path

from wonhee.db import queries, repo
from wonhee.intent.compiler import CompileResult, compile_message
from wonhee.intent.schemas import ScoutRequest
from wonhee.pipeline.orchestrator import run_stages
from wonhee.pipeline.registry import GRAB_STAGES, SCOUT_STAGES, stages_for_run
from wonhee.render import render_chain, render_library, render_run, render_shotlist

RELATIONS = ("ours", "friend", "external")


def ask(conn, settings, text: str, intent_id: int | None = None) -> CompileResult:
    return compile_message(conn, settings, text, intent_id=intent_id)


def _title(request: ScoutRequest) -> str:
    return request.niche or request.trend_name or (request.reference_urls[0] if request.reference_urls else "scout")


def _stages_of_run(conn, run_id: int):
    return stages_for_run(conn, run_id)


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
        if project["kind"] == "chain":
            return render_chain(conn, settings, run_id)
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


CHAIN_OVERRIDE_KEYS = ("no_cache", "cost_cap_usd", "force")
OVERRIDE_KEYS = ("count", "today", "cost_cap_usd", "force", "relation", "refetch", "no_cache")


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
    valid = CHAIN_OVERRIDE_KEYS if queries.project_of_run(conn, run_id)["kind"] == "chain" else OVERRIDE_KEYS
    unknown = sorted(set(overrides) - set(valid))
    if unknown:
        raise ValueError(f"unknown --set key(s) {', '.join(unknown)}; valid keys: {', '.join(valid)}")
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


# ---- nodes and chains ------------------------------------------------------------------------------------------

def _mime(path: Path) -> str:
    import mimetypes
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _literal_ref(conn, settings, base: str, path: Path) -> str:
    """Store a local file for a port of type `base` and return its sha."""
    from pydantic import ValidationError

    from wonhee.nodes.chain import ChainError
    from wonhee.nodes.subtitles import Subtitles, save_subs

    if base == "subtitles":
        try:
            subs = Subtitles.model_validate_json(path.read_bytes())
        except ValidationError as e:
            raise ChainError([f"{path}: not a valid subtitles JSON file ({e.error_count()} problem(s); "
                              f"want {{\"language\": ..., \"lines\": [{{id, start_ms, end_ms, text, source}}]}})"]) from None
        return save_subs(conn, settings.media_dir, subs)
    if base == "images":
        from wonhee.nodes.images import store_local
        try:
            return store_local(conn, settings.media_dir, path)
        except (ValueError, KeyError, TypeError) as e:
            raise ChainError([f"{path}: {e}"]) from None
    if base in ("video", "image"):
        return repo.save_file(conn, settings.media_dir, path, base, _mime(path))
    return repo.save_bytes(conn, settings.media_dir, path.read_bytes(), _mime(path))


def _resolve_paths(conn, settings, chain):
    """Replace local-file literals with stored refs (a copy). Anything else is left for validate to report."""
    from wonhee.nodes import registry
    from wonhee.nodes.chain import WIRE_RE
    from wonhee.nodes.types import is_literal, parse_type

    ids = {cn.id for cn in chain.nodes}
    out = chain.model_copy(deep=True)
    for cn in out.nodes:
        try:
            node = registry.get(cn.node)
        except KeyError:
            continue  # validate reports it
        for port, value in list(cn.in_.items()):
            m = WIRE_RE.match(value)
            if is_literal(value) or (m and m.group(1) in ids) or port not in node.inputs:
                continue
            path = Path(value).expanduser()
            base = parse_type(node.inputs[port])[0]
            if path.is_file() or (base == "images" and path.is_dir()):
                cn.in_[port] = _literal_ref(conn, settings, base, path)
    return out


def _missing_refs(conn, chain) -> list[str]:
    """Literal refs that point at nothing: a sha not in blob, a content id not in content_item."""
    from wonhee.nodes.types import CONTENT, SHA

    problems = []
    for cn in chain.nodes:
        for port, v in cn.in_.items():
            if SHA.match(v):
                if conn.execute("SELECT 1 FROM blob WHERE sha256=?", (v,)).fetchone() is None:
                    problems.append(f"{cn.id}: input '{port}': no such blob {v[:12]}…")
            elif (m := CONTENT.match(v)):
                if conn.execute("SELECT 1 FROM content_item WHERE id=?", (int(m.group(1)),)).fetchone() is None:
                    problems.append(f"{cn.id}: input '{port}': no content {m.group(1)}")
    return problems


def chain_run(conn, settings, deps, chain_obj, *, wait: bool = True, progress=None,
              no_cache: bool = False) -> tuple[int, str]:
    """Validate, record and run a chain. wait=False starts a detached child and returns at once."""
    from wonhee.nodes import detach, registry
    from wonhee.nodes.chain import Chain, ChainError, parse_chain, validate
    from wonhee.nodes.runner import stages_for_chain

    chain = chain_obj if isinstance(chain_obj, Chain) else parse_chain(chain_obj)
    if no_cache:
        chain = chain.model_copy(update={"no_cache": True})
    chain = _resolve_paths(conn, settings, chain)
    problems: list[str] = []
    try:
        validate(chain, registry.all_nodes())
    except ChainError as e:
        problems = list(e.problems)
    problems += _missing_refs(conn, chain)
    if problems:
        raise ChainError(problems)
    title = "→".join(cn.node for cn in chain.nodes)
    intent_id = repo.create_intent(conn, "chain " + title, "chain", 1, chain.model_dump(by_alias=True, mode="json"))
    repo.confirm_intent(conn, intent_id)
    project_id = repo.create_project(conn, title, "chain", intent_id)
    run_id = repo.start_run(conn, project_id, chain.cost_cap_usd if chain.cost_cap_usd is not None else settings.cost_cap_usd)
    if not wait:
        log = detach.spawn(settings, run_id)
        return run_id, (f"run {run_id} started in the background; check with: wonhee chain show {run_id} "
                        f"(log: {log})")
    status = run_stages(conn, settings, deps, run_id, stages_for_chain(chain), progress=progress)
    return run_id, _outcome(conn, settings, run_id, status)


def export(conn, settings, ref: str, dest: Path) -> list[Path]:
    """Copy a stored result into a folder for posting: a video or image as one file, an images manifest or a
    content:<id> photo post as 01.jpg, 02.jpg, ... Returns the written paths."""
    import shutil

    from wonhee.media import path_for
    from wonhee.nodes.images import Images
    from wonhee.nodes.types import CONTENT, SHA

    dest = Path(dest).expanduser()
    if m := CONTENT.match(ref):
        rows = conn.execute("SELECT media_sha256, kind FROM content_media WHERE content_id=? ORDER BY ord",
                            (int(m.group(1)),)).fetchall()
        files = [(r["media_sha256"], ".mp4" if r["kind"] == "video" else ".jpg") for r in rows]
        if not files:
            raise ValueError(f"{ref}: nothing stored for this post")
    elif SHA.match(ref):
        row = conn.execute("SELECT b.mime, m.kind FROM blob b LEFT JOIN media m ON m.blob_sha256=b.sha256 "
                           "WHERE b.sha256=?", (ref,)).fetchone()
        if row is None:
            raise ValueError(f"{ref}: not in the media store")
        if row["kind"] in ("video", "image"):
            files = [(ref, ".mp4" if row["kind"] == "video" else ".jpg")]
        else:
            try:
                images = Images.model_validate_json(path_for(settings.media_dir, ref).read_bytes())
            except ValueError:
                raise ValueError(f"{ref}: not a video, image or images manifest") from None
            files = [(s.sha, ".jpg") for s in images.items]
    else:
        raise ValueError(f"{ref}: want a 64-hex sha or content:<id>")
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (sha, ext) in enumerate(files, 1):
        p = dest / (f"{i:02d}{ext}" if len(files) > 1 else f"{sha[:12]}{ext}")
        shutil.copyfile(path_for(settings.media_dir, sha), p)
        out.append(p)
    return out


def node_run(conn, settings, deps, node: str, inputs: dict | None = None, params: dict | None = None,
             wait: bool = True, progress=None, no_cache: bool = False) -> tuple[int, str]:
    """One node as a one-node chain."""
    from wonhee.nodes import registry
    from wonhee.nodes.chain import parse_chain

    registry.get(node)  # KeyError listing the known nodes
    chain = parse_chain([{"id": node, "node": node, "in": dict(inputs or {}), "params": dict(params or {})}])
    return chain_run(conn, settings, deps, chain, wait=wait, progress=progress, no_cache=no_cache)


def chain_exec(conn, settings, deps, run_id: int, progress=None) -> str:
    """The detached child's entry: finish a chain run. No-op when done; ignores the 'running' guard (the
    run was created as running by the parent, and this process is the one meant to execute it)."""
    row = queries.run_row(conn, run_id)
    if queries.project_of_run(conn, run_id)["kind"] != "chain":
        raise ValueError(f"run {run_id} is not a chain run")
    if row["status"] in ("done", "weak"):
        return _outcome(conn, settings, run_id, row["status"])
    if conn.execute("SELECT 1 FROM step WHERE run_id=? AND status='running' LIMIT 1", (run_id,)).fetchone():
        raise ValueError(f"run {run_id} has a step in progress; another process may be executing it")
    stages = _stages_of_run(conn, run_id)
    start = _next_stage(conn, run_id)
    status = run_stages(conn, settings, deps, run_id, stages, start_at=start, progress=progress)
    return _outcome(conn, settings, run_id, status)


def chain_show(conn, settings, run_id: int) -> str:
    queries.run_row(conn, run_id)
    if queries.project_of_run(conn, run_id)["kind"] != "chain":
        raise ValueError(f"run {run_id} is not a chain run; use: wonhee show run {run_id}")
    return render_chain(conn, settings, run_id)
