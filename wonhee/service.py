"""What the CLI and the MCP server both call. Returns plain text; raises KeyError/ValueError for expected errors."""
from __future__ import annotations

import re
from pathlib import Path

from wonhee.db import queries, repo
from wonhee.fsutil import unique_dir, write_atomic, write_atomic_bytes
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


def larp(conn, settings, deps, task: str, *, image: str | None = None, video: str | None = None,
         pick: int | None = None, engine: str = "remote", registers: list[str] | None = None) -> str:
    """Frame a mundane task as professional moments; with an image or video, also render the chosen line onto it."""
    import json

    from wonhee.media import path_for

    params = {"task": task, "engine": engine, **({"registers": registers} if registers else {})}
    rid, text = node_run(conn, settings, deps, "larp", {}, params)
    step = queries.current_step(conn, rid, "larp")
    if step is None or step["status"] not in ("ok", "weak"):
        return text
    data = json.loads(path_for(settings.media_dir, queries.step_outputs(conn, step["id"])["lines"]["ref"]).read_text(encoding="utf-8"))
    fr = data["framings"]
    chosen = pick if pick is not None and 0 <= pick < len(fr) else data["best"]
    out = [f"task: {data['task']}", ""]
    for i, f in enumerate(fr):
        out.append(f"{'*' if i == chosen else ' '}{i} [{f['kind']}] {f['overlay']}")
        out.append(f"     {f['caption']}")
    out += ["", f"hashtags: {' '.join(data['hashtags'])}",
            f"render: preset={data['text_preset']} position={data['text_position']} look={data['look'] or '-'}",
            f"shot: {data['shot']}"]
    target = ("text_image", "image", image) if image else ("text_video", "video", video) if video else None
    if target:
        node, port, src = target
        tp = {"text": fr[chosen]["overlay"], "preset": data["text_preset"], "position": data["text_position"]}
        if data["look"]:
            tp["look"] = data["look"]
        _, rendered = node_run(conn, settings, deps, node, {port: src}, tp)
        out += ["", rendered]
    return "\n".join(out)


def _teardowns(conn, settings, deps, content_ids, since, relation, progress):
    """Teardowns for stored posts (cached per post). A failing post is dropped and the rest re-run.
    Returns (teardowns, metrics, failed + not reached, last run id)."""
    import json

    from wonhee.media import path_for

    if not content_ids:
        q = ["""SELECT DISTINCT l.content_id FROM v_content_latest l JOIN content_item c ON c.id=l.content_id
                WHERE (c.media_sha256 IS NOT NULL OR EXISTS (SELECT 1 FROM content_media m WHERE m.content_id=c.id))"""]
        args: list = []
        if since:
            q.append("AND l.observed_at >= ?"); args.append(since)
        if relation:
            q.append("AND l.relation = ?"); args.append(relation)
        content_ids = [r[0] for r in conn.execute(" ".join(q) + " ORDER BY 1", args)]
    if not content_ids:
        raise ValueError("no stored posts match; grab some first")
    remaining, failed, done, last_run = list(dict.fromkeys(content_ids)), [], {}, None
    while remaining:
        chain = {"nodes": [{"id": f"t{c}", "node": "teardown", "in": {"content": f"content:{c}"}} for c in remaining],
                 "cost_cap_usd": max(settings.cost_cap_usd, 0.01 * len(remaining))}
        last_run, _ = chain_run(conn, settings, deps, chain, progress=progress)
        bad = None
        for c in remaining:
            step = queries.current_step(conn, last_run, f"t{c}")
            if step is None:
                continue
            if step["status"] in ("ok", "weak"):
                ref = queries.step_outputs(conn, step["id"])["teardown"]["ref"]
                done[c] = json.loads(path_for(settings.media_dir, ref).read_text(encoding="utf-8"))
            elif step["status"] == "failed":
                bad = c
        if bad is None:
            break  # finished, or stopped at the cost cap (the report says what is missing)
        failed.append(bad)
        remaining = [c for c in remaining if c != bad and c not in done]
    tds = [done[c] for c in content_ids if c in done]
    if not tds:
        raise ValueError(f"no teardown succeeded (see wonhee chain show {last_run})")
    metrics = {r["content_id"]: dict(r) for r in conn.execute(
        f"SELECT content_id, plays, likes FROM v_content_latest WHERE content_id IN ({','.join('?' * len(tds))})",
        [t["content_id"] for t in tds])}
    skipped = [c for c in content_ids if c not in done and c not in failed]
    return tds, metrics, failed + skipped, last_run


def _run_cost(conn, run_id) -> float:
    return conn.execute("SELECT coalesce(sum(cost_usd),0) FROM step WHERE run_id=?", (run_id,)).fetchone()[0]


def inspire(conn, settings, deps, content_ids: list[int] | None = None, since: str | None = None,
            relation: str | None = None, progress=None) -> tuple[str, Path | None]:
    """Tear down stored posts (cached per post) and write one actionable report. Returns (summary, report path).

    With no ids: every post that has media, optionally only those observed since a date / of one relation.
    A post whose teardown fails is dropped and the rest re-run (finished teardowns come from the cache)."""
    import json
    from datetime import date

    from wonhee.inspire import report

    tds, metrics, missing, last_run = _teardowns(conn, settings, deps, content_ids, since, relation, progress)
    from wonhee.inspire import looks
    md = report.build(tds, metrics, missing)
    md += "\n" + looks.section(conn, settings.media_dir, [t["content_id"] for t in tds],
                               {t["content_id"]: t["summary"] for t in tds})
    out_dir = unique_dir(settings.home / "inspire", f"{date.today().isoformat()}-run{last_run}")
    write_atomic(out_dir / "report.md", md)
    write_atomic(out_dir / "teardowns.json", json.dumps(tds, ensure_ascii=False, indent=1))
    head = md.split("## Best to replicate")[0].strip()
    return (f"{head}\n\n{len(tds)} teardown(s), {len(missing)} failed or not reached; last run "
            f"{last_run} (${_run_cost(conn, last_run):.3f}). Full report: {out_dir / 'report.md'}"), out_dir / "report.md"


def ideate(conn, settings, deps, *, n: int = 20, focus: str | None = None, engine: str = "remote",
           content_ids: list[int] | None = None, since: str | None = None, relation: str | None = None,
           progress=None) -> tuple[str, Path | None]:
    """Ideas for posts to make and the b-roll to film, from inspiration teardowns and the creator's own brief
    (`<home>/brief/*.md`), kept apart so each idea's origin is known. Returns (summary, report path)."""
    import json
    from datetime import date

    from wonhee.inspire import brief as bf
    from wonhee.inspire import ideas as ix
    from wonhee.media import path_for
    from wonhee.nodes.subtitles import save_text

    tds, metrics, missing, _ = _teardowns(conn, settings, deps, content_ids, since, relation, progress)
    items = bf.load(settings.home / "brief")
    refs = {"inspiration": save_text(conn, settings.media_dir, ix.digest(tds, metrics)),
            "brief": save_text(conn, settings.media_dir, bf.render(items))}
    params = {"n": n, "engine": engine, **({"focus": focus} if focus else {})}
    rid, text = node_run(conn, settings, deps, "ideate", refs, params, progress=progress)
    step = queries.current_step(conn, rid, "ideate")
    if step is None or step["status"] not in ("ok", "weak"):
        return text, None
    data = json.loads(path_for(settings.media_dir, queries.step_outputs(conn, step["id"])["ideas"]["ref"]).read_text(encoding="utf-8"))
    flags = [f"{r['code']}: {r['detail']}" for r in conn.execute(
        "SELECT code, detail FROM step_flag WHERE step_id=? AND code IN ('UNKNOWN_REF','IDEA_COPY')", (step["id"],))]
    header = (f"Sources: {len(tds)} inspiration teardowns, {len(items)} brief items "
              f"({', '.join(sorted({i.source for i in items})) or 'no brief files'}). Run {rid}, "
              f"${_run_cost(conn, rid):.3f}, engine {engine}." + (f" Focus: {focus}." if focus else ""))
    if flags:
        header += "\n\nDropped by checks:\n" + "\n".join(f"- {f}" for f in flags)
    md = ix.report(data, {t["content_id"]: t["summary"] for t in tds}, {i.id: i.text for i in items}, header)
    out_dir = unique_dir(settings.home / "ideate", f"{date.today().isoformat()}-run{rid}")
    write_atomic(out_dir / "ideas.md", md)
    write_atomic(out_dir / "ideas.json", json.dumps(data, ensure_ascii=False, indent=1))
    head = md.split("\n## ")[0].strip()
    return f"{head}\n\nFull list: {out_dir / 'ideas.md'}", out_dir / "ideas.md"


def _step_out(conn, rid: int, node: str) -> dict[str, str] | None:
    step = queries.current_step(conn, rid, node)
    if step is None or step["status"] not in ("ok", "weak"):
        return None
    return {k: v["ref"] for k, v in queries.step_outputs(conn, step["id"]).items()}


def _ref_images(conn, settings, refs: list[str]) -> str:
    """An images manifest from folders / image files / content:<id> photo posts, in the order given."""
    from wonhee.inspire.looks import post_images
    from wonhee.nodes import images as im
    from wonhee.nodes.types import CONTENT

    items = []
    for r in refs:
        if m := CONTENT.match(r):
            paths = post_images(conn, settings.media_dir, int(m.group(1)))  # photos, else video keyframes
            if not paths:
                raise ValueError(f"{r}: no stored images or analysed keyframes (grab it first)")
            items += [im.Slide(sha=p.name) for p in paths]
        else:
            items += im.load_images(settings.media_dir, im.store_local(conn, settings.media_dir,
                                                                       Path(r).expanduser())).items
    if not items:
        raise ValueError("no reference images")
    return im.save_images(conn, settings.media_dir, im.Images(items=items))


def vibe(conn, settings, deps, refs: list[str], photos: list[Path], *, name: str = "look", engine: str = "local",
         guide: bool = True, guide_engine: str = "remote", overrides: dict | None = None,
         progress=None) -> tuple[str, Path | None]:
    """Learn a look from reference photos, write a shooting guide for it, and move each photo toward it: a vision
    model picks the edit knobs per photo (engine local = $0) and the edit itself is deterministic. Writes
    `<home>/vibe/<date>-<name>/`: the edited photos, a before/after sheet and report.md. Returns (summary, dir)."""
    import json
    from datetime import date

    import cv2
    import numpy as np

    from wonhee.edit import style
    from wonhee.media import path_for
    from wonhee.nodes.vibe import ShotGuide, VibeParams

    d = settings.media_dir
    refs_sha = _ref_images(conn, settings, refs)
    rid, text = node_run(conn, settings, deps, "style_profile", {"images": refs_sha}, {"name": name}, progress=progress)
    out = _step_out(conn, rid, "style_profile")
    if out is None:
        return text, None
    prof_sha = out["profile"]
    prof = style.Profile.model_validate_json(path_for(d, prof_sha).read_text(encoding="utf-8"))
    runs, gd = [rid], None
    if guide:
        rid, text = node_run(conn, settings, deps, "shot_guide", {"images": refs_sha, "profile": prof_sha},
                             {"engine": guide_engine}, progress=progress)
        runs.append(rid)
        if (o := _step_out(conn, rid, "shot_guide")) is not None:
            gd = ShotGuide.model_validate_json(path_for(d, o["guide"]).read_text(encoding="utf-8"))
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError(f"name {name!r} has no letters or digits to build a folder name from")
    out_dir = unique_dir(settings.home / "vibe", f"{date.today().isoformat()}-{slug}")
    rows, pairs = [], []
    for n, photo in enumerate(photos, 1):
        sha = repo.save_file(conn, d, Path(photo).expanduser(), "image", "image/jpeg")
        rid, text = node_run(conn, settings, deps, "vibe_review", {"image": sha, "profile": prof_sha},
                             {"engine": engine}, progress=progress)
        runs.append(rid)
        o = _step_out(conn, rid, "vibe_review")
        review = json.loads(path_for(d, o["review"]).read_text(encoding="utf-8")) if o else None
        params = {**(review["params"] if review else VibeParams().model_dump()), **(overrides or {})}
        rid, text = node_run(conn, settings, deps, "vibe_image", {"image": sha, "profile": prof_sha}, params,
                             progress=progress)
        runs.append(rid)
        o2 = _step_out(conn, rid, "vibe_image")
        if o2 is None:
            rows.append((photo, review, params, None, text, None))
            continue
        dest = out_dir / f"{n:02d}-{Path(photo).stem}.jpg"
        dest.write_bytes(path_for(d, o2["image"]).read_bytes())
        notes = json.loads(path_for(d, o2["notes"]).read_text(encoding="utf-8")) if o2.get("notes") else None
        rows.append((photo, review, params, dest, None, notes))
        pairs.append((style.load(photo), style.load(dest)))
    if pairs:
        tiles = []
        for a, b in pairs:
            h = 600
            tiles.append(np.hstack([cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h)) for x in (a, b)]))
        w = max(t.shape[1] for t in tiles)
        sheet = np.vstack([np.pad(t, ((0, 8), (0, w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles])
        cv2.imwrite(str(out_dir / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cost = sum(_run_cost(conn, r) for r in runs)
    md = [f"# Look: {name}", "", f"{prof.n} reference image(s). Runs {runs[0]}-{runs[-1]}, ${cost:.3f}, "
          f"reviewer engine {engine}.", "", "## What the references share (measured)", "",
          *[f"- {h}" for h in prof.hints], "",
          f"Tone (L* 1/5/25/50/75/95/99 %): {prof.stats.lum}. Colour by band: "
          + "; ".join(f"{k} a {v.a:+.1f} b {v.b:+.1f}" for k, v in prof.stats.bands.items()) + ".", ""]
    if gd:
        md += ["## How to shoot it", "", f"**{gd.summary}**", "", f"- **Lens:** {gd.lens}",
               f"- **Camera:** {gd.camera_position}; {gd.distance}", f"- **Light:** {'; '.join(gd.light)}",
               f"- **Exposure and motion:** {gd.exposure_and_motion}", f"- **Colour:** {gd.colour}",
               f"- **Composition:** {'; '.join(gd.composition)}",
               f"- **Directing:** {'; '.join(gd.subject_direction)}",
               f"- **Places like this:** {'; '.join(gd.places_like_this)}", f"- **Avoid:** {'; '.join(gd.avoid)}",
               f"- **The edit finishes:** {gd.edit_finish}", "", "On-set checklist:", "",
               *[f"- [ ] {c}" for c in gd.checklist], ""]
    md += ["## Your photos", ""]
    for photo, review, params, dest, err, notes in rows:
        md.append(f"### {Path(photo).name}")
        if err:
            md += [f"Edit failed: {err}", ""]
            continue
        if review:
            k = review["knobs"]
            md.append(f"Fit {k['fit']}/5. {k['reason']}")
            if k["reshoot"]:
                md += ["", "Next time you shoot it:", *[f"- {x}" for x in k["reshoot"]]]
        knobs = {k: v for k, v in params.items() if v not in (0, 0.0, "none") and k not in ("ink", "paper")}
        md += ["", f"Edit: `{json.dumps(knobs)}` -> `{dest.name}`"]
        if notes and notes["mask"] != "none":
            sl = notes.get("subject_L")
            md.append(f"Mask: {notes['mask']}, policy {notes['mask_policy']}, used step **{notes['step']}**"
                      + (f" (subject brightness L* {sl[0]} -> {sl[1]})" if sl else "") + ".")
        md.append("")
    write_atomic(out_dir / "report.md", "\n".join(md))
    summary = (f"Look '{name}' from {prof.n} reference(s): {'; '.join(prof.hints)}.\n"
               f"{len(pairs)} of {len(photos)} photo(s) edited, ${cost:.3f}. Report: {out_dir / 'report.md'}")
    return summary, out_dir


def _db(x) -> str:
    return "-" if x is None else f"{x:.0f}"


def audio(conn, settings, deps, src: Path, *, preset: str = "voice", check_only: bool = False,
          dest: Path | None = None, isolate: bool = False, music_db: float | None = None, progress=None) -> str:
    """Measure a clip's audio (loudness, peaks, noise) and, unless check_only, clean it with audio_clean."""
    import json

    from wonhee.media import path_for

    sha = repo.save_file(conn, settings.media_dir, Path(src).expanduser(), "video", "video/mp4")
    rid, text = node_run(conn, settings, deps, "audio_check", {"video": sha}, {}, progress=progress)
    o = _step_out(conn, rid, "audio_check")
    if o is None:
        return text
    chk = json.loads(path_for(settings.media_dir, o["report"]).read_text(encoding="utf-8"))
    s = chk["stats"]
    lines = [f"{Path(src).name}: {s['lufs']:.1f} LUFS, true peak {s['true_peak']:.1f} dBTP, range {s['lra']:.1f} LU, "
             f"speech/noise {_db(s['snr'])} dB",
             *[f"- {v}" for v in chk["verdicts"]]]
    if check_only:
        return "\n".join(lines)
    params = {"preset": preset, "isolate": isolate, **({"music_db": music_db} if music_db is not None else {})}
    rid, text = node_run(conn, settings, deps, "audio_clean", {"video": sha}, params, progress=progress)
    o = _step_out(conn, rid, "audio_clean")
    if o is None:
        return "\n".join(lines + [text])
    rep = json.loads(path_for(settings.media_dir, o["report"]).read_text(encoding="utf-8"))
    a = rep["after"]
    dest = dest or Path(src).with_name(Path(src).stem + ".clean.mp4")
    write_atomic_bytes(dest, path_for(settings.media_dir, o["video"]).read_bytes())
    lines.append(f"cleaned ({rep['denoiser']}): {a['lufs']:.1f} LUFS, true peak {a['true_peak']:.1f} dBTP, "
                 f"range {a['lra']:.1f} LU, speech/noise {_db(a['snr'])} dB -> {dest}")
    return "\n".join(lines)


def _audio_ext(mime: str | None) -> str:
    m = (mime or "").lower()
    for key, ext in (("mpeg", ".mp3"), ("mp4", ".m4a"), ("aac", ".aac"), ("ogg", ".ogg"), ("opus", ".opus"),
                     ("flac", ".flac"), ("wav", ".wav")):
        if key in m:
            return ext
    return ".m4a"


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
        if row["kind"] in ("video", "image", "audio"):
            files = [(ref, {"video": ".mp4", "image": ".jpg", "audio": _audio_ext(row["mime"])}[row["kind"]])]
        else:
            try:
                images = Images.model_validate_json(path_for(settings.media_dir, ref).read_bytes())
            except ValueError:
                raise ValueError(f"{ref}: not a video, image or images manifest") from None
            files = [(s.sha, ".jpg") for s in images.items]
    else:
        raise ValueError(f"{ref}: want a 64-hex sha or content:<id>")
    if dest.exists() and not dest.is_dir():
        raise ValueError(f"{dest} is a file; export wants a folder")
    for sha, _ in files:
        if not path_for(settings.media_dir, sha).is_file():
            raise ValueError(f"{ref}: stored file {sha[:12]} is missing from the media store")
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


# --- filters: the pack, AI-made filters, saved for later -------------------------------------------------------

def _filter_sheet(pairs: list[tuple], dest: Path, h: int = 360) -> None:
    import cv2
    import numpy as np
    tiles = [np.hstack([cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h)) for x in pair]) for pair in pairs]
    w = max(t.shape[1] for t in tiles)
    sheet = np.vstack([np.pad(t, ((0, 8), (0, w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles])
    cv2.imwrite(str(dest), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def _render_filter(conn, settings, deps, shas: list[str], spec, progress=None) -> list[str | None]:
    out = []
    for sha in shas:
        rid, _ = node_run(conn, settings, deps, "filter_image", {"image": sha}, spec.model_dump(), progress=progress)
        o = _step_out(conn, rid, "filter_image")
        out.append(o["image"] if o else None)
    return out


def filter_make(conn, settings, deps, description: str, photos: list[Path], *, name: str, rounds: int = 3,
                engine: str = "local", start: str | None = None, progress=None) -> tuple[str, Path | None]:
    """Make (or, with `start`, tweak) a filter from a description of a vibe, tune it on your photos and save it.

    Each round renders the filter on the photos, measures them (clipping, crushed shadows, skin drift), lets the
    guard fix what the numbers say is broken, then a vision model scores before|after against the description and
    moves a few knobs (clamped, step-limited). The best-scoring round is saved as <home>/filters/<name>.json.
    Returns (summary, the folder with the round sheets and final renders)."""
    import json

    from wonhee import filterstore
    from wonhee.edit import filters as F
    from wonhee.edit import style
    from wonhee.media import path_for
    from wonhee.nodes.filter import describe

    if not photos:
        return "give at least one photo to tune the filter on (2-3 different ones is best)", None
    key = filterstore.slug(name)
    if key in F.PACK:
        return f"'{key}' is a pack filter; pick another name (e.g. my-{key})", None
    d = settings.media_dir
    shas = [repo.save_file(conn, d, Path(p).expanduser(), "image", "image/jpeg") for p in photos[:3]]
    srcs = [style.load(path_for(d, s)) for s in shas]
    base, change = None, description
    if start:
        base, old = filterstore.load(settings.home, start)
        if old.get("description"):  # the goal stays the original vibe plus the change, not the change alone
            description = f"{old['description']}; then: {change}"[:300]
    params = {"description": change, "engine": engine, **({"start": base.model_dump_json()} if base else {})}
    rid, text = node_run(conn, settings, deps, "filter_draft", {}, params, progress=progress)
    runs = [rid]
    o = _step_out(conn, rid, "filter_draft")
    if o is None:
        return text, None
    draft = json.loads(path_for(d, o["filter"]).read_text())
    spec = F.Filter(**draft["filter"])
    out_dir = filterstore.folder(settings.home) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    for r in range(1, max(1, rounds) + 1):
        outs = _render_filter(conn, settings, deps, shas, spec, progress)
        if any(x is None for x in outs):
            return f"rendering the filter failed in round {r}", out_dir
        imgs = [style.load(path_for(d, x)) for x in outs]
        facts = [F.checks(a, b) for a, b in zip(srcs, imgs)]
        fixed, guarded = F.guard(spec, facts)
        if guarded:
            spec = fixed
            outs = _render_filter(conn, settings, deps, shas, spec, progress)
            imgs = [style.load(path_for(d, x)) for x in outs]
            facts = [F.checks(a, b) for a, b in zip(srcs, imgs)]
        sheet = out_dir / f"round-{r}.jpg"
        _filter_sheet(list(zip(srcs, imgs)), sheet)
        sheet_sha = repo.save_file(conn, d, sheet, "image", "image/jpeg")
        fact_text = "; ".join(f"photo {i + 1}: {F.verdicts(f)}" for i, f in enumerate(facts))
        rid, text = node_run(conn, settings, deps, "filter_review", {"sheet": sheet_sha},
                             {"description": description, "filter": spec.model_dump_json(), "facts": fact_text,
                              "engine": engine}, progress=progress)
        runs.append(rid)
        o = _step_out(conn, rid, "filter_review")
        rev = json.loads(path_for(d, o["review"]).read_text()) if o else None
        score = rev["score"] if rev else 0
        history.append({"round": r, "filter": spec.model_dump(), "score": score, "guard": guarded,
                        "verdict": rev["verdict"] if rev else text, "moves": rev["moves"] if rev else [],
                        "checks": facts})
        if best is None or score >= best[0]:
            best = (score, spec, outs, r)
        if rev is None or rev["done"] or r == rounds:
            break
        spec = F.Filter(**rev["filter"])
    score, spec, outs, best_round = best
    for i, (photo, x) in enumerate(zip(photos, outs), 1):
        (out_dir / f"{i:02d}-{Path(photo).stem}.jpg").write_bytes(path_for(d, x).read_bytes())
    cost = sum(_run_cost(conn, r) for r in runs)
    rec = {"description": description, "base": draft.get("base"), "draft_moves": draft.get("moves"),
           "from": start, "engine": engine,
           "filter": spec.model_dump(), "score": score, "best_round": best_round, "rounds": history,
           "samples": [str(p) for p in photos[:3]], "cost_usd": round(cost, 4)}
    saved = filterstore.save(settings.home, key, rec)
    lines = [f"Filter '{key}' saved: {saved}", f"  {draft.get('notes', '')}".rstrip(),
             f"  base: {draft.get('base')} + {', '.join(draft.get('moves') or []) or 'no moves'}",
             f"  best round {best_round} of {len(history)}, score {score}/5, ${cost:.3f}",
             f"  knobs: {describe(spec)}"]
    for h in history:
        lines.append(f"  round {h['round']}: {h['score']}/5 {h['verdict']}"
                     + (f" -> {', '.join(h['moves'])}" if h["moves"] else "")
                     + (f" [guard: {'; '.join(h['guard'])}]" if h["guard"] else ""))
    lines.append(f"  sheets and renders: {out_dir}")
    lines.append(f"  use it: wonhee filter apply {key} <photos...>   tweak: wonhee filter make \"<change>\" "
                 f"--from {key} --name {key} --on <photos...>")
    return "\n".join(lines), out_dir


def filter_apply(conn, settings, deps, name: str, photos: list[Path], *, intensity: float = 1.0,
                 dest: Path | None = None, progress=None) -> tuple[str, Path | None]:
    """Apply a pack or saved filter to photos; writes <home>/filtered/<date>-<name>/ (or `dest`)."""
    from datetime import date

    from wonhee import filterstore
    from wonhee.media import path_for

    spec, rec = filterstore.load(settings.home, name)
    key = rec["name"]
    out_dir = dest or settings.home / "filtered" / f"{date.today().isoformat()}-{key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    done, failed = 0, []
    for i, photo in enumerate(photos, 1):
        sha = repo.save_file(conn, settings.media_dir, Path(photo).expanduser(), "image", "image/jpeg")
        rid, text = node_run(conn, settings, deps, "filter_image", {"image": sha},
                             {**spec.model_dump(), "intensity": intensity}, progress=progress)
        o = _step_out(conn, rid, "filter_image")
        if o is None:
            failed.append(f"{photo}: {text}")
            continue
        (out_dir / f"{i:02d}-{Path(photo).stem}.jpg").write_bytes(path_for(settings.media_dir, o["image"]).read_bytes())
        done += 1
    return (f"{key} ({rec['source']}) at {intensity:g}: {done} of {len(photos)} photo(s) -> {out_dir}"
            + "".join(f"\n  failed {f}" for f in failed)), out_dir


def filter_list(settings) -> str:
    from wonhee import filterstore
    from wonhee.edit.filters import PACK
    lines = ["Pack:"] + [f"  {k:13} {d}" for k, (d, _) in PACK.items()]
    mine = filterstore.saved(settings.home)
    lines.append("Yours:" if mine else "Yours: none yet (wonhee filter make \"<a vibe>\" --name <name> --on <photos>)")
    lines += [f"  {r['name']:13} {r.get('description', '')}  [{r.get('score', '?')}/5, {r.get('updated', '')[:10]}]"
              for r in mine]
    return "\n".join(lines)


def filter_show(settings, name: str, cube: Path | None = None) -> str:
    """One filter's knobs and history; with `cube`, also write its colour part as a .cube LUT (phone / video
    editors: CapCut, VN, Lightroom, Resolve). Glow, grain and other spatial effects are not in a LUT."""
    from wonhee import filterstore
    from wonhee.edit import filters as F
    from wonhee.nodes.filter import describe
    spec, rec = filterstore.load(settings.home, name)
    lines = [f"{rec['name']} ({rec['source']}): {rec.get('description', '')}", f"  knobs: {describe(spec)}"]
    for h in rec.get("rounds", []):
        lines.append(f"  round {h['round']}: {h['score']}/5 {h['verdict']}")
    if cube:
        cube = Path(cube).expanduser()
        cube.write_text(F.cube(spec, title=rec["name"]), encoding="utf-8")
        skipped = [k for k in ("clarity", "sharpen", "glow", "bloom", "halation", "light_leak", "aberration",
                               "vignette", "grain") if getattr(spec, k)]
        lines.append(f"  LUT written: {cube}" + (f" (not in a LUT: {', '.join(skipped)})" if skipped else ""))
    return "\n".join(lines)
