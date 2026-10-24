"""MCP server `wonhee` (stdio). Not registered in moeka yet. Run: python -m wonhee.mcp_server"""
from __future__ import annotations

import functools
import json
from pathlib import Path

from wonhee import envfile, errlog, llm, service
from wonhee.config import REPO_ROOT, get_settings
from wonhee.db.connect import open_db
from mcp.server.fastmcp import Context
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ModuleNotFoundError:  # mcp 2.x renamed it
    from mcp.server.mcpserver import MCPServer as _Server

mcp = _Server("wonhee")


def _safe_log(label: str) -> str:
    try:
        return str(errlog.log_traceback(get_settings().home, label))
    except Exception:  # noqa: BLE001 - a broken log must not hide the real answer
        return "(log unavailable)"


def _under(p: Path, roots) -> bool:
    return any(p == r or r in p.parents for r in roots)


def _in_path(p: str) -> Path:
    """An input file or folder: resolved, and inside the user's home, the repo or the data home."""
    q = Path(p).expanduser().resolve()
    if not _under(q, {Path.home().resolve(), REPO_ROOT, get_settings().home.resolve()}):
        raise ValueError(f"{p}: input paths must be under {Path.home()}")
    return q


def _out_path(p: str) -> Path:
    """Somewhere to write: resolved, and inside the wonhee data home or the repo's data/ folder."""
    q = Path(p).expanduser().resolve()
    if not _under(q, {get_settings().home.resolve(), (REPO_ROOT / "data").resolve()}):
        raise ValueError(f"{p}: output paths must be under {get_settings().home}")
    return q


def _ref(v):
    """A port value that is a content:<id>, sha or <id>.<port> reference passes; anything path-like is checked."""
    from wonhee.nodes.types import CONTENT, SHA
    if not isinstance(v, str) or CONTENT.match(v) or SHA.match(v):
        return v
    return str(_in_path(v)) if ("/" in v or v.startswith("~") or Path(v).exists()) else v


def _guard(fn):
    """Expected errors become readable text; unexpected ones are logged and pointed to."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except (KeyError, ValueError) as e:
            _safe_log(f"mcp {fn.__name__} (expected)")
            return f"error: {e.args[0] if e.args else e}"
        except llm.LLMError as e:
            log = _safe_log(f"mcp {fn.__name__}")
            return f"model call failed: {errlog.one_line(e)} (details: {log})"
        except Exception:  # noqa: BLE001
            log = _safe_log(f"mcp {fn.__name__}")
            return f"unexpected error — see {log}"
    return wrapper


def _open():
    settings = get_settings()
    return open_db(settings), settings


def _deps(settings):
    from wonhee.pipeline.deps import Deps
    return Deps.live(settings)


@mcp.tool()
@_guard
def scout_ask(text: str, intent_id: int | None = None) -> str:
    """One compile turn of a trend-scout request. Returns questions, or a confirmation to answer with 'go'."""
    conn, settings = _open()
    try:
        r = service.ask(conn, settings, text, intent_id)
        return f"{r.message}\n[intent {r.intent_id}: {r.status}]"
    finally:
        conn.close()


@mcp.tool()
@_guard
def scout_go(intent_id: int) -> str:
    """Run the scout for a confirmed intent; returns the shot list (or what went wrong)."""
    conn, settings = _open()
    try:
        return service.go(conn, settings, _deps(settings), intent_id)[1]
    finally:
        conn.close()


@mcp.tool()
@_guard
def scout_resume(run_id: int, from_stage: str, overrides: dict | None = None) -> str:
    """Re-run a scout from a stage; overrides may include cost_cap_usd."""
    conn, settings = _open()
    try:
        return service.resume(conn, settings, _deps(settings), run_id, from_stage, overrides or {})
    finally:
        conn.close()


@mcp.tool()
@_guard
def library_grab(urls: list[str], relation: str = "external") -> str:
    """Store Instagram reels by URL: metadata, a metric snapshot, the video and its analysis (cuts, transcript,
    keyframe paths you can look at). relation: ours | friend | external."""
    conn, settings = _open()
    try:
        return service.grab(conn, settings, _deps(settings), urls, relation)[1]
    finally:
        conn.close()


@mcp.tool()
@_guard
def scout_show_run(run_id: int) -> str:
    """Per-stage status, cost and flags of a run."""
    conn, settings = _open()
    try:
        return service.show_run(conn, run_id, settings.media_dir)
    finally:
        conn.close()


@mcp.tool()
@_guard
def nodes_list(name: str | None = None) -> str:
    """The building blocks of a chain: each node's name, input and output ports (with types) and what it does.
    With name: that one node's ports and its params JSON schema (what node_run / chain_run accept)."""
    from wonhee.nodes import registry
    if name is not None:
        if name not in registry.all_nodes():
            raise ValueError(f"unknown node '{name}'; call nodes_list() with no name to see them")
        n = registry.get(name)
        return "\n".join([f"{n.name} (v{n.version}): {n.doc}",
                          "inputs:  " + (", ".join(f"{k}:{v}" for k, v in n.inputs.items()) or "-"),
                          "outputs: " + ", ".join(f"{k}:{v}" for k, v in n.outputs.items()),
                          "params (JSON schema):", json.dumps(n.params.model_json_schema(), indent=2)])
    lines = []
    for n in registry.all_nodes().values():
        ins = ", ".join(f"{k}:{v}" for k, v in n.inputs.items()) or "-"
        outs = ", ".join(f"{k}:{v}" for k, v in n.outputs.items())
        lines.append(f"{n.name}  [{ins} -> {outs}]  {n.doc}")
    return "\n".join(lines)


@mcp.tool()
@_guard
def node_run(node: str, inputs: dict, params: dict | None = None, wait: bool = True, no_cache: bool = False) -> str:
    """Run one node. inputs: port -> content:<id> | blob sha | local file path. wait=False returns a run id at once
    (check with chain_show). no_cache=True re-runs instead of reusing a cached result."""
    inputs = {k: _ref(v) for k, v in inputs.items()}
    if node == 'drive_download' and params and 'dest' in params:
        params = {**params, 'dest': str(_out_path(params['dest']))}
    conn, settings = _open()
    try:
        return service.node_run(conn, settings, _deps(settings), node, inputs, params or {}, wait,
                               no_cache=no_cache)[1]
    finally:
        conn.close()


@mcp.tool()
@_guard
def chain_run(chain: list | dict, wait: bool = True, no_cache: bool = False) -> str:
    """Run a chain: a list of {id, node, in: {port: "<id>.<port>" | ref | path}, params, cache}, or
    {nodes: [...], cost_cap_usd, no_cache}. no_cache=True skips the cache for every node. The whole chain is validated first and nothing runs if it is invalid.
    wait=False returns a run id at once (check with chain_show)."""
    for n in chain if isinstance(chain, list) else (chain.get("nodes") or []) if isinstance(chain, dict) else []:
        if isinstance(n, dict) and n.get('node') == 'drive_download' and 'dest' in (n.get('params') or {}):
            n['params'] = {**n['params'], 'dest': str(_out_path(n['params']['dest']))}
        if isinstance(n, dict) and isinstance(n.get("in"), dict):
            n["in"] = {k: _ref(v) for k, v in n["in"].items()}
    conn, settings = _open()
    try:
        return service.chain_run(conn, settings, _deps(settings), chain, wait=wait, no_cache=no_cache)[1]
    finally:
        conn.close()


@mcp.tool()
@_guard
def larp(task: str, image: str | None = None, video: str | None = None, pick: int | None = None,
         engine: str = "remote", register: list[str] | None = None) -> str:
    """Frame any mundane task as a professional moment (LinkedIn announcement, founder update, earnings call, ...):
    on-screen lines, captions, hashtags, the fitting text preset / look, and the shot to capture. With an image or
    video (path or sha), also render the chosen line onto it. engine="local" uses the on-box model ($0). register: ask for specific registers, e.g.
    ["linkedin_announcement", "earnings_call"]."""
    image, video = (_ref(image) if image else image), (_ref(video) if video else video)
    conn, settings = _open()
    try:
        return service.larp(conn, settings, _deps(settings), task, image=image, video=video, pick=pick, engine=engine,
                             registers=register or None)
    finally:
        conn.close()


@mcp.tool()
@_guard
def slop_next(n: int = 2, engine: str = "local") -> str:
    """GitHub slop: pick N unseen public repos, have a model write the voice lines (engine local|remote), render a
    credited 4:5 card set for each into data/slop/<date>/. Returns the output folders."""
    conn, settings = _open()
    try:
        return service.slop_next(conn, settings, _deps(settings), min(max(n, 1), 10), engine)[1]
    finally:
        conn.close()


@mcp.tool()
@_guard
def slop_pick(repo: str | None = None) -> str:
    """GitHub slop, agent path step 1: pick one unseen public repo (or repo="owner/name") and return its dossier JSON.
    Write a script from it (hook, what, why, roast, caption, hashtags; no invented facts), then call slop_render."""
    conn, settings = _open()
    try:
        return service.slop_pick(conn, settings, _deps(settings), repo)[1]
    finally:
        conn.close()


@mcp.tool()
@_guard
def slop_render(dossier: str, script: dict) -> str:
    """GitHub slop, agent path step 2: render the card set. dossier: "owner/name" from slop_pick (or a dossier file
    path); script: {hook <=9 words, what <=24, why <=24, roast <=16, caption <=300 chars, hashtags <=6 without #}."""
    import re
    ref = dossier if re.fullmatch(r"[\w.-]+/[\w.-]+", dossier) and not Path(dossier).exists() else _ref(dossier)
    conn, settings = _open()
    try:
        return service.slop_render(conn, settings, _deps(settings), ref, json.dumps(script))[1]
    finally:
        conn.close()


@mcp.tool()
@_guard
def inspire(content_ids: list[int] | None = None, since: str | None = None, relation: str | None = None) -> str:
    """Tear down stored inspiration posts (cached per post; ~$0.001 each, translation on-box) and write a report:
    formats, text/footage coupling, what to build, best to replicate, a shoot list and a line bank."""
    conn, settings = _open()
    try:
        return service.inspire(conn, settings, _deps(settings), content_ids, since, relation)[0]
    finally:
        conn.close()


@mcp.tool()
@_guard
def ideate(n: int = 20, focus: str | None = None, engine: str = "remote", content_ids: list[int] | None = None,
           since: str | None = None) -> str:
    """Ideas for posts to make and the b-roll plates to film, from saved inspiration posts (teardowns, cached) and
    the creator's own brief (data/brief/*.md). Each idea cites its sources and is tagged inspiration / own / blend /
    new; plates are ranked by how many ideas they serve. engine="local" uses the on-box model ($0)."""
    conn, settings = _open()
    try:
        return service.ideate(conn, settings, _deps(settings), n=n, focus=focus, engine=engine,
                              content_ids=content_ids, since=since)[0]
    finally:
        conn.close()


@mcp.tool()
@_guard
def vibe(like: list[str], photos: list[str] | None = None, name: str = "look", engine: str = "local",
         guide: bool = True, guide_engine: str = "remote", set: dict | None = None) -> str:
    """Learn a photographic look from references (folders, image files or content:<id> posts; reels give their
    keyframes), write a shooting guide for it (remote vision, ~$0.003, guide=False skips), and edit each photo
    toward it: a vision model picks clamped knobs (engine local = $0), the edit is deterministic (tone and
    per-band colour match, subject mask, fisheye / blur / bloom / vignette / grain). No image generation.
    guide_engine: remote | local, who writes the guide. set: force edit knobs, e.g. {"fisheye": 0.6}."""
    like = [_ref(x) for x in like]
    photos = [_in_path(p) for p in photos or []]
    conn, settings = _open()
    try:
        return service.vibe(conn, settings, _deps(settings), like, photos, name=name,
                            engine=engine, guide=guide, guide_engine=guide_engine, overrides=set or {})[0]
    finally:
        conn.close()


@mcp.tool()
@_guard
def filter_make(description: str, photos: list[str], name: str, start: str | None = None, rounds: int = 3,
                engine: str = "local") -> str:
    """Make a photo filter from a description of a vibe ("late-summer film, a bit faded"), tune it on 1-3 photos
    (render, measure, a vision model scores before|after and moves a few clamped knobs) and save it by name for
    later. start = an existing pack or saved filter to tweak instead. engine local = $0. No image generation."""
    photos = [_in_path(p) for p in photos]
    conn, settings = _open()
    try:
        return service.filter_make(conn, settings, _deps(settings), description, photos,
                                   name=name, rounds=rounds, engine=engine, start=start)[0]
    finally:
        conn.close()


@mcp.tool()
@_guard
def viz(kind: str, path: str | None = None, themes: list[str] | None = None, params: dict | None = None,
        then: str | None = None, name: str | None = None) -> str:
    """Picture from data. kind: depgraph | diskviz | history | wavegen. params: rev, include, exclude, focus, hops,
    chart, layout, title, size, ... (see nodes_list(name="viz_<kind>")). then: look=<preset> | filter=<name> |
    cutout=<kind>[:<background>]. themes default ["paper"]; ["paper", "dark"] for a light/dark pair."""
    folder = str(_in_path(path)) if path else None
    conn, settings = _open()
    try:
        return service.viz(conn, settings, _deps(settings), kind, folder, themes=tuple(themes or ["paper"]),
                           params=params, then=then, name=name)[0]
    finally:
        conn.close()


@mcp.tool()
@_guard
def reel(video: str, style: str, title: str | None = None, cta: str | None = None, items: list[str] | None = None,
         insets: list[str] | None = None, set: dict | None = None) -> str:
    """Make a reel from a talking clip (path) in a style (see reel_styles): speech_subs then compose_video with the
    style's template, one encode, pauses trimmed in the talking-head styles. title / cta / items (list entries,
    scrapbook chapters or lecture theses) / insets (image paths) fill the style's slots; set overrides any slot or
    tighten, encoder, quality, look, engine, name. The lecture style writes its own slots with the larp lecture register
    (local model). Writes <data home>/reels/<date>/<name>/ with the mp4 and report.md; returns the path."""
    conn, settings = _open()
    try:
        return service.reel(conn, settings, _deps(settings), str(_in_path(video)), style, title=title, cta=cta,
                            items=items, insets=[str(_in_path(p)) for p in insets or []] or None, overrides=set)
    finally:
        conn.close()


@mcp.tool()
@_guard
def reel_styles() -> str:
    """The reel styles `reel` can make, one line each."""
    return service.reel_styles()


@mcp.tool()
@_guard
def publish_muksite(image: str, slug: str, name: str, caption: str, alt: str, dark: str | None = None,
                    size: str = "hero", fit: str = "partial", overwrite: bool = False) -> str:
    """Upload a finished still (ref or file, plus an optional dark twin) to muksite's PUBLIC image storage under
    projects/<slug>/<name>; returns the YAML photos: entry. Only call this when the user asked to publish."""
    conn, settings = _open()
    try:
        return service.publish_muksite(conn, settings, _deps(settings), _ref(image), slug=slug, name=name,
                                       caption=caption, alt=alt, dark=_ref(dark) if dark else None, size=size,
                                       fit=fit, overwrite=overwrite)[0]
    finally:
        conn.close()


@mcp.tool()
@_guard
def filter_apply(name: str, photos: list[str] | None = None, intensity: float | None = None) -> str:
    """Apply a pack or saved filter to photos (intensity 0..2; default is your usual for that filter, else 1).
    With no photos, lists the pack and saved filters."""
    photos = [_in_path(p) for p in photos or []]
    conn, settings = _open()
    try:
        if not photos:
            return service.filter_list(settings)
        return service.filter_apply(conn, settings, _deps(settings), name, photos,
                                    intensity=intensity)[0]
    finally:
        conn.close()


@mcp.tool()
@_guard
def filter_show(name: str, cube: bool = False) -> str:
    """One filter's knobs and history (a pack or saved filter). cube=True also writes its colour part as a .cube LUT
    (CapCut, VN, Lightroom, Resolve) under the wonhee data home and returns its path."""
    conn, settings = _open()
    try:
        out = None
        if cube:
            out = _out_path(str(settings.home / "filters" / "cube" / f"{name}.cube"))
        return service.filter_show(settings, name, cube=out)
    finally:
        conn.close()


@mcp.tool()
@_guard
def filter_calibrate(photos: list[str], n: int = 40, engine: str = "local") -> str:
    """Measure which filter knobs the reviewer model can see (move one knob, show before|after, ask which moved);
    `filter_make` then hides the knobs it cannot see from the reviewer. Writes <data home>/filters/_calibration.json.
    engine local = $0."""
    paths = [_in_path(p) for p in photos]
    conn, settings = _open()
    try:
        return service.filter_calibrate(conn, settings, _deps(settings), paths, n=n, engine=engine)[0]
    finally:
        conn.close()


@mcp.tool()
@_guard
def audio(file: str, preset: str = "voice", check: bool = False, isolate: bool = False,
          music_db: float | None = None, out: str | None = None) -> str:
    """Measure a clip's audio (LUFS, true peak, range, speech-to-noise) and, unless check, clean it on the box:
    rumble cut, RNNoise denoise, compression, two-pass loudness to -14 LUFS / -1.5 dBTP. Writes <file>.clean.mp4,
    or out (a file path under the wonhee data home). preset: voice | mixed | music. isolate: GPU, split the voice
    from music / street noise first; music_db (with isolate): keep the music this many dB down, e.g. -15."""
    src = _in_path(file)
    dest = _out_path(out) if out else None
    conn, settings = _open()
    try:
        return service.audio(conn, settings, _deps(settings), src, preset=preset, check_only=check, isolate=isolate,
                             music_db=music_db, dest=dest)
    finally:
        conn.close()


@mcp.tool()
@_guard
def shots(query: str = "", video: str = "", k: int = 8) -> str:
    """Shot search over all indexed footage. With video: index that file first (cut into shots, SigLIP2 vector and
    tags per keyframe; needs the GPU group). With query: the k shots best matching a description, e.g.
    "bus at sunset", each with its keyframe path."""
    conn, settings = _open()
    try:
        parts = []
        if video:
            parts.append(service.shots_index(conn, settings, _deps(settings), _in_path(video)))
        if query:
            parts.append(service.shots_search(conn, settings, query, k=k))
        return "\n\n".join(parts) or "error: give a query, a video, or both"
    finally:
        conn.close()


@mcp.tool()
@_guard
def workers() -> str:
    """Warm GPU workers and the job queue. A detached node_run whose chain needs the GPU goes to a live worker
    (models already loaded, no per-call load) and otherwise to a cold background process. Start one with
    `wonhee worker serve --lane cuda --preload whisper,mask`."""
    from wonhee import worker
    conn, _ = _open()
    try:
        return worker.status(conn)
    finally:
        conn.close()


@mcp.tool()
@_guard
def drive_download(url: str, dest: str) -> dict:
    """Download a public Google Drive file or folder into dest, preserving subfolders. Never overwrites files."""
    destination = _out_path(dest)
    conn, settings = _open()
    try:
        return service.drive_download(conn, settings, _deps(settings), url, destination)
    finally:
        conn.close()


@mcp.tool()
@_guard
def export(ref: str, dest: str) -> str:
    """Copy a stored result into a local folder for posting: a video or image (one file), an images manifest or
    content:<id> photo post (01.jpg, 02.jpg, ...). Returns the written paths."""
    out = _out_path(dest)
    conn, settings = _open()
    try:
        return "\n".join(str(p) for p in service.export(conn, settings, ref, out))
    finally:
        conn.close()


@mcp.tool()
@_guard
def chain_show(run_id: int) -> str:
    """Status, steps and final outputs (full file paths) of a chain run."""
    conn, settings = _open()
    try:
        return service.chain_show(conn, settings, run_id)
    finally:
        conn.close()



class _CreativeRoute(BaseModel):
    model_config = ConfigDict(extra='forbid')
    channel: Literal['telegram']
    chat_id: str = Field(min_length=1, max_length=64)
    owner_id: str = Field(min_length=1, max_length=128)
    message_id: int | None = None
    message_thread_id: int | None = None
    route_key: str = Field(min_length=1, max_length=256)


class _CreativeOrigin(BaseModel):
    version: Literal[1]
    project_id: int = Field(gt=0)
    request_id: str = Field(min_length=1, max_length=256)
    route: _CreativeRoute


def _creative_arguments(action: str, kwargs: dict, ctx: Context | None) -> dict:
    """Bind provenance and destination outside the model-controlled argument schema."""
    if ctx is None or ctx.request_context.meta is None:
        return kwargs
    meta = ctx.request_context.meta.model_dump(mode='json')
    if meta.get('wonhee') is None:
        return kwargs
    origin = _CreativeOrigin.model_validate(meta['wonhee'])
    route = origin.route.model_dump(mode='json')
    kwargs = dict(kwargs)
    if action in {'memory_remember', 'memory_correct', 'memory_forget', 'idea_save', 'idea_update'}:
        source = dict(kwargs.get('source') or {})
        source.update({**route, 'request_id': origin.request_id, 'origin_project_id': origin.project_id})
        kwargs['source'] = source
    if action in {'memory_remember', 'memory_query', 'idea_save', 'make_submit', 'outbox_add', 'inbox_add'}:
        if kwargs.get('scope') == 'creator':
            kwargs['project_id'] = None
        elif kwargs.get('project_id') is None:
            kwargs['project_id'] = origin.project_id
    if action in {'make_submit', 'outbox_add'}:
        kwargs['route'] = route
    if action == 'make_submit':
        kwargs['request_key'] = origin.request_id
    if action == 'project_select':
        kwargs['route_key'] = route['route_key']
    return kwargs


def _register_creative_tools() -> None:
    """Project services keep their real signatures while connections stay behind the MCP boundary."""
    import inspect
    from typing import get_type_hints
    from wonhee.creative import ACTIONS
    descriptions = {
        'memory_query': 'Search wonhee memory with ordinary words and optional project, kind, status, since/until filters. '
                        'scope auto includes creator preferences and the project; all explicitly searches all wonhee projects. '
                        'Use operation count/list/timeline for structured questions. Sources accompany every record.',
        'memory_remember': 'Persist a stated user preference, decision or feedback before acknowledging it. '
                           'In a creative conversation the active project is the default; scope=creator explicitly saves wonhee-wide preferences. '
                           'Save model suggestions as proposed, never confirmed.',
        'make_submit': 'Create one durable background reel from a stored video asset. request_key must identify the '
                       'source user request; reuse it after timeouts. Use make_show for status; never poll in a loop. '
                       'Supply the trusted route from the peer context for delivery.',
        'inbox_add': 'Save a local file or public Google Drive file/folder into the project. Folders return assets; dest is optional. '
                     'Use the transport source_key on retries. Does not render or start GPU analysis.',
    }
    # Delivery acknowledgements are host-only: agents can inspect/retry but cannot claim a file was sent.
    excluded = {'delivery_show', 'media_info'}
    for action, fn in ACTIONS.items():
        if action in excluded:
            continue
        name = 'creative_' + action if action.startswith('project_') or action == 'health' else action
        target = getattr(service, name)
        def build(target=target, action=action):
            @functools.wraps(target)
            def call(ctx: Context | None = None, **kwargs):
                kwargs = _creative_arguments(action, kwargs, ctx)
                if action == 'inbox_add' and not str(kwargs.get('source','')).startswith(('http:', 'https:')):
                    kwargs['source'] = str(_in_path(kwargs['source']))
                if action == 'inbox_add':
                    from wonhee.drive import folder_identity
                    folder = folder_identity(str(kwargs.get('source', '')))
                    if kwargs.get('dest') is not None or folder:
                        default = get_settings().home / 'drive' / folder[0] if folder else None
                        kwargs['dest'] = str(_out_path(kwargs.get('dest') or str(default)))
                conn, settings = _open()
                try:
                    return target(conn, settings, **kwargs)
                finally:
                    conn.close()
            signature = inspect.signature(target)
            hints = get_type_hints(target)
            parameters = [p.replace(annotation=hints.get(k, p.annotation)) for k, p in signature.parameters.items()
                          if k not in ('conn', 'settings')]
            parameters.append(inspect.Parameter('ctx', inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Context))
            call.__annotations__ = {k:v for k,v in hints.items() if k not in ('conn', 'settings')}
            call.__annotations__['ctx'] = Context
            call.__signature__ = signature.replace(parameters=parameters, return_annotation=hints.get('return', dict))
            call.__doc__ = descriptions.get(action, target.__doc__ or f'Wonhee creative action: {action}. Returns structured records.')
            return call
        mcp.tool(name=name)(_guard(build()))


@mcp.tool()
@_guard
def delivery_show(delivery_id: int | None = None) -> dict:
    """Read delivery status. Transport acknowledgements can only be written by the trusted host adapter."""
    conn, settings = _open()
    try:
        return service.delivery_show(conn, settings, delivery_id=delivery_id)
    finally:
        conn.close()


@mcp.tool()
@_guard
def media_info(asset_id: int, analyze: bool = False):
    """Probe a stored asset and read saved transcripts. analyze=true explicitly makes a CPU contact sheet
    and returns its real image for visual inspection. Never infer video content from filenames alone."""
    import base64
    import json
    from mcp.types import CallToolResult, ImageContent, TextContent
    conn, settings = _open()
    try:
        result = service.media_info(conn, settings, asset_id=asset_id, analyze=analyze)
        content = [TextContent(type='text', text=json.dumps(result, ensure_ascii=False))]
        if result.get('contact_sheet'):
            data = Path(result['contact_sheet']['path']).read_bytes()
            content.append(ImageContent(type='image', data=base64.b64encode(data).decode(), mimeType='image/jpeg'))
        return CallToolResult(content=content, structuredContent=result)
    finally:
        conn.close()


_register_creative_tools()

def main() -> None:
    envfile.load_entry_env()
    mcp.run()


if __name__ == "__main__":
    main()
