"""MCP server `wonhee` (stdio). Not registered in moeka yet. Run: python -m wonhee.mcp_server"""
from __future__ import annotations

import functools
from pathlib import Path

from wonhee import envfile, errlog, llm, service
from wonhee.config import REPO_ROOT, get_settings
from wonhee.db.connect import open_db

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
def nodes_list() -> str:
    """The building blocks of a chain: each node's name, input and output ports (with types) and what it does."""
    from wonhee.nodes import registry
    lines = []
    for n in registry.all_nodes().values():
        ins = ", ".join(f"{k}:{v}" for k, v in n.inputs.items()) or "-"
        outs = ", ".join(f"{k}:{v}" for k, v in n.outputs.items())
        lines.append(f"{n.name}  [{ins} -> {outs}]  {n.doc}")
    return "\n".join(lines)


@mcp.tool()
@_guard
def node_run(node: str, inputs: dict, params: dict | None = None, wait: bool = True) -> str:
    """Run one node. inputs: port -> content:<id> | blob sha | local file path. wait=False returns a run id at once
    (check with chain_show)."""
    inputs = {k: _ref(v) for k, v in inputs.items()}
    conn, settings = _open()
    try:
        return service.node_run(conn, settings, _deps(settings), node, inputs, params or {}, wait)[1]
    finally:
        conn.close()


@mcp.tool()
@_guard
def chain_run(chain: list | dict, wait: bool = True) -> str:
    """Run a chain: a list of {id, node, in: {port: "<id>.<port>" | ref | path}, params, cache}, or
    {nodes: [...], cost_cap_usd, no_cache}. The whole chain is validated first and nothing runs if it is invalid.
    wait=False returns a run id at once (check with chain_show)."""
    for n in chain if isinstance(chain, list) else (chain.get("nodes") or []) if isinstance(chain, dict) else []:
        if isinstance(n, dict) and isinstance(n.get("in"), dict):
            n["in"] = {k: _ref(v) for k, v in n["in"].items()}
    conn, settings = _open()
    try:
        return service.chain_run(conn, settings, _deps(settings), chain, wait=wait)[1]
    finally:
        conn.close()


@mcp.tool()
@_guard
def larp(task: str, image: str | None = None, video: str | None = None, pick: int | None = None,
         engine: str = "remote") -> str:
    """Frame any mundane task as a professional moment (LinkedIn announcement, founder update, earnings call, ...):
    on-screen lines, captions, hashtags, the fitting text preset / look, and the shot to capture. With an image or
    video (path or sha), also render the chosen line onto it. engine="local" uses the on-box model ($0)."""
    image, video = (_ref(image) if image else image), (_ref(video) if video else video)
    conn, settings = _open()
    try:
        return service.larp(conn, settings, _deps(settings), task, image=image, video=video, pick=pick, engine=engine)
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
         guide: bool = True) -> str:
    """Learn a photographic look from references (folders, image files or content:<id> posts; reels give their
    keyframes), write a shooting guide for it (remote vision, ~$0.003, guide=False skips), and edit each photo
    toward it: a vision model picks clamped knobs (engine local = $0), the edit is deterministic (tone and
    per-band colour match, subject mask, fisheye / blur / bloom / vignette / grain). No image generation."""
    like = [_ref(x) for x in like]
    photos = [_in_path(p) for p in photos or []]
    conn, settings = _open()
    try:
        return service.vibe(conn, settings, _deps(settings), like, photos, name=name,
                            engine=engine, guide=guide)[0]
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
    chart, layout, title, size, ... (see node_show viz_<kind>). then: look=<preset> | filter=<name> |
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
def audio(file: str, preset: str = "voice", check: bool = False) -> str:
    """Measure a clip's audio (LUFS, true peak, range, speech-to-noise) and, unless check, clean it on the box:
    rumble cut, RNNoise denoise, compression, two-pass loudness to -14 LUFS / -1.5 dBTP. Writes <file>.clean.mp4.
    preset: voice | mixed | music."""
    src = _in_path(file)
    conn, settings = _open()
    try:
        return service.audio(conn, settings, _deps(settings), src, preset=preset, check_only=check)
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


def main() -> None:
    envfile.load_entry_env()
    mcp.run()


if __name__ == "__main__":
    main()
