"""MCP server `wonhee` (stdio). Not registered in moeka yet. Run: python -m wonhee.mcp_server"""
from __future__ import annotations

import functools
from pathlib import Path

from wonhee import envfile, errlog, llm, service
from wonhee.config import get_settings
from wonhee.db.connect import open_db

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ModuleNotFoundError:  # mcp 2.x renamed it
    from mcp.server.mcpserver import MCPServer as _Server

mcp = _Server("wonhee")


def _guard(fn):
    """Expected errors become readable text; unexpected ones are logged and pointed to."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except (KeyError, ValueError) as e:
            errlog.log_traceback(get_settings().home, f"mcp {fn.__name__} (expected)")
            return f"error: {e.args[0] if e.args else e}"
        except llm.LLMError as e:
            log = errlog.log_traceback(get_settings().home, f"mcp {fn.__name__}")
            return f"model call failed: {errlog.one_line(e)} (details: {log})"
        except Exception:  # noqa: BLE001
            log = errlog.log_traceback(get_settings().home, f"mcp {fn.__name__}")
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
def export(ref: str, dest: str) -> str:
    """Copy a stored result into a local folder for posting: a video or image (one file), an images manifest or
    content:<id> photo post (01.jpg, 02.jpg, ...). Returns the written paths."""
    conn, settings = _open()
    try:
        return "\n".join(str(p) for p in service.export(conn, settings, ref, Path(dest)))
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
