"""MCP server `wonhee` (stdio). Not registered in moeka yet. Run: python -m wonhee.mcp_server"""
from __future__ import annotations

import functools

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


def main() -> None:
    envfile.load_entry_env()
    mcp.run()


if __name__ == "__main__":
    main()
