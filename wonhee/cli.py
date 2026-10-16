"""`wonhee` command line. Expected errors (KeyError/ValueError) print one line; anything else is logged."""
from __future__ import annotations

import argparse
import json
import sys

from wonhee import envfile, errlog, llm, service
from wonhee.config import Settings, get_settings
from wonhee.db.connect import open_db

READ_ONLY_PREFIXES = ("select", "with", "pragma")


def parse_sets(items: list[str]) -> dict:
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected k=v, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def build_deps(settings: Settings):
    from wonhee.pipeline.deps import Deps
    return Deps.live(settings)


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _table(cursor) -> str:
    cols = [d[0] for d in cursor.description or []]
    rows = [["" if v is None else str(v) for v in r] for r in cursor.fetchall()]
    widths = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(cols)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
    return "\n".join([line(cols), line(["-" * w for w in widths]), *(line(r) for r in rows), f"({len(rows)} rows)"])


def _cmd_doctor(a, settings) -> int:
    from wonhee import doctor
    failed = 0
    for c in doctor.checks(settings):
        mark = "✓" if c.ok else ("✗" if c.severity == "required" else "!")
        print(f"{mark} {c.name}: {c.detail}")
        failed += (not c.ok) and c.severity == "required"
    print("all required checks pass" if not failed else f"{failed} required check(s) failed")
    return 1 if failed else 0


def _cmd_ask(a, settings) -> int:
    conn = open_db(settings)
    r = service.ask(conn, settings, a.text, a.intent)
    print(f"{r.message}\n[intent {r.intent_id}: {r.status}]")
    return 0


def _cmd_scout(a, settings) -> int:
    conn = open_db(settings)
    r = service.ask(conn, settings, a.text)
    while True:
        print(f"{r.message}\n[intent {r.intent_id}: {r.status}]")
        if r.status == "confirmed":
            break
        try:
            line = input("> ").strip()
        except EOFError:
            line = ""
        if not line:
            print(f"Paused. Continue later with: wonhee ask --intent {r.intent_id} \"<your answer>\"")
            return 0
        r = service.ask(conn, settings, line, r.intent_id)
    print("Running it now; progress goes to stderr.")
    run_id, text = service.go(conn, settings, build_deps(settings), r.intent_id, progress=_progress)
    print(text)
    return 0


def _cmd_go(a, settings) -> int:
    conn = open_db(settings)
    print(service.go(conn, settings, build_deps(settings), a.intent_id, progress=_progress)[1])
    return 0


def _cmd_resume(a, settings) -> int:
    conn = open_db(settings)
    print(service.resume(conn, settings, build_deps(settings), a.run_id, a.from_stage, parse_sets(a.set),
                         progress=_progress))
    return 0


def _cmd_show(a, settings) -> int:
    conn = open_db(settings)
    if a.what == "run":
        print(service.show_run(conn, a.id, settings.media_dir))
    else:
        print(service.show_project(conn, a.id, settings.media_dir))
    return 0


def _cmd_grab(a, settings) -> int:
    conn = open_db(settings)
    relation = "ours" if a.mine else "friend" if a.friend else "external"
    run_id, text = service.grab(conn, settings, build_deps(settings), a.urls, relation, progress=_progress,
                                refetch=a.refetch)
    print(text)
    print(f"\nrun {run_id} · details: wonhee show run {run_id}")
    return 0


def _cmd_account(a, settings) -> int:
    print(service.mark_account(open_db(settings), a.handle, a.relation))
    return 0


def _cmd_sql(a, settings) -> int:
    q = a.query.strip()
    if not q.lower().startswith(READ_ONLY_PREFIXES) or (q.lower().startswith("pragma") and "=" in q):
        print("db sql is read-only: only SELECT, WITH and read-form PRAGMA queries are allowed", file=sys.stderr)
        return 2
    conn = open_db(settings)
    conn.execute("PRAGMA query_only = ON")
    try:
        print(_table(conn.execute(q)))
    except Exception as e:  # noqa: BLE001 - sqlite errors are the user's query problem
        import sqlite3
        if isinstance(e, sqlite3.Error):
            print(f"sql error: {e}", file=sys.stderr)
            return 1
        raise
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wonhee", description="Trend scout: request -> confirmed scout -> shot list.")
    p.add_argument("--env-file", metavar="PATH",
                   help="KEY=VALUE file with API keys (default: $WONHEE_ENV_FILE, else .env in the repo); "
                        "variables already in the environment win")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the environment").set_defaults(fn=_cmd_doctor)
    s = sub.add_parser("scout", help="interactive: ask, confirm, run"); s.add_argument("text"); s.set_defaults(fn=_cmd_scout)
    s = sub.add_parser("ask", help="one non-interactive compile turn"); s.add_argument("text")
    s.add_argument("--intent", type=int); s.set_defaults(fn=_cmd_ask)
    s = sub.add_parser("go", help="run a confirmed intent"); s.add_argument("intent_id", type=int); s.set_defaults(fn=_cmd_go)
    s = sub.add_parser("resume", help="re-run from a stage (e.g. --set cost_cap_usd=3)")
    s.add_argument("run_id", type=int); s.add_argument("--from", dest="from_stage", required=True)
    s.add_argument("--set", action="append", default=[], metavar="k=v"); s.set_defaults(fn=_cmd_resume)
    s = sub.add_parser("show", help="show a run or a project's shot list")
    s.add_argument("what", choices=["run", "project"]); s.add_argument("id", type=int); s.set_defaults(fn=_cmd_show)
    s = sub.add_parser("grab", help="store reels by URL: data, metric snapshot, video, analysis")
    s.add_argument("urls", nargs="+", metavar="URL")
    who = s.add_mutually_exclusive_group()
    who.add_argument("--mine", action="store_true", help="the post is ours (marks the account ours)")
    who.add_argument("--friend", action="store_true", help="the post is a friend's")
    s.add_argument("--refetch", action="store_true", help="download the video again even if one is stored")
    s.set_defaults(fn=_cmd_grab)
    s = sub.add_parser("account", help="account bookkeeping").add_subparsers(dest="sub", required=True).add_parser("mark")
    s.add_argument("handle"); s.add_argument("relation", choices=list(service.RELATIONS)); s.set_defaults(fn=_cmd_account)
    s = sub.add_parser("db", help="database helpers").add_subparsers(dest="sub", required=True).add_parser("sql")
    s.add_argument("query"); s.set_defaults(fn=_cmd_sql)
    return p


def _argv(argv) -> str:
    return " ".join(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = None
    try:
        envfile.load_entry_env(args.env_file)
        settings = get_settings()
        return args.fn(args, settings)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (KeyError, ValueError) as e:
        errlog.log_traceback(settings.home if settings else None, f"wonhee {_argv(argv)} (expected)")
        print(e.args[0] if e.args else e, file=sys.stderr)
        return 1
    except llm.LLMError as e:
        log = errlog.log_traceback(settings.home if settings else None, f"wonhee {_argv(argv)}")
        print(f"model call failed: {errlog.one_line(e)} (details: {log})", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - unexpected: log the traceback, point to it
        log = errlog.log_traceback(settings.home if settings else None, f"wonhee {_argv(argv)}")
        print(f"unexpected error — see {log}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
