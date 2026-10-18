"""`wonhee` command line. Expected errors (KeyError/ValueError) print one line; anything else is logged."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wonhee import envfile, errlog, llm, service
from wonhee.config import Settings, get_settings
from wonhee.db.connect import open_db
from wonhee.nodes.chain import ChainError

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
        print(f"{mark} {c.name}: {c.detail}" + (f" ({c.hint})" if c.hint else ""))
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


def _cmd_node(a, settings) -> int:
    from wonhee.nodes import registry
    if a.sub == "list":
        for n in registry.all_nodes().values():
            ins = ", ".join(f"{k}:{v}" for k, v in n.inputs.items()) or "-"
            outs = ", ".join(f"{k}:{v}" for k, v in n.outputs.items())
            print(f"{n.name}  [{ins} -> {outs}]  {n.doc}")
    elif a.sub == "show":
        n = registry.get(a.name)
        print(f"{n.name} (v{n.version}): {n.doc}")
        print("inputs:  " + (", ".join(f"{k}:{v}" for k, v in n.inputs.items()) or "-"))
        print("outputs: " + ", ".join(f"{k}:{v}" for k, v in n.outputs.items()))
        print("params (JSON schema):")
        print(json.dumps(n.params.model_json_schema(), indent=2))
    else:
        inputs = {}
        for item in a.inputs:
            if "=" not in item:
                raise ValueError(f"expected port=ref|path, got {item!r}")
            k, v = item.split("=", 1)
            inputs[k.strip()] = v
        conn = open_db(settings)
        _, text = service.node_run(conn, settings, build_deps(settings), a.name, inputs, parse_sets(a.set),
                                   wait=not a.detach, progress=_progress, no_cache=a.no_cache)
        print(text)
    return 0


def _cmd_chain(a, settings) -> int:
    conn = open_db(settings)
    if a.sub == "run":
        raw = sys.stdin.read() if a.file == "-" else Path(a.file).read_text()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ChainError([f"chain file is not valid JSON: {e}"]) from None
        _, text = service.chain_run(conn, settings, build_deps(settings), obj, wait=not a.detach,
                                    progress=_progress, no_cache=a.no_cache)
        print(text)
    elif a.sub == "show":
        print(service.chain_show(conn, settings, a.run_id))
    else:
        print(service.chain_exec(conn, settings, build_deps(settings), a.run_id, progress=_progress))
    return 0


def _cmd_account(a, settings) -> int:
    print(service.mark_account(open_db(settings), a.handle, a.relation))
    return 0


def _cmd_larp(a, settings) -> int:
    print(service.larp(open_db(settings), settings, build_deps(settings), a.task, image=a.image, video=a.video,
                       pick=a.pick, engine=a.engine, registers=a.register or None))
    return 0


def _cmd_inspire(a, settings) -> int:
    text, _ = service.inspire(open_db(settings), settings, build_deps(settings), a.ids or None, a.since,
                              a.relation, progress=_progress)
    print(text)
    return 0


def _cmd_ideate(a, settings) -> int:
    text, _ = service.ideate(open_db(settings), settings, build_deps(settings), n=a.n, focus=a.focus, engine=a.engine,
                             content_ids=a.ids or None, since=a.since, relation=a.relation, progress=_progress)
    print(text)
    return 0


def _cmd_vibe(a, settings) -> int:
    over = {}
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        over[k] = v if not v.replace(".", "", 1).lstrip("-").isdigit() else float(v)
    text, _ = service.vibe(open_db(settings), settings, build_deps(settings), a.like, [Path(p) for p in a.photos],
                           name=a.name, engine=a.engine, guide=not a.no_guide, guide_engine=a.guide_engine,
                           overrides=over, progress=_progress)
    print(text)
    return 0


def _cmd_audio(a, settings) -> int:
    conn, deps = open_db(settings), build_deps(settings)
    text = service.audio(conn, settings, deps, Path(a.file), preset=a.preset, check_only=a.check, isolate=a.isolate,
                         music_db=a.music_db,
                         dest=Path(a.out) if a.out else None, progress=_progress)
    print(text)
    return 0


def _cmd_export(a, settings) -> int:
    for p in service.export(open_db(settings), settings, a.ref, Path(a.dest)):
        print(p)
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
    s = sub.add_parser("node", help="list, inspect and run single nodes").add_subparsers(dest="sub", required=True)
    s.add_parser("list", help="all nodes").set_defaults(fn=_cmd_node)
    n = s.add_parser("show", help="one node, with its params schema"); n.add_argument("name"); n.set_defaults(fn=_cmd_node)
    n = s.add_parser("run", help="run one node: --in port=ref|path --set k=v")
    n.add_argument("name"); n.add_argument("--in", dest="inputs", action="append", default=[], metavar="port=ref|path")
    n.add_argument("--set", action="append", default=[], metavar="k=v")
    n.add_argument("--no-cache", action="store_true", help="do not reuse cached results")
    n.add_argument("--detach", action="store_true", help="run in the background, print the run id"); n.set_defaults(fn=_cmd_node)
    s = sub.add_parser("chain", help="run and inspect chains of nodes").add_subparsers(dest="sub", required=True)
    n = s.add_parser("run", help="run a chain JSON file (or - for stdin)"); n.add_argument("file")
    n.add_argument("--no-cache", action="store_true", help="do not reuse cached results")
    n.add_argument("--detach", action="store_true", help="run in the background, print the run id"); n.set_defaults(fn=_cmd_chain)
    n = s.add_parser("show", help="steps and final outputs of a chain run"); n.add_argument("run_id", type=int)
    n.set_defaults(fn=_cmd_chain)
    n = s.add_parser("exec", help=argparse.SUPPRESS); n.add_argument("run_id", type=int); n.set_defaults(fn=_cmd_chain)
    s = sub.add_parser("account", help="account bookkeeping").add_subparsers(dest="sub", required=True).add_parser("mark")
    s.add_argument("handle"); s.add_argument("relation", choices=list(service.RELATIONS)); s.set_defaults(fn=_cmd_account)
    s = sub.add_parser("db", help="database helpers").add_subparsers(dest="sub", required=True).add_parser("sql")
    s.add_argument("query"); s.set_defaults(fn=_cmd_sql)
    s = sub.add_parser("larp", help="frame any mundane task as a professional moment; optionally render it on media")
    s.add_argument("task", help='e.g. "drank an iced matcha next to nvtop"')
    media_in = s.add_mutually_exclusive_group()
    media_in.add_argument("--image", metavar="PATH|SHA", help="render the chosen line on this photo")
    media_in.add_argument("--video", metavar="PATH|SHA", help="render the chosen line on this clip")
    s.add_argument("--pick", type=int, help="which framing to render (default: the model's best)")
    s.add_argument("--engine", choices=["remote", "local"], default="remote", help="local = on-box model, $0")
    s.add_argument("--register", action="append", metavar="NAME",
                   help="ask for specific registers (repeatable), e.g. linkedin_announcement, earnings_call")
    s.set_defaults(fn=_cmd_larp)
    s = sub.add_parser("inspire", help="tear down stored posts and write an actionable report (what to film/build)")
    s.add_argument("ids", nargs="*", type=int, metavar="CONTENT_ID", help="default: every stored post with media")
    s.add_argument("--since", metavar="YYYY-MM-DD", help="only posts observed (grabbed) since this date")
    s.add_argument("--relation", choices=["ours", "friend", "external"]); s.set_defaults(fn=_cmd_inspire)
    s = sub.add_parser("ideate", help="ideas for posts and the b-roll to film, from saved posts + your brief "
                                      "(data/brief/*.md), each tagged with where it came from")
    s.add_argument("ids", nargs="*", type=int, metavar="CONTENT_ID", help="inspiration posts (default: all stored)")
    s.add_argument("-n", type=int, default=20, help="ideas to write (3-40)")
    s.add_argument("--focus", help='steer this batch, e.g. "cafe shoot day" or "carousels only"')
    s.add_argument("--engine", choices=["remote", "local"], default="remote", help="local = on-box model, $0")
    s.add_argument("--since", metavar="YYYY-MM-DD"); s.add_argument("--relation", choices=["ours", "friend", "external"])
    s.set_defaults(fn=_cmd_ideate)
    s = sub.add_parser("vibe", help="learn a look from reference photos, get a shooting guide, and edit your photos "
                                    "toward it (no image generation)")
    s.add_argument("photos", nargs="*", help="your photos to edit (none = just profile + guide)")
    s.add_argument("--like", action="append", required=True, metavar="REF",
                   help="reference folder, image or content:<id> photo post (repeatable)")
    s.add_argument("--name", default="look", help="name of the look, e.g. kpop-night")
    s.add_argument("--engine", choices=["local", "remote"], default="local", help="who picks the edit knobs")
    s.add_argument("--no-guide", action="store_true", help="skip the shooting guide")
    s.add_argument("--guide-engine", choices=["remote", "local"], default="remote",
                   help="who writes the guide: remote ~$0.003 (default, best), local $0")
    s.add_argument("--set", action="append", metavar="KNOB=VALUE", help="force an edit knob, e.g. fisheye=0.6")
    s.set_defaults(fn=_cmd_vibe)
    s = sub.add_parser("audio", help="check or clean a clip's audio on the box: denoise, peaks, -14 LUFS loudness")
    s.add_argument("file"); s.add_argument("--preset", choices=["voice", "mixed", "music"], default="voice")
    s.add_argument("--check", action="store_true", help="measure only"); s.add_argument("--out", help="write here")
    s.add_argument("--isolate", action="store_true", help="GPU: split the voice from music / street noise first")
    s.add_argument("--music-db", type=float, metavar="DB", help="with --isolate: keep the music this many dB down, e.g. -15")
    s.set_defaults(fn=_cmd_audio)
    s = sub.add_parser("export", help="copy a result (video, image, images manifest, content:<id>) into a folder")
    s.add_argument("ref", help="64-hex sha or content:<id>"); s.add_argument("dest"); s.set_defaults(fn=_cmd_export)
    s = sub.add_parser("profile", help="workload profiler (gauge): `wonhee profile doctor`, `... run <workload>`; "
                                       "see docs/profiling/GUIDE.md")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    return p


def _argv(argv) -> str:
    return " ".join(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    if raw[:1] == ["profile"]:  # hand the rest to gauge untouched; wonhee's settings and env are not needed
        from gauge.cli import main as gauge_main
        return gauge_main(raw[1:] or ["--help"])
    args = _parser().parse_args(argv)
    settings = None
    try:
        envfile.load_entry_env(args.env_file)
        settings = get_settings()
        return args.fn(args, settings)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except ChainError as e:
        errlog.log_traceback(settings.home if settings else None, f"wonhee {_argv(argv)} (expected)")
        print(e, file=sys.stderr)
        return 2
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
