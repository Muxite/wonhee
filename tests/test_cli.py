import json

import pytest

from wonhee import cli, service
from wonhee.cli import main
from wonhee.db import queries, repo
from wonhee.db.connect import open_db
from tests.fakes import fake_deps


def test_db_sql_refuses_writes(settings, capsys):
    assert main(["db", "sql", "DELETE FROM hashtag"]) == 2
    assert "read-only" in capsys.readouterr().err


def test_db_sql_select(settings, capsys):
    assert main(["db", "sql", "SELECT code FROM flag_code WHERE code='COST_CAP'"]) == 0
    assert "COST_CAP" in capsys.readouterr().out


def test_db_sql_cannot_write_through_with(settings, capsys):
    rc = main(["db", "sql", "WITH x AS (SELECT 1) DELETE FROM flag_code"])
    assert rc != 0
    assert "COST_CAP" in open_db(settings).execute("SELECT code FROM flag_code WHERE code='COST_CAP'").fetchone()[0]


def test_ask_prints_questions(settings, fake_llm, capsys):
    fake_llm.queue(ScoutExtract=[{"niche": "gym humor"}])
    assert main(["ask", "gym humor"]) == 0
    out = capsys.readouterr().out
    assert "Which time?" in out and "intent 1" in out


def test_resume_set_parsing():
    from wonhee.cli import parse_sets
    assert parse_sets(["count=8", "today=2026-10-15", "flag=true"]) == {"count": 8, "today": "2026-10-15", "flag": True}


def test_parse_sets_rejects_missing_equals():
    from wonhee.cli import parse_sets
    with pytest.raises(ValueError, match="k=v"):
        parse_sets(["oops"])


def test_show_unknown_run(settings, capsys):
    assert main(["show", "run", "99"]) == 1
    err = capsys.readouterr().err
    assert "no run 99" in err and "Traceback" not in err


def test_account_mark_unknown(settings, capsys):
    assert main(["account", "mark", "@nobody", "ours"]) == 1
    assert "nobody" in capsys.readouterr().err


def test_unexpected_error_logged_not_printed(settings, capsys, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(service, "show_run", boom)
    assert main(["show", "run", "1"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and "wonhee.log" in err
    assert "kaboom" in (settings.home / "logs" / "wonhee.log").read_text()


REQ = {"mode": "discover", "niche": "gym humor", "window": {"start": "2026-09-15", "end": "2026-10-15", "label": "now"},
       "constraints": {}}


def test_go_requires_confirmed(conn, settings):
    i = repo.create_intent(conn, "x", "scout", 1, REQ)
    with pytest.raises(ValueError, match=f"intent {i} is not confirmed yet — answer its questions or reply go"):
        service.go(conn, settings, fake_deps(), i)


def _confirmed(conn):
    i = repo.create_intent(conn, "x", "scout", 1, REQ)
    repo.confirm_intent(conn, i)
    return i


def _stub_stages(monkeypatch, stages):
    from wonhee.pipeline import registry
    monkeypatch.setattr(service, "SCOUT_STAGES", stages)
    monkeypatch.setitem(registry.STAGES_BY_KIND, "scout", stages)


def test_go_failure_shows_run_and_hint(conn, settings, monkeypatch):
    from wonhee.pipeline.context import Stage, StageFailed

    def fail(ctx):
        raise StageFailed("NO_CANDIDATES", "nothing in window")
    _stub_stages(monkeypatch, [Stage("resolve", (), lambda c: "ok"), Stage("discover", ("resolve",), fail)])
    run_id, text = service.go(conn, settings, fake_deps(), _confirmed(conn))
    assert "failed" in text and "NO_CANDIDATES" in text
    assert "wonhee scout" in text and "wider window" in text and "wonhee resume" not in text
    p = queries.project_of_run(conn, run_id)
    assert p["title"] == "gym humor" and p["kind"] == "scout"


def test_cost_cap_stop_then_resume_with_higher_cap(conn, settings, monkeypatch):
    from wonhee.pipeline.context import Stage
    calls = []

    def spend(ctx):
        ctx.add_cost(2.0)
        return "ok"

    def last(ctx):
        calls.append(1)
        return "ok"
    _stub_stages(monkeypatch, [Stage("resolve", (), spend), Stage("discover", ("resolve",), last)])
    run_id, text = service.go(conn, settings, fake_deps(), _confirmed(conn))
    assert queries.run_row(conn, run_id)["status"] == "stopped"
    assert f"wonhee resume {run_id} --from discover --set cost_cap_usd=" in text
    assert not calls
    out = service.resume(conn, settings, fake_deps(), run_id, "discover", {"cost_cap_usd": 5})
    assert calls == [1]
    row = queries.run_row(conn, run_id)
    assert row["cost_cap_usd"] == 5 and row["status"] == "done"
    step = queries.current_step(conn, run_id, "discover")
    assert "cost_cap_usd" not in json.loads(step["params_json"])
    assert out


def test_resume_unknown_stage_and_run(conn, settings):
    with pytest.raises(KeyError, match="no run 7"):
        service.resume(conn, settings, fake_deps(), 7, "pick", {})


def test_set_run_cap_validates(conn):
    i = repo.create_intent(conn, "x", "scout", 1, REQ)
    r = repo.start_run(conn, repo.create_project(conn, "t", "scout", i), 1.0)
    with pytest.raises(ValueError):
        repo.set_run_cap(conn, r, 0)
    with pytest.raises(KeyError):
        repo.set_run_cap(conn, 99, 2)


def test_scout_loop_asks_then_confirms_and_runs(settings, fake_llm, monkeypatch, capsys):
    from wonhee.pipeline.context import Stage
    monkeypatch.setattr(service, "SCOUT_STAGES", [Stage("resolve", (), lambda c: "ok")])
    monkeypatch.setattr(cli, "build_deps", lambda s: fake_deps())
    fake_llm.queue(ScoutExtract=[{"niche": "gym humor"}])
    answers = iter(["go", "go"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert main(["scout", "gym humor"]) == 0
    out = capsys.readouterr().out
    assert "intent 1" in out and "Confirmed" in out
    assert open_db(settings).execute("SELECT status FROM run WHERE id=1").fetchone()[0] == "done"


def test_scout_loop_exits_on_empty_line(settings, fake_llm, monkeypatch, capsys):
    fake_llm.queue(ScoutExtract=[{"niche": "gym humor"}])
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert main(["scout", "gym humor"]) == 0
    assert "wonhee ask --intent 1" in capsys.readouterr().out


def test_scout_loop_exits_on_eof(settings, fake_llm, monkeypatch, capsys):
    fake_llm.queue(ScoutExtract=[{"niche": "gym humor"}])

    def eof(*a):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert main(["scout", "gym humor"]) == 0
    assert "intent 1" in capsys.readouterr().out


def test_pragma_write_refused(settings, capsys):
    assert main(["db", "sql", "PRAGMA query_only=OFF"]) == 2
    assert "read-only" in capsys.readouterr().err


def test_keyboard_interrupt_exits_130(settings, monkeypatch, capsys):
    def stop(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(service, "show_run", stop)
    assert main(["show", "run", "1"]) == 130
    assert "Traceback" not in capsys.readouterr().err


def test_go_twice_reuses_project(conn, settings, monkeypatch):
    from wonhee.pipeline.context import Stage
    _stub_stages(monkeypatch, [Stage("resolve", (), lambda c: "ok")])
    i = _confirmed(conn)
    run_id, _ = service.go(conn, settings, fake_deps(), i)
    run2, text = service.go(conn, settings, fake_deps(), i)
    assert run2 == run_id and "already ran" in text
    assert conn.execute("SELECT count(*) FROM project").fetchone()[0] == 1


def test_go_refuses_while_run_in_progress(conn, settings):
    i = _confirmed(conn)
    p = repo.create_project(conn, "gym humor", "scout", i)
    r = repo.start_run(conn, p, 1.0)
    with pytest.raises(ValueError, match=f"intent {i} already has run {r} in progress .*wonhee show run {r}"):
        service.go(conn, settings, fake_deps(), i)
    assert conn.execute("SELECT count(*) FROM run").fetchone()[0] == 1


def test_go_starts_run_when_project_has_none(conn, settings, monkeypatch):
    from wonhee.pipeline.context import Stage
    _stub_stages(monkeypatch, [Stage("resolve", (), lambda c: "ok")])
    i = _confirmed(conn)
    repo.create_project(conn, "gym humor", "scout", i)
    run_id, _ = service.go(conn, settings, fake_deps(), i)
    assert conn.execute("SELECT count(*) FROM project").fetchone()[0] == 1 and run_id == 1


def test_no_what_to_change_header_when_empty(conn, settings, monkeypatch):
    from wonhee.pipeline.context import Stage
    _stub_stages(monkeypatch, [Stage("resolve", (), lambda c: "ok")])
    run_id, _ = service.go(conn, settings, fake_deps(), _confirmed(conn))
    repo.finish_run(conn, run_id, "failed")
    assert "What to change" not in service._outcome(conn, settings, run_id, "failed")


def test_mcp_server_tools_and_call(settings):
    import asyncio
    from wonhee.mcp_server import mcp
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    props = {n: set((getattr(t, "input_schema", None) or t.inputSchema)["properties"]) for n, t in tools.items()}
    assert props == {"scout_ask": {"text", "intent_id"}, "scout_go": {"intent_id"},
                     "scout_resume": {"run_id", "from_stage", "overrides"}, "scout_show_run": {"run_id"},
                     "library_grab": {"urls", "relation"},
                     "nodes_list": set(), "node_run": {"node", "inputs", "params", "wait"},
                     "chain_run": {"chain", "wait"}, "chain_show": {"run_id"}, "export": {"ref", "dest"}, "inspire": {"content_ids", "since", "relation"},
                     "ideate": {"n", "focus", "engine", "content_ids", "since"},
                     "vibe": {"like", "photos", "name", "engine", "guide"}, "audio": {"file", "preset", "check"},
                     "larp": {"task", "image", "video", "pick", "engine"}}
    res = asyncio.run(mcp.call_tool("scout_show_run", {"run_id": 99}))
    assert "no run 99" in str(res)


def test_go_failure_other_than_empty_suggests_resume(conn, settings, monkeypatch):
    from wonhee.pipeline.context import Stage

    def boom(ctx):
        raise RuntimeError("kaboom")
    _stub_stages(monkeypatch, [Stage("resolve", (), boom)])
    run_id, text = service.go(conn, settings, fake_deps(), _confirmed(conn))
    assert f"wonhee resume {run_id} --from resolve" in text


def test_what_to_change_uses_current_steps_only(conn, settings, monkeypatch):
    from wonhee.pipeline.context import Stage, StageFailed
    state = {"n": 0}

    def flaky(ctx):
        state["n"] += 1
        if state["n"] == 1:
            raise StageFailed("MISSING_KEY", "no key")
        return "ok"
    _stub_stages(monkeypatch, [Stage("resolve", (), flaky)])
    run_id, first = service.go(conn, settings, fake_deps(), _confirmed(conn))
    assert "MISSING_KEY" in first or "key" in first
    out = service.resume(conn, settings, fake_deps(), run_id, "resolve", {})
    assert "What to change" not in out and "MISSING_KEY" not in out.split("\n\n")[-1]


def test_resume_rejects_unknown_override_keys(conn, settings, monkeypatch):
    from wonhee.pipeline.context import Stage
    _stub_stages(monkeypatch, [Stage("resolve", (), lambda c: "ok")])
    run_id, _ = service.go(conn, settings, fake_deps(), _confirmed(conn))
    with pytest.raises(ValueError, match="unknown --set key.*cnt.*valid keys: count, today, cost_cap_usd, force, relation, refetch"):
        service.resume(conn, settings, fake_deps(), run_id, "resolve", {"cnt": 2})


def test_resume_refuses_running_run(conn, settings):
    i = _confirmed(conn)
    p = repo.create_project(conn, "x", "scout", i)
    r = repo.start_run(conn, p, 1.0)
    with pytest.raises(ValueError, match=f"run {r} is in progress"):
        service.resume(conn, settings, fake_deps(), r, "resolve", {})
    assert "--set force=true" in str(pytest.raises(ValueError, service.resume, conn, settings, fake_deps(), r,
                                                    "resolve", {}).value)


def test_go_preflight_missing_apify_token(conn, settings):
    from wonhee.pipeline.deps import Missing
    with pytest.raises(ValueError, match="APIFY_API_KEY.*\.env.*wonhee doctor"):
        service.go(conn, settings, fake_deps(apify=Missing("APIFY_API_KEY")), _confirmed(conn))
    assert conn.execute("SELECT count(*) FROM project").fetchone()[0] == 0


def test_llm_error_is_friendly_and_logged(settings, monkeypatch, capsys):
    from wonhee import llm

    def boom(*a, **k):
        raise llm.SchemaFail("ScoutExtract", "bad\njson")
    monkeypatch.setattr(service, "ask", boom)
    assert main(["ask", "hi"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "model call failed" in err and "wonhee.log" in err and "Traceback" not in err
    assert "SchemaFail" in (settings.home / "logs" / "wonhee.log").read_text()


def test_expected_error_traceback_logged(settings, capsys):
    assert main(["show", "run", "99"]) == 1
    assert "no run 99" in capsys.readouterr().err
    assert "KeyError" in (settings.home / "logs" / "wonhee.log").read_text()


def test_mcp_guard_llm_error_and_expected_logged(settings):
    from wonhee import llm
    from wonhee.mcp_server import _guard

    @_guard
    def bad_llm():
        raise llm.LLMError("backend down\nsecond line")

    @_guard
    def bad_key():
        raise KeyError("no run 5")
    out = bad_llm()
    assert out.startswith("model call failed: backend down second line") and "wonhee.log" in out
    assert bad_key() == "error: no run 5"
    log = (settings.home / "logs" / "wonhee.log").read_text()
    assert "LLMError" in log and "KeyError" in log


def test_cli_go_and_resume_print_progress_to_stderr(conn, settings, monkeypatch, capsys):
    from wonhee.pipeline.context import Stage
    _stub_stages(monkeypatch, [Stage("resolve", (), lambda c: "ok")])
    monkeypatch.setattr(cli, "build_deps", lambda s: fake_deps())
    i = _confirmed(conn)
    assert main(["go", str(i)]) == 0
    cap = capsys.readouterr()
    assert "stage 1/1 resolve…" in cap.err and "resolve: ok" in cap.err
    assert "stage 1/1" not in cap.out
