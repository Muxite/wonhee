import json
import os
import sqlite3
import subprocess
import sys
import time

import pytest

from wonhee import service
from wonhee.cli import main
from wonhee.config import REPO_ROOT
from wonhee.db import queries
from wonhee.nodes.chain import ChainError
from wonhee.nodes.subtitles import Line, Subtitles
from wonhee.procenv import clean_env

SUBS = Subtitles(language="en", lines=[
    Line(id=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"line {i}", source="speech") for i in range(5)])


@pytest.fixture
def subs_file(tmp_path):
    p = tmp_path / "subs.json"
    p.write_text(SUBS.model_dump_json())
    return p


def counts(conn):
    return [conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("intent", "project", "run", "step")]


def test_chain_run_invalid_records_nothing(conn, settings, subs_file):
    before = counts(conn)
    chain = [{"id": "a", "node": "to_srt", "in": {"subtitles": str(subs_file)}},
             {"id": "b", "node": "nope"}, {"id": "c", "node": "to_srt", "in": {"subtitles": "zz.file"}}]
    with pytest.raises(ChainError) as e:
        service.chain_run(conn, settings, None, chain)
    assert len(e.value.problems) >= 2
    assert counts(conn) == before


def test_path_literal_is_stored_as_sha(conn, settings, subs_file):
    rid, text = service.chain_run(conn, settings, None, [{"id": "a", "node": "to_srt", "in": {"subtitles": str(subs_file)}}])
    stored = json.loads(queries.intent_row(conn, 1)["compiled_json"])["nodes"][0]["in"]["subtitles"]
    assert len(stored) == 64 and stored != str(subs_file)
    assert queries.run_row(conn, rid)["status"] == "done"
    assert "a.file file" in text


def test_non_subtitles_json_path_is_a_chain_error(conn, settings, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{\"lines\": 3}")
    with pytest.raises(ChainError, match="not a valid subtitles"):
        service.node_run(conn, settings, None, "to_srt", {"subtitles": str(bad)})


def test_node_run_to_srt(conn, settings, subs_file):
    rid, text = service.node_run(conn, settings, None, "to_srt", {"subtitles": str(subs_file)}, {})
    out = queries.step_outputs(conn, queries.current_step(conn, rid, "to_srt")["id"])["file"]
    from wonhee.media import path_for
    assert "line 0" in path_for(settings.media_dir, out["ref"]).read_text()
    assert str(path_for(settings.media_dir, out["ref"])) in text


def test_render_chain_shows_final_outputs_only(conn, settings, subs_file):
    chain = [{"id": "a", "node": "to_srt", "in": {"subtitles": str(subs_file)}}]
    rid, _ = service.chain_run(conn, settings, None, chain)
    text = service.chain_show(conn, settings, rid)
    assert "outputs:" in text and "a.file file" in text


def test_render_chain_hides_consumed_outputs(conn, settings, toy_registry):
    from pydantic import BaseModel, ConfigDict

    from wonhee.nodes.base import Node

    class P(BaseModel):
        model_config = ConfigDict(extra="forbid")
    h = lambda c: c * 64  # noqa: E731
    toy_registry(Node("t_src", 1, {}, {"out": "text"}, P, lambda c, i, p: {"out": h("a")}, "toy", cacheable=False))
    toy_registry(Node("t_mid", 1, {"in": "text"}, {"out": "text"}, P, lambda c, i, p: {"out": h("b")}, "toy"))
    rid, text = service.chain_run(conn, settings, None, [
        {"id": "s", "node": "t_src"}, {"id": "m", "node": "t_mid", "in": {"in": "s.out"}}])
    out = text.split("outputs:")[1]
    assert "m.out text" in out and "s.out" not in out


def test_chain_exec_done_is_noop(conn, settings, subs_file):
    rid, _ = service.node_run(conn, settings, None, "to_srt", {"subtitles": str(subs_file)})
    steps = counts(conn)
    assert "outputs:" in service.chain_exec(conn, settings, None, rid)
    assert counts(conn) == steps


def test_chain_exec_takes_over_running_run(conn, settings, subs_file):
    from wonhee.db import repo
    from wonhee.nodes.chain import parse_chain
    sha = service._literal_ref(conn, settings, "subtitles", subs_file)
    chain = parse_chain([{"id": "a", "node": "to_srt", "in": {"subtitles": sha}}])
    iid = repo.create_intent(conn, "x", "chain", 1, chain.model_dump(by_alias=True, mode="json"))
    repo.confirm_intent(conn, iid)
    rid = repo.start_run(conn, repo.create_project(conn, "x", "chain", iid), 1.0)
    service.chain_exec(conn, settings, None, rid)
    assert queries.run_row(conn, rid)["status"] == "done"
    with pytest.raises(ValueError, match="in progress"):
        repo.reopen_run(conn, rid)
        service.resume(conn, settings, None, rid, "a", {})


def test_cli_node_list_has_all_nodes(settings, capsys):
    assert main(["node", "list"]) == 0
    out = capsys.readouterr().out
    for n in ("grab", "speech_subs", "merge_subs", "translate", "screen_text", "burn_subtitles", "to_srt"):
        assert n in out


def test_cli_node_show_has_schema(settings, capsys):
    assert main(["node", "show", "to_srt"]) == 0
    assert "max_chars" in capsys.readouterr().out


def test_cli_invalid_chain_exits_2(settings, tmp_path, capsys):
    f = tmp_path / "c.json"
    f.write_text(json.dumps([{"id": "a", "node": "nope"}]))
    assert main(["chain", "run", str(f)]) == 2
    assert "unknown node" in capsys.readouterr().err


def test_detached_chain_run(tmp_path, subs_file):
    home = tmp_path / "home"
    env = {**clean_env(), "WONHEE_HOME": str(home), "WONHEE_ENV_FILE": str(tmp_path / "none.env")}
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps([{"id": "a", "node": "to_srt", "in": {"subtitles": str(subs_file)}}]))
    p = subprocess.run([sys.executable, "-m", "wonhee.cli", "chain", "run", str(chain), "--detach"],
                       cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "started in the background" in p.stdout
    db = home / "wonhee.db"
    status, deadline = None, time.time() + 20
    while time.time() < deadline:
        try:
            c = sqlite3.connect(db)
            row = c.execute("SELECT status FROM run WHERE id=1").fetchone()
            c.close()
            status = row[0] if row else None
        except sqlite3.Error:
            pass
        if status == "done":
            break
        time.sleep(0.3)
    assert status == "done"
    assert (home / "logs" / "run-1.log").exists()


def test_child_env_filters_and_absolutizes(monkeypatch, tmp_path):
    from wonhee.nodes.detach import child_env
    monkeypatch.setenv("FOO_SECRET", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel").mkdir()
    monkeypatch.setenv("WONHEE_HOME", "rel")
    env = child_env()
    assert "FOO_SECRET" not in env and env["OPENROUTER_API_KEY"] == "k" and "PATH" in env
    assert env["WONHEE_HOME"] == str(tmp_path / "rel")


def test_spawn_failure_marks_run_failed(conn, settings, subs_file, monkeypatch):
    from wonhee.nodes import detach
    monkeypatch.setattr(detach.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(OSError):
        service.chain_run(conn, settings, None, [{"id": "a", "node": "to_srt", "in": {"subtitles": str(subs_file)}}],
                          wait=False)
    assert queries.run_row(conn, 1)["status"] == "failed"


def test_chain_exec_refuses_step_in_progress(conn, settings, subs_file):
    from wonhee.db import repo
    rid, _ = service.chain_run(conn, settings, None, [{"id": "a", "node": "to_srt", "in": {"subtitles": str(subs_file)}}])
    repo.reopen_run(conn, rid)
    repo.start_step(conn, rid, "a", {}, queries.current_step(conn, rid, "a")["id"])
    with pytest.raises(ValueError, match="in progress"):
        service.chain_exec(conn, settings, None, rid)


def test_render_chain_shows_unconsumed_port_of_consumed_node(conn, settings, toy_registry):
    from pydantic import BaseModel, ConfigDict

    from wonhee.nodes.base import Node

    class P(BaseModel):
        model_config = ConfigDict(extra="forbid")
    h = lambda c: c * 64  # noqa: E731
    toy_registry(Node("t_two", 1, {}, {"subs": "text", "notes": "text"}, P,
                      lambda c, i, p: {"subs": h("a"), "notes": h("b")}, "toy", cacheable=False))
    toy_registry(Node("t_use", 1, {"in": "text"}, {"out": "text"}, P, lambda c, i, p: {"out": h("c")}, "toy"))
    _, text = service.chain_run(conn, settings, None, [
        {"id": "t", "node": "t_two"}, {"id": "u", "node": "t_use", "in": {"in": "t.subs"}}])
    out = text.split("outputs:")[1]
    assert "t.notes text" in out and "u.out text" in out and "t.subs" not in out


def test_missing_refs_and_files_are_problems_and_record_nothing(conn, settings):
    before = counts(conn)
    chain = [{"id": "a", "node": "to_srt", "in": {"subtitles": "ab" * 32}},
             {"id": "b", "node": "translate", "in": {"subs": "a.subtitles", "caption": "content:999"}},
             {"id": "c", "node": "to_srt", "in": {"subtitles": "/no/such/clip.json"}}]
    with pytest.raises(ChainError) as e:
        service.chain_run(conn, settings, None, chain)
    text = "\n".join(e.value.problems)
    assert "no such blob abababababab" in text and "no content 999" in text
    assert "`/no/such/clip.json`: no such file, ref or wire" in text and "not of the form" not in text
    assert counts(conn) == before


def test_no_cache_flag_is_stored_and_cost_cap_must_be_positive(conn, settings, subs_file):
    from wonhee.nodes.chain import parse_chain
    rid, _ = service.node_run(conn, settings, None, "to_srt", {"subtitles": str(subs_file)}, no_cache=True)
    stored = json.loads(queries.intent_row(conn, 1)["compiled_json"])
    assert stored["no_cache"] is True
    with pytest.raises(ChainError):
        parse_chain({"nodes": [], "cost_cap_usd": 0})


def test_cli_no_cache_option_parses():
    from wonhee.cli import _parser
    p = _parser()
    assert p.parse_args(["chain", "run", "f.json", "--no-cache"]).no_cache is True
    assert p.parse_args(["node", "run", "to_srt", "--no-cache"]).no_cache is True
    assert p.parse_args(["chain", "run", "f.json"]).no_cache is False


def test_child_env_passes_gpu_and_cache_vars(monkeypatch):
    from wonhee.nodes.detach import child_env
    for k, v in (("CUDA_VISIBLE_DEVICES", "1"), ("HF_HOME", "/hf"), ("HF_HUB_OFFLINE", "1"), ("XDG_CACHE_HOME", "/xdg")):
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("SOME_OTHER", "y")
    env = child_env()
    assert (env["CUDA_VISIBLE_DEVICES"], env["HF_HOME"], env["HF_HUB_OFFLINE"], env["XDG_CACHE_HOME"]) == ("1", "/hf", "1", "/xdg")
    assert "SOME_OTHER" not in env
    monkeypatch.delenv("HF_HOME")
    assert "HF_HOME" not in child_env()


def test_spawn_forwards_env_file(conn, settings, subs_file, monkeypatch, tmp_path):
    from wonhee import envfile
    from wonhee.nodes import detach
    seen = {}
    monkeypatch.setattr(detach.subprocess, "Popen", lambda argv, **k: seen.setdefault("argv", argv))
    monkeypatch.setattr(envfile, "EXPLICIT", [])
    envfile.load_entry_env(tmp_path / "my.env")
    service.chain_run(conn, settings, None, [{"id": "a", "node": "to_srt", "in": {"subtitles": str(subs_file)}}], wait=False)
    argv = seen["argv"]
    assert argv[argv.index("--env-file") + 1] == str(tmp_path / "my.env")
    assert argv.index("--env-file") < argv.index("chain")
