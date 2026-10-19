"""Infra fixes: atomic writes, local-model retries, probe edge cases, rollback, cli/mcp/export guards."""
import json
import subprocess
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from wonhee import fsutil, localllm, media
from wonhee.config import Settings
from wonhee.db.connect import connect, tx


class Out(BaseModel):
    n: int


def test_write_atomic_and_unique_dir(tmp_path):
    p = fsutil.write_atomic(tmp_path / "a" / "r.md", "héllo")
    assert p.read_text(encoding="utf-8") == "héllo" and [x.name for x in p.parent.iterdir()] == ["r.md"]
    fsutil.write_atomic(p, "two")
    assert p.read_text() == "two"
    d1 = fsutil.unique_dir(tmp_path, "run")
    assert fsutil.unique_dir(tmp_path, "run") == d1  # empty dirs are reused
    (d1 / "x").write_text("1")
    d2 = fsutil.unique_dir(tmp_path, "run")
    (d2 / "x").write_text("1")
    assert (d2.name, fsutil.unique_dir(tmp_path, "run").name) == ("run-2", "run-3")


class Resp:
    def __init__(self, code, data=None):
        self.status_code, self._d = code, data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("x", request=httpx.Request("POST", "http://x"), response=self)

    def json(self):
        return self._d


def test_post_retries_transport_errors_then_succeeds(monkeypatch):
    seq = [httpx.ConnectError("no"), Resp(503), Resp(200, {"ok": 1})]
    sleeps = []
    monkeypatch.setattr(localllm, "_sleep", sleeps.append)

    def post(url, json, timeout):
        assert timeout.connect == 10
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(httpx, "post", post)
    assert localllm._post(Settings(), {}) == {"ok": 1} and sleeps == [1.0, 3.0]


def test_post_gives_up_after_three_tries(monkeypatch):
    monkeypatch.setattr(localllm, "_sleep", lambda s: None)
    n = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: n.append(1) or (_ for _ in ()).throw(httpx.ReadTimeout("t")))
    with pytest.raises(httpx.ReadTimeout):
        localllm._post(Settings(), {})
    assert len(n) == 3


def test_cjk_estimate_counts_a_token_per_character():
    assert localllm.est_tokens("한국어テスト中文") == 8
    assert localllm.est_tokens("abcdefghi") == 3


def test_model_ready_matches_latest_both_ways(monkeypatch):
    def tags(names):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(200, {"models": [{"name": n} for n in names]}))
    tags(["gemma4:e4b", "qwen:latest"])
    assert localllm.model_ready(Settings(local_llm_model="qwen"))[0]
    assert localllm.model_ready(Settings(local_llm_model="gemma4:e4b"))[0]
    assert not localllm.model_ready(Settings(local_llm_model="llama"))[0]
    tags(["qwen"])
    assert localllm.model_ready(Settings(local_llm_model="qwen:latest"))[0]
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(500))
    assert not localllm.model_ready(Settings())[0]


def test_validation_retry_carries_the_error(monkeypatch):
    bodies = []

    def post(s, body):
        bodies.append(body)
        return {"message": {"content": "nope" if len(bodies) == 1 else '{"n": 2}'}}
    monkeypatch.setattr(localllm, "_post", post)
    assert localllm.chat_json(Settings(), Out, system="s", prompt="p").n == 2
    assert bodies[0]["messages"] != bodies[1]["messages"]
    last = bodies[1]["messages"][-1]
    assert last["role"] == "user" and "failed validation" in last["content"] and "valid JSON" in last["content"]
    assert bodies[1]["keep_alive"] == "30m"


def test_translate_reasks_once_on_count_mismatch(monkeypatch):
    replies = iter([["a"], ["x", "y"]])
    prompts = []

    def post(s, body):
        prompts.append(body["messages"][1]["content"])
        return {"message": {"content": json.dumps({"texts": next(replies)})}}
    monkeypatch.setattr(localllm, "_post", post)
    assert localllm.translate_texts(Settings(), ["1", "2"]) == ["x", "y"]
    assert len(prompts) == 2 and "exactly 2" in prompts[1]
    replies = iter([["a"], ["b"]])
    with pytest.raises(localllm.LocalLLMError):
        localllm.translate_texts(Settings(), ["1", "2"])


def _fake_probe(monkeypatch, info):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(info), ""))


def test_probe_handles_na_and_odd_frame_rates(monkeypatch):
    _fake_probe(monkeypatch, {"format": {"duration": "N/A"}, "streams": [
        {"codec_type": "video", "width": 4, "height": 2, "avg_frame_rate": "30"}]})
    p = media.probe(Path("x"))
    assert (p.duration_ms, p.fps, p.width) == (None, 30.0, 4)
    _fake_probe(monkeypatch, {"format": {"duration": "1.5"}, "streams": [
        {"codec_type": "video", "avg_frame_rate": "0/0"}]})
    p = media.probe(Path("x"))
    assert (p.duration_ms, p.fps) == (1500, None)


def test_probe_missing_ffprobe_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ffprobe")))
    with pytest.raises(RuntimeError, match="not found"):
        media.probe(Path("x"))


def test_ffmpeg_timeout_is_reported(monkeypatch):
    def run(*a, **k):
        assert k["timeout"] == 1800
        raise subprocess.TimeoutExpired("ffmpeg", k["timeout"])
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="timed out"):
        fsutil.run_timed(["ffmpeg", "-i", "x"])


def test_separate_errors_carry_stderr_and_handle_odd_audio(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    from wonhee.edit import separate
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b"boom: bad stream"))
    with pytest.raises(RuntimeError, match="boom: bad stream"):
        separate._decode(tmp_path / "a.mp4", 44100)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, b"\0" * 5, b""))
    with pytest.raises(ValueError, match="no decodable audio"):
        separate._decode(tmp_path / "a.mp4", 44100)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, b"\0" * 19, b""))
    assert separate._decode(tmp_path / "a.mp4", 44100).shape == (2, 2)


def test_rnn_model_rejects_a_bad_download(monkeypatch, tmp_path):
    from wonhee.edit import audio
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(200))
    Resp.content = b"<html>error</html>" * 2000
    assert audio.rnn_model(tmp_path / "m") is None and not (tmp_path / "m" / "rnnoise-sh.rnnn").exists()


def test_tx_rollback_error_keeps_the_original(tmp_path):
    conn = connect(tmp_path / "t.db")

    class Wrap:
        in_transaction = False

        def execute(self, sql, *a):
            if sql == "ROLLBACK":
                conn.execute("ROLLBACK")
                import sqlite3
                raise sqlite3.OperationalError("cannot rollback")
            return conn.execute(sql, *a)
    with pytest.raises(KeyError, match="orig"):
        with tx(Wrap()):
            raise KeyError("orig")


def test_cli_set_parsing():
    from wonhee.cli import parse_sets
    assert parse_sets(["a=3", "b=2.5", "c=true", "d=False", "e=hello", "f=1_0", "g=nan", "h=[1, 2]", "i=-4"]) == {
        "a": 3, "b": 2.5, "c": True, "d": False, "e": "hello", "f": "1_0", "g": "nan", "h": [1, 2], "i": -4}


def test_mcp_rejects_paths_outside_allowed_roots(settings, tmp_path):
    from wonhee import mcp_server
    assert "error" in mcp_server.export("a" * 64, "/etc/wonhee-out")
    assert "error" in mcp_server.audio("/etc/passwd")
    assert "error" in mcp_server.vibe(["content:1"], ["/etc/passwd"])
    assert mcp_server._in_path("~/x") == Path.home().resolve() / "x"
    assert mcp_server._out_path(str(settings.home / "o")) == (settings.home / "o").resolve()


def test_guard_survives_a_broken_error_log(settings, monkeypatch):
    from wonhee import errlog, mcp_server
    monkeypatch.setattr(errlog, "log_traceback", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))

    @mcp_server._guard
    def f():
        raise ValueError("bad")
    assert f() == "error: bad"


def test_export_to_a_file_path_and_missing_source(conn, settings, tmp_path):
    from wonhee import service
    from wonhee.db import repo
    from wonhee.media import path_for
    img = tmp_path / "i.jpg"
    img.write_bytes(b"\xff\xd8jpegish")
    sha = repo.save_file(conn, settings.media_dir, img, "image", "image/jpeg")
    f = tmp_path / "dest"
    f.write_text("i am a file")
    with pytest.raises(ValueError, match="is a file"):
        service.export(conn, settings, sha, f)
    path_for(settings.media_dir, sha).unlink()
    with pytest.raises(ValueError, match="missing"):
        service.export(conn, settings, sha, tmp_path / "out")
