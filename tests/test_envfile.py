import os
import subprocess
import sys
from pathlib import Path

import pytest

from wonhee import envfile, media, procenv
from wonhee.analyze import frames
from wonhee.sources import fetch


def test_load_env_file(monkeypatch, tmp_path):
    f = tmp_path / ".env"
    f.write_text("# c\n\nexport A_KEY='a'\nB_KEY=\"b\"\nC_KEY=c=d\nEXISTING=new\nbad line\n")
    for k in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EXISTING", "caller")
    monkeypatch.setattr(envfile, "LOADED", {})
    try:
        assert envfile.load_env_file(f) == ["A_KEY", "B_KEY", "C_KEY"]
        assert (os.environ["A_KEY"], os.environ["B_KEY"], os.environ["C_KEY"]) == ("a", "b", "c=d")
        assert os.environ["EXISTING"] == "caller"
        assert envfile.LOADED == {"A_KEY": str(f), "B_KEY": str(f), "C_KEY": str(f)}
    finally:
        for k in ("A_KEY", "B_KEY", "C_KEY"):
            os.environ.pop(k, None)


def test_missing_file_is_silent(tmp_path):
    assert envfile.load_env_file(tmp_path / "nope") == []


def test_cli_env_file_option(monkeypatch, tmp_path, capsys):
    from wonhee import doctor
    from wonhee.cli import main
    f = tmp_path / "x.env"
    f.write_text("WONHEE_TEST_ONLY_KEY=1\n")
    monkeypatch.delenv("WONHEE_TEST_ONLY_KEY", raising=False)
    monkeypatch.setattr(doctor, "checks", lambda s: [])
    try:
        assert main(["--env-file", str(f), "doctor"]) == 0
        assert os.environ["WONHEE_TEST_ONLY_KEY"] == "1"
    finally:
        os.environ.pop("WONHEE_TEST_ONLY_KEY", None)


def test_clean_env_has_no_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("APIFY_API_KEY", "y")
    env = procenv.clean_env()
    assert set(env) <= {"PATH", "LANG", "HOME", "TMPDIR"} and "PATH" in env
    assert not [k for k in env if k.endswith(("_API_KEY", "_TOKEN"))]


def test_subprocesses_get_clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    seen = []

    def fake_run(argv, **kw):
        seen.append(kw.get("env"))
        return subprocess.CompletedProcess(argv, 1, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        media.probe(tmp_path / "a.mp4")
    with pytest.raises(RuntimeError):
        frames.extract_frame(tmp_path / "a.mp4", 0, tmp_path / "f.jpg")
    with pytest.raises(fetch.DownloadError):
        fetch.LiveDownloader().download(None, "https://x/y", tmp_path / "d.mp4")
    assert len(seen) == 3 and all(e is not None and "OPENROUTER_API_KEY" not in e for e in seen)


def test_no_user_scoped_paths_in_package():
    root = Path(__file__).resolve().parent.parent / "wonhee"
    bad = [str(p) for p in root.rglob("*.py") if ".nanobot" in p.read_text()]
    assert bad == []
