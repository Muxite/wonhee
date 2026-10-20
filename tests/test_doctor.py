import sys
import types

from wonhee import doctor
from wonhee.doctor import checks, Check


def test_doctor_reports_missing_keys_by_name(settings, monkeypatch):
    monkeypatch.delenv("APIFY_API_KEY", raising=False)
    by = {c.name: c for c in checks(settings)}
    assert by["key APIFY_API_KEY"].ok is False and "not set" in by["key APIFY_API_KEY"].detail
    assert all("=" not in c.detail for c in by.values())  # never print values


def test_check_detail_never_contains_equals():
    assert "=" not in Check("x", False, "a=b").detail


def test_moeka_importable(settings):
    by = {c.name: c for c in checks(settings)}
    assert by["moeka import"].ok and by["moeka import"].severity == "required"


def test_key_sources_by_name_only(settings, monkeypatch, tmp_path):
    from wonhee import envfile
    for k in ("OPENROUTER_API_KEY", "BRAVE_API_KEY", "APIFY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "k.env"
    f.write_text("BRAVE_API_KEY=secretvalue\n")
    monkeypatch.setattr(envfile, "LOADED", {})
    envfile.load_env_file(f)
    monkeypatch.setenv("APIFY_API_KEY", "shellvalue")
    by = {c.name: c for c in checks(settings)}
    assert by["key BRAVE_API_KEY"].ok and str(f) in by["key BRAVE_API_KEY"].detail
    assert by["key APIFY_API_KEY"].ok and by["key APIFY_API_KEY"].detail == "shell env"
    assert not by["key OPENROUTER_API_KEY"].ok
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    assert "secretvalue" not in " ".join(c.detail for c in by.values())


def test_cuda_check(settings, monkeypatch):
    from wonhee.analyze import speech
    settings.whisper_device = "cuda"
    fake = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)

    def bad():
        raise RuntimeError("CUDA library package missing: pip install nvidia-cudnn-cu12")
    monkeypatch.setattr(speech, "_preload_cuda_libs", bad)
    c = {x.name: x for x in checks(settings)}["cuda"]
    assert not c.ok and c.severity == "required" and "nvidia-cudnn-cu12" in c.detail

    monkeypatch.setattr(speech, "_preload_cuda_libs", lambda: None)
    assert {x.name: x for x in checks(settings)}["cuda"].ok

    fake.get_cuda_device_count = lambda: 0
    c = {x.name: x for x in checks(settings)}["cuda"]
    assert not c.ok and "no CUDA device" in c.detail


def test_cuda_skipped_on_cpu(settings):
    settings.whisper_device = "cpu"
    assert {x.name: x for x in checks(settings)}["cuda"].ok


def test_prices_and_creator_are_warnings(settings):
    by = {c.name: c for c in checks(settings)}
    assert by["llm prices"].severity == "warning" and not by["llm prices"].ok
    assert by["creator.yaml"].severity == "warning" and "creator.example.yaml" in by["creator.yaml"].detail


def test_exit_code_only_on_required_failures(settings, monkeypatch, capsys):
    from wonhee.cli import main
    monkeypatch.setattr(doctor, "checks", lambda s: [Check("w", False, "meh", "warning"), Check("r", True, "fine")])
    assert main(["doctor"]) == 0
    monkeypatch.setattr(doctor, "checks", lambda s: [Check("r", False, "bad")])
    assert main(["doctor"]) == 1
    assert "✗" in capsys.readouterr().out


def test_stale_chain_run_is_a_warning(settings, conn):
    from wonhee.db import repo
    iid = repo.create_intent(conn, "x", "chain", 1, {"nodes": []})
    rid = repo.start_run(conn, repo.create_project(conn, "x", "chain", iid), 1.0)
    assert {c.name: c for c in checks(settings)}["stale chain runs"].ok
    conn.execute("UPDATE run SET started_at='2020-01-01T00:00:00+00:00' WHERE id=?", (rid,))
    c = {c.name: c for c in checks(settings)}["stale chain runs"]
    assert not c.ok and c.severity == "warning" and "force=true" in c.hint and "=" not in c.detail and str(rid) in c.detail


def test_missing_bundled_fonts_message_is_about_the_package_not_apt():
    from wonhee.doctor import _fonts_fix, _fonts_missing
    assert "package" in _fonts_missing(["bundled fonts"]) and "apt" not in _fonts_fix(["bundled fonts"])
    assert "apt" in _fonts_fix(["emoji"])
