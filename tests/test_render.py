from datetime import date

from wonhee.db import repo
from wonhee.media import path_for
from wonhee.pipeline.context import Stage
from wonhee.pipeline.orchestrator import run_stages
from wonhee.render import render_run, render_shotlist
from tests.fakes import fake_deps


def _seed(conn, weak=True):
    i = repo.create_intent(conn, "x", "scout", 1, {"mode": "discover", "niche": "x", "constraints": {}})
    p = repo.create_project(conn, "gym run", "scout", i)
    r = repo.start_run(conn, p, 5.0)
    s = repo.start_step(conn, r, "shotlist", {}, None)
    acc = repo.upsert_account(conn, "instagram", "u", "u")
    c1 = repo.upsert_content(conn, platform="instagram", external_id="V1", url="https://www.instagram.com/reel/V1/", account_id=acc)
    c2 = repo.upsert_content(conn, platform="instagram", external_id="V2", url="https://www.instagram.com/reel/V2/", account_id=acc)
    w = repo.upsert_web_source(conn, "https://blog.example/a", "t", "2026-10-15T00:00:00+00:00", None)
    t = repo.create_trend(conn, name="Leg day stairs", summary=None, window_start=None, window_end=None,
                          step_id=s, example_ids=[c1, c2], source_ids=[w])
    repo.set_project_trends(conn, p, t, [])
    repo.create_shotlist(conn, p, s, [
        repo.ShotIn(2000, "close-up", "lift fails", "POV", "bass drop", [c1, c2], []),
        repo.ShotIn(3500, "wide", "stairs win", None, None, [c2], [w]),
        repo.ShotIn(1000, "medium", "uncited", None, None, [], [])])
    repo.add_cost(conn, s, 0.25)
    if weak:
        repo.add_flag(conn, s, "UNCITED_CLAIM", "shot 3")
    repo.finish_step(conn, s, "weak" if weak else "ok")
    return p, r


def test_render_shotlist_shows_warnings_and_refs(conn):
    p, r = _seed(conn)
    out = render_shotlist(conn, p)
    lines = out.splitlines()
    assert "Leg day stairs" in lines[1] or "Leg day stairs" in lines[0]
    assert any(l.startswith("⚠ ") and "no supporting source" in l and "shot 3" in l for l in lines)
    assert '1. [2.0s] close-up — lift fails | text: "POV" | audio: bass drop  (refs 1,2)' in out
    assert "2. [3.5s] wide — stairs win  (refs 2,3)" in out
    assert "3. [1.0s] medium — uncited" in out and "refs" not in out.split("3. [1.0s]")[1].split("\n")[0]
    refs = out.split("References:")[1]
    assert "1. https://www.instagram.com/reel/V1/" in refs and "3. https://blog.example/a" in refs
    assert lines[-1] == f"run {r} · $0.25 · wonhee show run {r}"


def test_render_shotlist_clean_has_no_warning(conn):
    p, _ = _seed(conn, weak=False)
    assert "⚠" not in render_shotlist(conn, p)


def test_render_shotlist_uses_latest_and_its_run(conn):
    p, r1 = _seed(conn)
    r2 = repo.start_run(conn, p, 5.0)
    s2 = repo.start_step(conn, r2, "shotlist", {}, None)
    repo.create_shotlist(conn, p, s2, [repo.ShotIn(1000, "wide", "second", None, None, [], [])])
    repo.add_cost(conn, s2, 0.5)
    out = render_shotlist(conn, p)
    assert "second" in out and "lift fails" not in out and "⚠" not in out
    assert out.splitlines()[-1] == f"run {r2} · $0.50 · wonhee show run {r2}"


def test_render_shotlist_without_shotlist(conn):
    i = repo.create_intent(conn, "x", "scout", 1, {"mode": "discover", "constraints": {}})
    p = repo.create_project(conn, "x", "scout", i)
    out = render_shotlist(conn, p)
    assert "\n" not in out and "No shot list" in out


def test_render_run_shows_error_path(conn, settings):
    def boom(ctx):
        raise ValueError("kaboom")
    i = repo.create_intent(conn, "x", "scout", 1, {"mode": "discover", "niche": "x", "constraints": {}})
    repo.confirm_intent(conn, i)
    p = repo.create_project(conn, "x", "scout", i)
    r = repo.start_run(conn, p, 5.0)
    st = [Stage("a", (), lambda c: "ok"), Stage("b", ("a",), boom)]
    assert run_stages(conn, settings, fake_deps(), r, st, today=date(2026, 10, 15)) == "failed"
    out = render_run(conn, r, settings.media_dir)
    (err,) = conn.execute("SELECT blob_sha256 FROM step_artifact WHERE role='error'").fetchone()
    assert "failed" in out and str(path_for(settings.media_dir, err)) in out
    assert "EXCEPTION" in out and path_for(settings.media_dir, err).exists()
    assert out.count("\n") >= 3


def test_render_run_lists_superseded_attempts(conn, settings):
    i = repo.create_intent(conn, "x", "scout", 1, {"mode": "discover", "niche": "x", "constraints": {}})
    repo.confirm_intent(conn, i)
    p = repo.create_project(conn, "x", "scout", i)
    r = repo.start_run(conn, p, 5.0)
    st = [Stage("a", (), lambda c: "ok")]
    run_stages(conn, settings, fake_deps(), r, st, today=date(2026, 10, 15))
    run_stages(conn, settings, fake_deps(), r, st, today=date(2026, 10, 15))
    lines = [l for l in render_run(conn, r, settings.media_dir).splitlines() if l.startswith("a ")]
    assert len(lines) == 2
    assert "(superseded)" in lines[0] and "(superseded)" not in lines[1]
