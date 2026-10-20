import pytest
from wonhee.db import queries, repo


@pytest.fixture
def ctx(conn):
    i = repo.create_intent(conn, "gym humor", "scout", 1, {"niche": "gym humor"})
    p = repo.create_project(conn, "gym humor", "scout", i)
    run = repo.start_run(conn, p, 1.0)
    s = repo.start_step(conn, run, "discover", {}, None)
    a = repo.upsert_account(conn, "instagram", "u1", "alice")
    c = repo.upsert_content(conn, platform="instagram", external_id="X", url="https://instagram.com/reel/X", account_id=a)
    return {"intent": i, "project": p, "run": run, "step": s, "content": c}


def test_confirm_twice_fails(conn, ctx):
    repo.confirm_intent(conn, ctx["intent"])
    with pytest.raises(ValueError):
        repo.confirm_intent(conn, ctx["intent"])


def test_turns_ordered(conn, ctx):
    assert repo.add_intent_turn(conn, ctx["intent"], "user", "a", None) == 1
    assert repo.add_intent_turn(conn, ctx["intent"], "compiler", "b", [{"field": "window"}]) == 2


def test_set_project_trends_replaces(conn, ctx):
    t1 = repo.create_trend(conn, name="a", summary=None, window_start=None, window_end=None, step_id=ctx["step"], example_ids=[ctx["content"]], source_ids=[])
    t2 = repo.create_trend(conn, name="b", summary=None, window_start=None, window_end=None, step_id=ctx["step"], example_ids=[], source_ids=[])
    repo.set_project_trends(conn, ctx["project"], t1, [t2])
    repo.set_project_trends(conn, ctx["project"], t2, [])
    assert queries.primary_trend(conn, ctx["project"])["trend_id"] == t2
    assert conn.execute("SELECT count(*) FROM project_trend").fetchone()[0] == 1


def test_shotlist_versions_and_citations(conn, ctx):
    shots = [repo.ShotIn(duration_ms=2000, framing="close", action="look", on_screen_text="pov", audio_cue=None,
                         content_ids=[ctx["content"]], web_source_ids=[])]
    v1 = repo.create_shotlist(conn, ctx["project"], ctx["step"], shots)
    v2 = repo.create_shotlist(conn, ctx["project"], ctx["step"], shots)
    assert [r["version"] for r in conn.execute("SELECT version FROM shotlist ORDER BY id")] == [1, 2]
    assert queries.latest_shotlist(conn, ctx["project"])["id"] == v2
    assert len(queries.citations(conn, v1)) == 1


def test_trend_card_replaces(conn, ctx):
    t = repo.create_trend(conn, name="a", summary=None, window_start=None, window_end=None, step_id=ctx["step"], example_ids=[], source_ids=[])
    repo.set_trend_card(conn, t, "s1", [("hook", "x"), ("payoff", "y")], [("invariant", "z")])
    repo.set_trend_card(conn, t, "s2", [("hook", "x2")], [])
    assert conn.execute("SELECT count(*) FROM trend_beat").fetchone()[0] == 1
    assert conn.execute("SELECT summary FROM trend").fetchone()[0] == "s2"


def test_update_intent_rejects_confirmed_and_unknown(conn, ctx):
    repo.update_intent(conn, ctx["intent"], {"niche": "new"})
    repo.confirm_intent(conn, ctx["intent"])
    with pytest.raises(ValueError):
        repo.update_intent(conn, ctx["intent"], {"niche": "later"})
    with pytest.raises(ValueError):
        repo.update_intent(conn, 9999, {})


def test_trend_examples_ordering(conn, ctx, tmp_path):
    a = conn.execute("SELECT account_id FROM content_item").fetchone()[0]
    c2 = repo.upsert_content(conn, platform="instagram", external_id="Y", url="u2", account_id=a, caption="cap2")
    c3 = repo.upsert_content(conn, platform="instagram", external_id="Z", url="u3", account_id=a)
    c4 = repo.upsert_content(conn, platform="instagram", external_id="W", url="u4", account_id=a)
    sha = repo.save_media_bytes(conn, tmp_path, b"vid", "video", "video/mp4")
    repo.set_content_media(conn, c2, sha)
    s = ctx["step"]
    repo.add_snapshot(conn, ctx["content"], s, "2026-10-01T00:00:00+00:00", 10, 1, 1, 1)
    repo.add_snapshot(conn, c2, s, "2026-10-01T00:00:00+00:00", 500, 1, 1, 1)
    repo.add_snapshot(conn, c3, s, "2026-10-01T00:00:00+00:00", None, 1, 1, 1)
    t = repo.create_trend(conn, name="t", summary=None, window_start=None, window_end=None, step_id=s,
                          example_ids=[ctx["content"], c2, c3, c4], source_ids=[])
    rows = queries.trend_examples(conn, t)
    assert [r["content_id"] for r in rows] == [c2, ctx["content"], c3, c4]
    assert rows[0]["media_sha256"] == sha and rows[0]["caption"] == "cap2"


def test_analysis_roundtrip(conn, ctx, tmp_path):
    sha = repo.save_media_bytes(conn, tmp_path, b"vid", "video", "video/mp4")
    kf = repo.save_media_bytes(conn, tmp_path, b"img", "image", "image/png")
    repo.save_analysis(conn, sha=sha, analyzer_version=2, step_id=ctx["step"], first_word_ms=300, language="en",
                       transcript_sha=None,
                       shots=[repo.ShotAnalysis(0, 1000, kf), repo.ShotAnalysis(1000, 2500, None, "hi", "gym")])
    assert queries.has_analysis(conn, sha, 2) is True
    assert queries.has_analysis(conn, sha, 3) is False
    rows = conn.execute("SELECT ord, start_ms, end_ms FROM media_shot ORDER BY ord").fetchall()
    assert [(r["ord"], r["start_ms"], r["end_ms"]) for r in rows] == [(1, 0, 1000), (2, 1000, 2500)]


def test_trends_of_step_scoped(conn, ctx):
    run2 = repo.start_run(conn, ctx["project"], 1.0)
    s2 = repo.start_step(conn, run2, "discover", {}, None)
    s = ctx["step"]
    repo.add_snapshot(conn, ctx["content"], s, "2026-10-01T00:00:00+00:00", 40, 1, 1, 1)
    t1 = repo.create_trend(conn, name="a", summary=None, window_start=None, window_end=None, step_id=s, example_ids=[ctx["content"]], source_ids=[])
    repo.create_trend(conn, name="b", summary=None, window_start=None, window_end=None, step_id=s2, example_ids=[], source_ids=[])
    rows = queries.trends_of_step(conn, s)
    assert [r["trend_id"] for r in rows] == [t1]
    assert rows[0]["example_count"] == 1 and rows[0]["median_plays"] == 40


def test_intent_row_unknown_raises(conn):
    with pytest.raises(KeyError):
        queries.intent_row(conn, 12345)


def test_save_image_analyses(conn, ctx, tmp_path):
    """design: DESIGN contract 6 (writes go through wonhee/db/repo/)."""
    a = repo.save_media_bytes(conn, tmp_path, b"i1", "image", "image/png")
    b = repo.save_media_bytes(conn, tmp_path, b"i2", "image", "image/png")
    repo.save_image_analyses(conn, analyzer_version=1, step_id=ctx["step"], rows=[(a, "hello", "a gym"), (b, None, None)])
    got = conn.execute("SELECT media_sha256, on_screen_text, visual FROM image_analysis ORDER BY media_sha256").fetchall()
    assert {(r[0], r[1], r[2]) for r in got} == {(a, "hello", "a gym"), (b, None, None)}
