from datetime import date

import pytest

from tests.fakes import FakeApify, FakeDownloader, fake_deps, make_item
from wonhee.db import queries, repo
from wonhee.db.repo import ShotAnalysis
from wonhee.intent.creator import Creator
from wonhee.intent.schemas import ScoutRequest
from wonhee.pipeline.context import Ctx, Stage, StageFailed
from wonhee.pipeline.orchestrator import run_stages
from wonhee.pipeline.registry import SCOUT_STAGES
from wonhee.render import render_shotlist
from wonhee.scout.stages import describe, shotlist
from wonhee.sources.apify import ACTOR_HASHTAG, ACTOR_SCRAPER

TODAY = date(2026, 10, 15)


def _ctx(conn, settings, stage="shotlist", request=None, creator=None, n=2, sources=1):
    """Project + run + step, a primary trend with n analysed examples and `sources` web sources."""
    i = repo.create_intent(conn, "x", "scout", 1, {"mode": "discover", "niche": "x", "constraints": {}})
    p = repo.create_project(conn, "x", "scout", i)
    r = repo.start_run(conn, p, 5.0)
    s = repo.start_step(conn, r, stage, {}, None)
    acc = repo.upsert_account(conn, "instagram", "u", "u")
    ids = []
    for k in range(n):
        c = repo.upsert_content(conn, platform="instagram", external_id=f"V{k}",
                                url=f"https://www.instagram.com/reel/V{k}/", account_id=acc)
        sha = repo.save_media_bytes(conn, settings.media_dir, f"video{k}-{p}".encode(), "video", "video/mp4")
        repo.set_content_media(conn, c, sha)
        tsha = repo.save_media_bytes(
            conn, settings.media_dir, b'{"language":"en","segments":[{"start_ms":0,"end_ms":900,"text":"hello gym"}]}',
            "transcript", "application/json")
        repo.save_analysis(conn, sha=sha, analyzer_version=1, step_id=s, first_word_ms=0, language="en",
                           transcript_sha=tsha, shots=[ShotAnalysis(0, 1000, None, "POV: leg day", "a man squats")])
        ids.append(c)
    srcs = [repo.upsert_web_source(conn, f"https://blog.example/{k}", "t", "2026-10-15T00:00:00+00:00", None)
            for k in range(sources)]
    t = repo.create_trend(conn, name="Leg day stairs", summary="people fail stairs", window_start=None,
                          window_end=None, step_id=s, example_ids=ids, source_ids=srcs)
    repo.set_project_trends(conn, p, t, [])
    ctx = Ctx(conn, settings, fake_deps(), r, p, request or ScoutRequest(), creator or Creator(), s, {}, TODAY)
    return ctx, ids, srcs, t


def _shot(d=5, content=(), sources=(), text=None):
    return {"duration_s": d, "framing": "close-up", "action": "do it", "on_screen_text": text,
            "audio_cue": None, "cite_content_ids": list(content), "cite_source_ids": list(sources)}


def _flags(conn, step):
    return {f["code"]: f["detail"] for f in queries.step_flags(conn, step)}


def test_shotlist_drops_foreign_citations_and_flags(conn, settings, fake_llm):
    ctx, ids, srcs, _ = _ctx(conn, settings)
    fake_llm.queue(ShotPlan=[{"shots": [_shot(content=[ids[0], 77], sources=[srcs[0], 555]), _shot()]}])
    assert shotlist.run(ctx) == "weak"
    sl = queries.latest_shotlist(conn, ctx.project_id)
    cites = queries.citations(conn, sl["id"])
    assert [(c["shot_ord"], c["content_id"], c["web_source_id"]) for c in cites] == [
        (1, ids[0], None), (1, None, srcs[0])]
    assert _flags(conn, ctx.step_id) == {"UNCITED_CLAIM": "shot 2"}
    assert [s["duration_ms"] for s in queries.shots(conn, sl["id"])] == [5000, 5000]


def test_ok_when_all_cited_and_prompt_has_constraints(conn, settings, fake_llm):
    ctx, ids, srcs, _ = _ctx(conn, settings, request=ScoutRequest.model_validate({"constraints": {"max_length_s": 12, "on_camera": False}}))
    fake_llm.queue(ShotPlan=[{"shots": [_shot(content=[ids[1]])]}])
    assert shotlist.run(ctx) == "ok"
    prompt = fake_llm.calls()[0]["prompt"]
    assert "max length 12s" in prompt and "off-camera" in prompt
    assert f"Allowed cite_content_ids: {sorted(ids)}" in prompt and f"Allowed cite_source_ids: {srcs}" in prompt


def test_length_over_flag(conn, settings, fake_llm):
    ctx, ids, _, _ = _ctx(conn, settings, creator=Creator(max_length_s=30))
    fake_llm.queue(ShotPlan=[{"shots": [_shot(15, [ids[0]]) for _ in range(3)]}])
    assert shotlist.run(ctx) == "weak"
    assert list(_flags(conn, ctx.step_id)) == ["LENGTH_OVER"]


def test_request_length_overrides_creator_and_equal_is_fine(conn, settings, fake_llm):
    req = ScoutRequest.model_validate({"constraints": {"max_length_s": 10}})
    ctx, ids, _, _ = _ctx(conn, settings, request=req, creator=Creator(max_length_s=60))
    fake_llm.queue(ShotPlan=[{"shots": [_shot(5, [ids[0]]), _shot(5, [ids[0]])]}])
    assert shotlist.run(ctx) == "ok"
    ctx2, ids2, _, _ = _ctx(conn, settings, request=req, creator=Creator(max_length_s=60))
    fake_llm.queue(ShotPlan=[{"shots": [_shot(5.5, [ids2[0]]), _shot(5, [ids2[0]])]}])
    assert shotlist.run(ctx2) == "weak" and "LENGTH_OVER" in _flags(conn, ctx2.step_id)


def test_shotlist_sources_are_only_the_primary_trends(conn, settings, fake_llm):
    ctx, ids, srcs, _ = _ctx(conn, settings)
    other = repo.upsert_web_source(conn, "https://other.example/x", "o", "2026-10-15T00:00:00+00:00", None)
    fake_llm.queue(ShotPlan=[{"shots": [_shot(sources=[other])]}])
    assert shotlist.run(ctx) == "weak"
    assert conn.execute("SELECT count(*) FROM shot_citation").fetchone()[0] == 0


def test_empty_plan_fails(conn, settings, fake_llm):
    ctx, *_ = _ctx(conn, settings)
    fake_llm.queue(ShotPlan=[{"shots": []}])
    with pytest.raises(StageFailed) as ei:
        shotlist.run(ctx)
    assert ei.value.code == "SCHEMA_FAIL"


def _card(beats, inv=("keeps stairs",), var=("different gyms",)):
    return {"summary": "new summary", "beats": beats, "invariants": list(inv), "variations": list(var)}


def test_describe_drops_unknown_evidence(conn, settings, fake_llm):
    ctx, ids, _, t = _ctx(conn, settings, stage="describe")
    fake_llm.queue(TrendCard=[_card([
        {"role": "hook", "description": "lift fails", "evidence_content_ids": [999]},
        {"role": "payoff", "description": "stairs win", "evidence_content_ids": [ids[0], 999, ids[0]]}])])
    assert describe.run(ctx) == "weak"
    assert _flags(conn, ctx.step_id) == {"UNCITED_CLAIM": "hook"}
    assert [(b["role"], b["description"]) for b in queries.trend_beats(conn, t)] == [
        ("hook", "lift fails"), ("payoff", "stairs win")]
    assert [(r["kind"], r["text"]) for r in queries.trend_rules(conn, t)] == [
        ("invariant", "keeps stairs"), ("variation", "different gyms")]
    assert queries.primary_trend(conn, ctx.project_id)["summary"] == "new summary"


def test_describe_ok_and_prompt_uses_data_blocks(conn, settings, fake_llm):
    ctx, ids, _, _ = _ctx(conn, settings, stage="describe")
    fake_llm.queue(TrendCard=[_card([{"role": "hook", "description": "d", "evidence_content_ids": [ids[1]]}])])
    assert describe.run(ctx) == "ok"
    prompt = fake_llm.calls()[0]["prompt"]
    assert "<<<BEGIN transcript" in prompt and "hello gym" in prompt
    assert "<<<BEGIN on-screen text" in prompt and "POV: leg day" in prompt and "a man squats" in prompt
    assert "<<<BEGIN trend name" in prompt and "Leg day stairs" in prompt
    assert all(f"content_id={c}" in prompt for c in ids)
    assert "shot_count=1" in prompt


def test_describe_without_primary_trend_fails(conn, settings, fake_llm):
    ctx, *_ = _ctx(conn, settings, stage="describe")
    conn.execute("DELETE FROM project_trend")
    with pytest.raises(StageFailed):
        describe.run(ctx)


def test_shotlist_prompt_includes_card(conn, settings, fake_llm):
    ctx, ids, _, t = _ctx(conn, settings)
    repo.set_trend_card(conn, t, "s", [("hook", "lift fails")], [("invariant", "keeps stairs")])
    fake_llm.queue(ShotPlan=[{"shots": [_shot(content=[ids[0]])]}])
    shotlist.run(ctx)
    prompt = fake_llm.calls()[0]["prompt"]
    assert "hook" in prompt and "lift fails" in prompt and "keeps stairs" in prompt


def _items():
    return [make_item(c, f"u{c}", "2026-10-01", 100 * n, audio="7") for n, c in enumerate("ABC", 1)]


def test_full_pipeline_with_fakes(conn, settings, fake_llm, tiny_video):
    fake_llm.queue(
        HashtagPlan=[{"tags": ["gym"]}],
        ClusterVerdict=[{"items": [{"key": "7", "name": "T7", "summary": "s", "is_trend": True}]}],
        ShotTexts=[{"shots": [{"ord": 1, "on_screen_text": "POV", "visual": "red"}]}] * 3,
        TrendCard=[_card([{"role": "hook", "description": "d", "evidence_content_ids": [1, 2]}])],
        ShotPlan=[{"shots": [_shot(4, [1, 2], text="POV"), _shot(6, [3])]}])
    i = repo.create_intent(conn, "x", "scout", 1, {
        "mode": "discover", "niche": "gym",
        "window": {"start": "2026-09-15", "end": "2026-10-15", "label": "now"}, "constraints": {}})
    repo.confirm_intent(conn, i)
    p = repo.create_project(conn, "gym project", "scout", i)
    r = repo.start_run(conn, p, 5.0)
    deps = fake_deps(apify=FakeApify({ACTOR_HASHTAG: [_items()], ACTOR_SCRAPER: [_items()]}),
                     downloader=FakeDownloader(tiny_video))
    assert [s.name for s in SCOUT_STAGES] == ["resolve", "discover", "pick", "fetch", "analyze", "describe", "shotlist"]
    assert run_stages(conn, settings, deps, r, SCOUT_STAGES, today=TODAY) == "done"
    text = render_shotlist(conn, p)
    assert "1. [4.0s]" in text and "References:" in text and f"wonhee show run {r}" in text
    assert "⚠" not in text


def test_describe_empty_beats_fails(conn, settings, fake_llm):
    ctx, *_ = _ctx(conn, settings, stage="describe")
    fake_llm.queue(TrendCard=[_card([])])
    with pytest.raises(StageFailed) as ei:
        describe.run(ctx)
    assert ei.value.code == "SCHEMA_FAIL" and "no beats" in ei.value.detail


def test_describe_flags_unreadable_transcript(conn, settings, fake_llm):
    ctx, ids, _, _ = _ctx(conn, settings, stage="describe")
    from wonhee.media import path_for
    for (sha,) in conn.execute("SELECT transcript_sha256 FROM media_analysis").fetchall():
        path_for(settings.media_dir, sha).unlink(missing_ok=True)
    fake_llm.queue(TrendCard=[_card([{"role": "hook", "description": "d", "evidence_content_ids": [ids[0]]}])])
    describe.run(ctx)
    assert "LOW_SAMPLE" in _flags(conn, ctx.step_id) and "transcript unreadable for" in _flags(conn, ctx.step_id)["LOW_SAMPLE"]


def test_shotlist_header_trend_is_the_cited_one(conn, settings, fake_llm):
    ctx, ids, srcs, t = _ctx(conn, settings)
    fake_llm.queue(ShotPlan=[{"shots": [_shot(content=[ids[0]])]}])
    shotlist.run(ctx)
    other = repo.create_trend(conn, name="Newer trend", summary=None, window_start=None, window_end=None,
                              step_id=ctx.step_id, example_ids=[], source_ids=[])
    repo.set_project_trends(conn, ctx.project_id, other, [])
    assert "Trend: Leg day stairs" in render_shotlist(conn, ctx.project_id)
