import subprocess
from datetime import date
from pathlib import Path

import pytest

from tests.fakes import fake_deps
from wonhee.db import queries, repo
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.scout.stages import analyze


def _ctx(conn, settings, tiny_video, n_videos=3):
    i = repo.create_intent(conn, "x", "scout", 1, {"mode": "discover", "niche": "x", "constraints": {}})
    p = repo.create_project(conn, "x", "scout", i)
    r = repo.start_run(conn, p, 5.0)
    s = repo.start_step(conn, r, "analyze", {}, None)
    acc = repo.upsert_account(conn, "instagram", "u", "u")
    sha = repo.save_file(conn, settings.media_dir, tiny_video, "video", "video/mp4")
    ids = []
    for k in range(n_videos):
        c = repo.upsert_content(conn, platform="instagram", external_id=f"V{k}", url=f"https://www.instagram.com/reel/V{k}/", account_id=acc)
        repo.set_content_media(conn, c, sha)  # same file reused: analysis must happen once
        ids.append(c)
    t = repo.create_trend(conn, name="t", summary=None, window_start=None, window_end=None, step_id=s, example_ids=ids, source_ids=[])
    repo.set_project_trends(conn, p, t, [])
    from wonhee.intent.creator import Creator
    from wonhee.intent.schemas import ScoutRequest
    return Ctx(conn, settings, fake_deps(), r, p, ScoutRequest(), Creator(), s, {}, date(2026, 10, 15))


def test_analyze_once_and_cached(conn, settings, fake_llm, tiny_video):
    fake_llm.queue(ShotTexts=[{"shots": [{"ord": 1, "on_screen_text": "POV", "visual": "red"},
                                         {"ord": 2, "on_screen_text": None, "visual": "blue"},
                                         {"ord": 3, "on_screen_text": None, "visual": "green"}]}])
    ctx = _ctx(conn, settings, tiny_video)
    assert analyze.run(ctx) == "ok"
    row = conn.execute("SELECT shot_count, first_word_ms FROM v_media_style").fetchone()
    assert row["shot_count"] == 3 and row["first_word_ms"] == 400
    assert conn.execute("SELECT on_screen_text FROM media_shot WHERE ord=1").fetchone()[0] == "POV"
    assert conn.execute("SELECT count(*) FROM media_phash").fetchone()[0] == 3
    assert len(fake_llm.calls()) == 1


def test_analyze_skips_cached(conn, settings, fake_llm, tiny_video):
    fake_llm.queue(ShotTexts=[{"shots": []}])
    ctx = _ctx(conn, settings, tiny_video)
    analyze.run(ctx)
    analyze.run(ctx)  # second run: no new LLM call queued; must not call it or hit PK errors
    assert len(fake_llm.calls()) == 1


def test_analyze_weak_below_three(conn, settings, fake_llm, tiny_video):
    fake_llm.queue(ShotTexts=[{"shots": []}])
    ctx = _ctx(conn, settings, tiny_video, n_videos=2)
    assert analyze.run(ctx) == "weak"
    assert conn.execute("SELECT count(*) FROM step_flag WHERE code='LOW_SAMPLE'").fetchone()[0] == 1


def test_analyze_no_videos(conn, settings, fake_llm, tiny_video):
    ctx = _ctx(conn, settings, tiny_video, n_videos=0)
    with pytest.raises(StageFailed) as ei:
        analyze.run(ctx)
    assert ei.value.code == "LOW_SAMPLE"


def test_ignores_unsampled_ords_and_cuda_error_propagates(conn, settings, fake_llm, tiny_video):
    fake_llm.queue(ShotTexts=[{"shots": [{"ord": 99, "on_screen_text": "X", "visual": "v"}]}])
    ctx = _ctx(conn, settings, tiny_video)
    analyze.run(ctx)
    assert conn.execute("SELECT count(*) FROM media_shot WHERE visual='v'").fetchone()[0] == 0

    class Boom:
        def transcribe(self, path):
            raise RuntimeError("no CUDA; set WONHEE_WHISPER_DEVICE=cpu")
    ctx2 = _ctx(conn, settings, tiny_video)
    conn.execute("DELETE FROM media_shot"); conn.execute("DELETE FROM media_analysis")
    ctx2.deps.transcriber = Boom()
    with pytest.raises(RuntimeError, match="CUDA"):
        analyze.run(ctx2)


def test_sample_spread():
    idx = analyze._sample(20)
    assert len(idx) == len(set(idx)) <= 8 and idx[0] == 0 and idx[-1] == 19
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    assert max(gaps) - min(gaps) <= 1
    assert analyze._sample(3) == [0, 1, 2]


def test_stored_artifacts(conn, settings, fake_llm, tiny_video):
    fake_llm.queue(ShotTexts=[{"shots": []}])
    ctx = _ctx(conn, settings, tiny_video)
    analyze.run(ctx)
    imgs = fake_llm.calls()[0]["images"]
    assert len(imgs) == 3
    assert all(Path(i).is_relative_to(settings.media_dir) and Path(i).exists() for i in imgs)
    assert conn.execute("SELECT count(*) FROM media WHERE kind='transcript'").fetchone()[0] == 1
    mids = [(a + b) // 2 for a, b in conn.execute("SELECT start_ms, end_ms FROM media_shot ORDER BY ord")]
    assert sorted(r[0] for r in conn.execute("SELECT frame_ms FROM media_phash")) == sorted(mids)


def test_frame_failure_flagged(conn, settings, fake_llm, tiny_video, monkeypatch):
    fake_llm.queue(ShotTexts=[{"shots": []}])
    real = analyze.extract_frame
    calls = []

    def flaky(path, at_ms, dest):
        calls.append(at_ms)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, "ffmpeg")
        return real(path, at_ms, dest)

    monkeypatch.setattr(analyze, "extract_frame", flaky)
    ctx = _ctx(conn, settings, tiny_video)
    assert analyze.run(ctx) == "ok"
    assert conn.execute("SELECT count(*) FROM step_flag WHERE code='FRAME_FAILED'").fetchone()[0] == 1
    assert conn.execute("SELECT keyframe_sha256 FROM media_shot WHERE ord=2").fetchone()[0] is None
    assert conn.execute("SELECT count(*) FROM media_shot WHERE keyframe_sha256 IS NOT NULL").fetchone()[0] == 2


def test_one_bad_video_is_flagged_and_others_continue(conn, settings, fake_llm, tiny_video, monkeypatch):
    fake_llm.queue(ShotTexts=[{"shots": []}] * 3)
    ctx = _ctx(conn, settings, tiny_video, n_videos=3)
    # distinct media per video so each is analysed separately
    for k, row in enumerate(conn.execute("SELECT id FROM content_item ORDER BY id").fetchall()):
        sha = repo.save_media_bytes(conn, settings.media_dir, tiny_video.read_bytes() + bytes([k]), "video", "video/mp4")
        repo.set_content_media(conn, row[0], sha)
    real = analyze.analyze_video
    calls = []

    def flaky(c, sha):
        calls.append(sha)
        if len(calls) == 2:
            raise ValueError("ffmpeg exploded")
        return real(c, sha)
    monkeypatch.setattr(analyze, "analyze_video", flaky)
    assert analyze.run(ctx) == "weak"  # 2 analysed < 3
    flags = {f["code"]: f["detail"] for f in queries.step_flags(conn, ctx.step_id)}
    assert flags["ANALYZE_FAILED"] == f"{calls[1][:12]}: ffmpeg exploded"
    assert len(calls) == 3
    assert conn.execute("SELECT count(*) FROM media_analysis").fetchone()[0] == 2


def test_ensure_ready_error_propagates_once(conn, settings, fake_llm, tiny_video):
    class NoGpu:
        def ensure_ready(self):
            raise RuntimeError("whisper_device=cuda but no CUDA device")

        def transcribe(self, path):
            raise AssertionError("never reached")
    ctx = _ctx(conn, settings, tiny_video)
    ctx.deps.transcriber = NoGpu()
    with pytest.raises(RuntimeError, match="CUDA"):
        analyze.run(ctx)


def test_transcript_saves_word_timings_when_given(conn, settings, fake_llm, tiny_video):
    import json
    from wonhee.analyze.speech import Segment, SpokenWord, Transcript
    from wonhee.media import path_for

    class W:
        def transcribe(self, path):
            return Transcript("en", [Segment(0, 900, "hi there", [SpokenWord(0, 400, "hi"), SpokenWord(450, 900, "there")]),
                                     Segment(1000, 1200, "plain")])
    fake_llm.queue(ShotTexts=[{"shots": []}])
    ctx = _ctx(conn, settings, tiny_video, n_videos=1)
    ctx.deps.transcriber = W()
    analyze.run(ctx)
    tsha = conn.execute("SELECT transcript_sha256 FROM media_analysis").fetchone()[0]
    segs = json.loads(path_for(settings.media_dir, tsha).read_text())["segments"]
    assert segs[0]["words"] == [{"start_ms": 0, "end_ms": 400, "text": "hi"}, {"start_ms": 450, "end_ms": 900, "text": "there"}]
    assert "words" not in segs[1]
