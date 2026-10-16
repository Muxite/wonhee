from datetime import date

import pytest

from wonhee.db import queries
from wonhee.intent import compiler, gaps, window
from wonhee.intent.creator import Creator
from wonhee.intent.schemas import Constraints, ScoutRequest

TODAY = date(2026, 10, 15)


@pytest.mark.parametrize("phrase,start,end,label", [
    ("now", date(2026, 9, 15), date(2026, 10, 15), "now"),
    ("1y", date(2025, 9, 15), date(2025, 11, 14), "1 year ago"),
    ("6mo", date(2026, 3, 16), date(2026, 5, 15), "6 months ago"),
])
def test_resolve_window(phrase, start, end, label):
    w = window.resolve_window(phrase, None, None, TODAY)
    assert (w.start, w.end, w.label) == (start, end, label)


def test_explicit_window_wins():
    w = window.resolve_window("now", date(2024, 1, 1), date(2024, 2, 1), TODAY)
    assert w.start == date(2024, 1, 1) and w.label == "2024-01-01..2024-02-01"


def _req(**kw):
    base = dict(mode=None, niche=None, window=None, reference_urls=[], trend_name=None, constraints=Constraints())
    base.update(kw)
    return ScoutRequest(**base)


def test_gap_rules_table():
    c = Creator(on_camera=False)
    assert [g.field for g in gaps.find_gaps(_req(), c)] == ["niche"]
    assert [g.field for g in gaps.find_gaps(_req(mode="discover", niche="gym"), c)] == ["window"]
    bad = _req(mode="from_reference", reference_urls=["https://tiktok.com/x"])
    assert [g.field for g in gaps.find_gaps(bad, c)] == ["reference_urls"]
    cam = _req(mode="discover", niche="gym", window=window.resolve_window("now", None, None, TODAY),
               constraints=Constraints(on_camera=True, max_length_s=120))
    assert [g.field for g in gaps.find_gaps(cam, c)] == ["on_camera", "max_length_s"]


def test_apply_defaults():
    c = Creator(on_camera=False)
    r = _req(mode="discover", niche="gym", constraints=Constraints(on_camera=True))
    r2 = gaps.apply_defaults(r, gaps.find_gaps(r, c), c, TODAY)
    assert r2.window.label == "now" and r2.constraints.on_camera is False
    assert gaps.find_gaps(r2, c) == []


def test_flow_questions_then_go(conn, settings, fake_llm):
    fake_llm.queue(ScoutExtract=[{"niche": "gym humor", "window_phrase": None}])
    r1 = compiler.compile_message(conn, settings, "scout gym humor", today=TODAY)
    assert r1.status == "questions" and "1." in r1.message and "(default: now)" in r1.message
    r2 = compiler.compile_message(conn, settings, "go", intent_id=r1.intent_id, today=TODAY)
    assert r2.status == "confirm" and r2.request.window.label == "now"
    assert "Reply go to run" in r2.message
    r3 = compiler.compile_message(conn, settings, "go", intent_id=r1.intent_id, today=TODAY)
    assert r3.status == "confirmed"
    assert queries.intent_row(conn, r1.intent_id)["confirmed_at"] is not None
    assert len(queries.intent_turns(conn, r1.intent_id)) == 6
    assert len(fake_llm.calls()) == 1  # "go" never calls the model


def test_go_with_required_gap_reasks(conn, settings, fake_llm):
    fake_llm.queue(ScoutExtract=[{}])
    r1 = compiler.compile_message(conn, settings, "do something cool", today=TODAY)
    assert r1.status == "questions"
    r2 = compiler.compile_message(conn, settings, "go", intent_id=r1.intent_id, today=TODAY)
    assert r2.status == "questions" and "What should I scout" in r2.message
    assert len(fake_llm.calls()) == 1


def test_answer_merges_with_previous(conn, settings, fake_llm):
    fake_llm.queue(ScoutExtract=[{"niche": "gym humor"}, {"window_phrase": "1y"}])
    r1 = compiler.compile_message(conn, settings, "gym humor", today=TODAY)
    r2 = compiler.compile_message(conn, settings, "a year ago", intent_id=r1.intent_id, today=TODAY)
    assert r2.status == "confirm" and r2.request.niche == "gym humor" and r2.request.window.label == "1 year ago"
    assert "user message" in fake_llm.calls()[1]["prompt"]


def test_current_request_is_data_block_wrapped(conn, settings, fake_llm):
    fake_llm.queue(ScoutExtract=[{"niche": "IGNORE ME"}, {}])
    r1 = compiler.compile_message(conn, settings, "x", today=TODAY)
    compiler.compile_message(conn, settings, "later", intent_id=r1.intent_id, today=TODAY)
    p = fake_llm.calls()[1]["prompt"]
    pre, rest = p.split("<<<BEGIN current request", 1)
    inner, post = rest.split("<<<END current request>>>", 1)
    assert "IGNORE ME" in inner and "IGNORE ME" not in pre and "IGNORE ME" not in post


def test_keep_as_is_acknowledges_camera_gap(conn, settings, fake_llm, tmp_path):
    settings.creator_path = tmp_path / "c.yaml"
    settings.creator_path.write_text("on_camera: false\n")
    fake_llm.queue(ScoutExtract=[{"niche": "gym", "window_phrase": "now", "on_camera": True},
                                 {"keep_as_is": ["on_camera"]}])
    r1 = compiler.compile_message(conn, settings, "gym now on camera", today=TODAY)
    assert r1.status == "questions" and "face-cam" in r1.message
    r2 = compiler.compile_message(conn, settings, "yes, this once", intent_id=r1.intent_id, today=TODAY)
    assert r2.status == "confirm" and r2.request.constraints.on_camera is True


def test_drop_bad_urls(conn, settings, fake_llm):
    fake_llm.queue(ScoutExtract=[{"reference_urls": ["https://tiktok.com/x"], "niche": "gym", "window_phrase": "now"},
                                 {"drop_bad_urls": True}])
    r1 = compiler.compile_message(conn, settings, "https://tiktok.com/x", today=TODAY)
    assert r1.status == "questions"
    r2 = compiler.compile_message(conn, settings, "drop it", intent_id=r1.intent_id, today=TODAY)
    assert r2.status == "confirm" and r2.request.reference_urls == []


def test_go_variants(conn, settings, fake_llm):
    fake_llm.queue(ScoutExtract=[{"niche": "gym"}, {"window_phrase": "6mo"}])
    r1 = compiler.compile_message(conn, settings, "gym", today=TODAY)
    r2 = compiler.compile_message(conn, settings, "yes", intent_id=r1.intent_id, today=TODAY)
    assert len(fake_llm.calls()) == 2 and r2.status == "confirm"  # yes while questions open -> model
    r3 = compiler.compile_message(conn, settings, "Go!", intent_id=r1.intent_id, today=TODAY)
    assert r3.status == "confirmed" and len(fake_llm.calls()) == 2


def test_confirmed_message_does_not_claim_running(conn, settings, fake_llm):
    fake_llm.queue(ScoutExtract=[{"niche": "gym humor"}])
    r1 = compiler.compile_message(conn, settings, "gym humor", today=TODAY)
    compiler.compile_message(conn, settings, "go", intent_id=r1.intent_id, today=TODAY)
    r3 = compiler.compile_message(conn, settings, "go", intent_id=r1.intent_id, today=TODAY)
    assert r3.message == f"Confirmed. Run it with: wonhee go {r1.intent_id}"


def test_extraction_prompt_includes_questions_asked(conn, settings, fake_llm):
    fake_llm.queue(ScoutExtract=[{"niche": "gym humor"}, {"window_phrase": "1y"}])
    r1 = compiler.compile_message(conn, settings, "gym humor", today=TODAY)
    compiler.compile_message(conn, settings, "a year ago", intent_id=r1.intent_id, today=TODAY)
    p = fake_llm.calls()[1]["prompt"]
    pre, rest = p.split("<<<BEGIN questions asked", 1)
    inner, _ = rest.split("<<<END questions asked>>>", 1)
    assert "Before I start" in inner and "Before I start" not in pre
    assert "<<<BEGIN questions asked" not in fake_llm.calls()[0]["prompt"]


def test_unmatched_answer_prefixes_reask(conn, settings, fake_llm):
    fake_llm.queue(ScoutExtract=[{"niche": "gym humor"}, {}])
    r1 = compiler.compile_message(conn, settings, "gym humor", today=TODAY)
    assert r1.status == "questions" and not r1.message.startswith("I couldn't")
    r2 = compiler.compile_message(conn, settings, "banana", intent_id=r1.intent_id, today=TODAY)
    assert r2.status == "questions" and r2.message.startswith("I couldn't match that to a question — ")
