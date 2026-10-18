import json

import pytest

from tests.fakes import FakeApify, FakeDownloader, fake_deps, make_item, make_photo_item
from wonhee import localllm, service
from wonhee.inspire import report
from wonhee.inspire.dossier import copies_source, looks_foreign
from wonhee.sources.apify import ACTOR_SCRAPER

SHOT_TEXTS = {"shots": [{"ord": i, "on_screen_text": "POV when you're working and remember your salary" if i == 1
                         else None, "visual": "desk"} for i in (1, 2, 3)]}
IMAGE_TEXTS = {"images": [{"n": 1, "on_screen_text": "당근 프렙", "visual": "a cake shop"}]}


def teardown_reply(template="POV when you're [activity] and remember [trigger]", caps=("text_overlay",)):
    return {"summary": "a desk POV meme", "format": "pov-text-meme",
            "coupling": {"level": "decoupled", "why": "text carries the joke"},
            "hook": {"first_2s": "text on screen", "mechanism": "relatable"},
            "text_overlay": {"present": True, "english": "POV ...", "style": "outline", "position": "upper"},
            "look": "moody", "audio": {"kind": "trending-song", "role": "pace"},
            "footage_to_record": [{"subject": "you typing", "setting": "desk", "light": "window", "camera": "static",
                                   "framing": "medium", "duration_s": 7, "person_on_camera": True,
                                   "difficulty_1to5": 1},
                                  {"subject": "bus at stop", "setting": "street", "light": "dusk", "camera": "pan",
                                   "framing": "wide", "duration_s": 3, "person_on_camera": False,
                                   "difficulty_1to5": 2}],
            "edit_recipe": ["one shot", "one line"], "why_it_works": [{"claim": "relatable", "evidence": "3.8M plays"}],
            "replicate": {"score_1to5": 4, "effort_minutes": 30, "blockers": [], "who_can_make_it": "anyone with a phone"},
            "text_template": template, "learn": ["text is the product"], "capabilities": list(caps) * 2,
            "confidence": "high", "gaps": []}


@pytest.fixture
def local_calls(monkeypatch):
    calls = []

    def post(settings, body):
        calls.append(body)
        n = body["messages"][1]["content"].count("\n[")
        return {"message": {"content": json.dumps({"texts": [f"english {i}" for i in range(n)]})}}
    monkeypatch.setattr(localllm, "_post", post)
    return calls


def _store(conn, settings, fake_llm, tiny_video, tiny_jpg):
    fake_llm.queue(ShotTexts=[SHOT_TEXTS], ImageTexts=[IMAGE_TEXTS])
    items = [make_item("AAA", "brandy", "2026-07-05", 3_783_028),
             {**make_photo_item("KKK", "yujin", "2026-10-17", 297_378), "caption": "우주떡집 안녀엉 당근프렙"}]
    deps = fake_deps(apify=FakeApify({ACTOR_SCRAPER: [items]}),
                     downloader=FakeDownloader(tiny_video, image=tiny_jpg))
    service.grab(conn, settings, deps, ["https://www.instagram.com/reel/AAA/", "https://www.instagram.com/p/KKK/"])
    return deps


def test_foreign_detection_and_copy_guard():
    assert looks_foreign("우주떡집 안녕") and looks_foreign("hola", "es") and not looks_foreign("POV: deadline")
    assert copies_source("POV when you're working really hard and [x]", ["POV when you're working really hard"])
    assert copies_source("POV when you're [activity] and remember [trigger]",
                         ["POV when you're working really hard and remember your salary"]) is None


def test_local_translate_batches_one_call(settings, local_calls):
    out = localllm.translate_texts(settings, ["우주떡집", "안녕"], context="caption")
    assert out == ["english 0", "english 1"] and len(local_calls) == 1
    assert local_calls[0]["model"] == settings.local_llm_model and local_calls[0]["keep_alive"] == "30m"


def test_inspire_translates_tears_down_and_reports(conn, settings, fake_llm, tiny_video, tiny_jpg, local_calls):
    deps = _store(conn, settings, fake_llm, tiny_video, tiny_jpg)
    fake_llm.queue(Teardown=[teardown_reply(), teardown_reply(caps=("carousel_layout", "speed_ramp"))])
    text, path = service.inspire(conn, settings, deps)
    assert len(local_calls) == 1  # only the Korean post needed the on-box model
    prompts = [c["prompt"] for c in fake_llm.calls() if c["response_format"]["json_schema"]["name"] == "Teardown"]
    assert "English translations" not in prompts[0] and "english 0" in prompts[1]
    assert not any("Creator profile" in p for p in prompts)  # teardowns stay creator-blind (provenance)
    md = path.read_text()
    assert "## What to build" in md and "| speed_ramp | 1 | **missing** |" in md
    assert "| text_overlay | 1 | text_image / text_video |" in md  # capabilities deduped per post
    assert "**You on camera**" in md and "**Places and things**" in md
    assert "→ `text_video preset=outline position=upper look=moody`" in md
    assert json.loads((path.parent / "teardowns.json").read_text())[0]["format"] == "pov-text-meme"
    assert "2 teardown(s), 0 failed or not reached" in text


def test_inspire_drops_a_copied_template(conn, settings, fake_llm, tiny_video, tiny_jpg, local_calls):
    deps = _store(conn, settings, fake_llm, tiny_video, tiny_jpg)
    fake_llm.queue(Teardown=[teardown_reply("POV when you're working and remember your salary [x]"),
                             teardown_reply()])
    _, path = service.inspire(conn, settings, deps, [1])
    tds = json.loads((path.parent / "teardowns.json").read_text())
    assert tds[0]["text_template"] is None
    assert conn.execute("SELECT count(*) FROM step_flag WHERE code='STYLE_LEAK'").fetchone()[0] == 1


def test_inspire_skips_a_failed_post_and_finishes_the_rest(conn, settings, fake_llm, tiny_video, tiny_jpg,
                                                           local_calls):
    deps = _store(conn, settings, fake_llm, tiny_video, tiny_jpg)
    fake_llm.queue(Teardown=[teardown_reply()])  # the second post gets no reply -> its step fails
    text, path = service.inspire(conn, settings, deps)
    assert "1 teardown(s), 1 failed" in text and "Teardown failed for content 2" in path.read_text()


def test_report_counts_on_fixture_teardowns():
    t = {**teardown_reply(), "content_id": 7, "capabilities": ["text_overlay", "green_screen_key"]}
    md = report.build([t, {**t, "content_id": 8, "coupling": {"level": "coupled", "why": "x"}}],
                      {7: {"plays": 10}, 8: {"plays": 5}})
    assert "decoupled 1, coupled 1" in md and "| green_screen_key | 2 | **missing** |" in md
    t2 = {**t, "content_id": 9, "capabilities": ["on_camera_performance"]}
    assert "Shoot side (your job, not tooling): on_camera_performance 1" in report.build([t2], {})
    assert md.count("from 7") >= 2 and "from 8 " not in md.split("## Shoot list")[1].split("## Line bank")[0]
