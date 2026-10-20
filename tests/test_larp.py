import json

from wonhee import localllm, service
from tests.fakes import fake_deps

REPLY = {"framings": [{"kind": "linkedin_announcement", "overlay": "thrilled to share I drank a matcha 🍵",
                       "caption": "grateful for the team (me)."},
                      {"kind": "earnings_call", "overlay": "matcha consumption up 100% QoQ",
                       "caption": "we beat guidance."}],
         "best": 1, "hashtags": ["#founder", "#matcha", "#buildinpublic"], "text_preset": "band", "text_position": "top",
         "look": "warm_pink", "shot": "laptop with nvtop next to an iced matcha, top-down"}


def test_larp_lists_framings_marks_best(conn, settings, fake_llm):
    fake_llm.queue(LarpReply=[REPLY])
    text = service.larp(conn, settings, fake_deps(), "drank an iced matcha next to nvtop")
    assert "*1 [earnings_call] matcha consumption up 100% QoQ" in text
    assert "render: preset=band position=top look=warm_pink" in text and "#founder" in text
    assert "drank an iced matcha" in fake_llm.calls()[0]["prompt"]


def test_larp_renders_the_picked_line_on_an_image(conn, settings, fake_llm, tiny_jpg):
    fake_llm.queue(LarpReply=[REPLY])
    text = service.larp(conn, settings, fake_deps(), "drank a matcha", image=str(tiny_jpg), pick=0)
    assert "text_image.image image" in text
    step = conn.execute("SELECT params_json FROM step WHERE stage='text_image'").fetchone()
    assert step is not None


def test_larp_local_engine_costs_nothing(conn, settings, fake_llm, monkeypatch):
    monkeypatch.setattr(localllm, "_post", lambda s, body: {"message": {"content": json.dumps(REPLY)}})
    text = service.larp(conn, settings, fake_deps(), "made toast", engine="local")
    assert "matcha consumption" in text and fake_llm.calls() == []


# --- lecture register (spec 2026-10-20-reel-styles-design §7) ---
import pytest  # noqa: E402

from wonhee.nodes.larp import LectureRejected, LectureReply, lecture_check  # noqa: E402

LECTURE = {"title": "On the Sufficiency of Iced Matcha", "subtitle": "a preliminary note on a cup",
           "abstract": "We argue that one iced matcha, consumed beside nvtop, constitutes a complete afternoon.",
           "theses": ["I. The cup was cold.", "The nvtop window was open.", "Therefore the afternoon was complete."],
           "footnotes": ["The matcha was 6 dollars.", "See also: the straw."], "citation": "(Muk, 2026)",
           "overlay": "“the cup was cold; therefore, afternoon”", "caption": "a short note.",
           "hashtags": ["#matcha", "#thesis", "#deadpan"], "text_preset": "script", "text_position": "upper",
           "look": None, "shot": "matcha on a desk, top-down"}


def _run(conn, settings, fake_llm, replies, **kw):
    fake_llm.queue(LectureReply=replies)
    service.larp(conn, settings, fake_deps(), "drank a matcha", registers=["lecture"], **kw)
    from wonhee.db import queries
    from wonhee.media import path_for
    rid = conn.execute("SELECT max(id) FROM run").fetchone()[0]
    step = queries.current_step(conn, rid, "larp")
    return json.loads(path_for(settings.media_dir, queries.step_outputs(conn, step["id"])["lines"]["ref"]).read_text())


def test_lecture_fills_slots_and_strips_numbering(conn, settings, fake_llm):
    data = _run(conn, settings, fake_llm, [LECTURE])
    lec = data["lecture"]
    assert lec["theses"][0] == "The cup was cold." and 2 <= len(lec["theses"]) <= 5
    assert lec["citation"] == "(Muk, 2026)" and len(lec["footnotes"]) == 2 and lec["title"] and lec["subtitle"]
    assert data["framings"][0]["kind"] == "lecture" and data["hashtags"]


def test_existing_registers_output_has_no_lecture_key(conn, settings, fake_llm):
    fake_llm.queue(LarpReply=[REPLY])
    service.larp(conn, settings, fake_deps(), "drank a matcha")
    assert "lecture" not in fake_llm.calls()[0]["prompt"].split("from:")[-1]


@pytest.mark.parametrize("field,value", [
    ("citation", "(Smith et al., 2019)"), ("citation", "Journal of Nature 2020"),
    ("subtitle", "a study at Harvard University"), ("abstract", "published in the Journal of Matcha, doi 10.1000/xyz"),
    ("caption", "Dr. Chen proved it"), ("footnotes", ["Smith (2019) disagrees."]), ("footnotes", ["See Lee, 2021."])])
def test_lecture_guard_rejects_real_things(field, value):
    with pytest.raises(LectureRejected):
        lecture_check(LectureReply.model_validate({**LECTURE, field: value}))


def test_lecture_rejection_retries_once_with_message(conn, settings, fake_llm):
    bad = {**LECTURE, "subtitle": "at Stanford University"}
    data = _run(conn, settings, fake_llm, [bad, LECTURE])
    assert data["lecture"]["subtitle"] == LECTURE["subtitle"]
    calls = fake_llm.calls()
    assert len(calls) == 2 and "rejected" in calls[1]["prompt"] and "Stanford" in calls[1]["prompt"]


def test_lecture_second_rejection_fails_the_step(conn, settings, fake_llm):
    bad = {**LECTURE, "citation": "(Smith, 2019)"}
    fake_llm.queue(LectureReply=[bad, bad])
    text = service.larp(conn, settings, fake_deps(), "drank a matcha", registers=["lecture"])
    assert "task:" not in text and len(fake_llm.calls()) == 2


def test_lecture_alongside_other_registers(conn, settings, fake_llm):
    fake_llm.queue(LarpReply=[REPLY], LectureReply=[LECTURE])
    text = service.larp(conn, settings, fake_deps(), "drank a matcha", registers=["earnings_call", "lecture"])
    assert "[earnings_call]" in text and "[lecture]" in text
