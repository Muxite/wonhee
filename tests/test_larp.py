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
