import json

from tests.test_inspire import _store, local_calls, teardown_reply  # noqa: F401 (fixture)
from wonhee import service
from wonhee.inspire import brief as bf
from wonhee.inspire import ideas as ix

BRIEF = """# My ideas
intro paragraph about the niche

## Founder LARP
* call the 5070 Ti + Jetson a private inference cluster
- post "still shipping" at 3:17 AM
1. post "still shipping" at 3:17 AM
"""


def plate(subject, person=False):
    return {"subject": subject, "setting": "desk", "camera": "static", "framing": "wide", "light": "lamp",
            "duration_s": 6, "person_on_camera": person}


def idea(title, insp=(), brief=(), plates=(0,), line="a line"):
    return {"title": title, "format": "text-on-broll", "inspiration": list(insp), "brief": list(brief),
            "how": "takes the format", "line": line, "caption": "cap", "plates": list(plates),
            "text_preset": "outline", "text_position": "upper", "look": "moody", "effort_minutes": 20, "cost_usd": 0}


def test_brief_items_have_stable_ids_and_dedupe():
    items = bf.parse(BRIEF, "b1")
    assert [i.text for i in items] == ["intro paragraph about the niche",
                                       "call the 5070 Ti + Jetson a private inference cluster",
                                       'post "still shipping" at 3:17 AM']  # the repeated bullet is one item
    assert items[1].section == "Founder LARP" and items[1].id == bf.item_id("CALL the 5070 Ti  + Jetson a private "
                                                                            "inference cluster")
    assert bf.render(items).count("[b1 / ") == 2


def test_origin_is_computed_from_citations():
    s = ix.IdeaSet.model_validate({"ideas": [idea("a", insp=["c1"]), idea("b", brief=["b-00000"]),
                                             idea("c", insp=["c1"], brief=["b-00000"]), idea("d")],
                                   "shoot_list": [plate("desk")]})
    assert [ix.origin_of(i) for i in s.ideas] == ["inspiration", "own", "blend", "new"]
    notes = ix.clean(s, post_ids={2}, brief_ids={"b-00000"})  # c1 is not an input post
    assert [ix.origin_of(i) for i in s.ideas] == ["new", "own", "own", "new"] and len(notes) == 2


def test_clipped_brief_id_is_repaired_only_when_unambiguous():
    assert ix._brief_ref("b-94c4", {"b-94c4b", "b-11111"}) == "b-94c4b"
    assert ix._brief_ref("b-94c4", {"b-94c4b", "b-94c4a"}) is None
    assert ix._brief_ref("b-9", {"b-94c4b"}) is None  # too short to trust
    assert ix.split(24) == {"blends": 10, "from_inspiration": 7, "from_brief": 7}


def test_ideate_keeps_sources_apart_and_checks_citations(conn, settings, fake_llm, tiny_video, tiny_jpg, local_calls):
    deps = _store(conn, settings, fake_llm, tiny_video, tiny_jpg)
    (settings.home / "brief").mkdir(parents=True)
    (settings.home / "brief" / "2026-10-17-mine.md").write_text(BRIEF)
    bid = bf.item_id("call the 5070 Ti + Jetson a private inference cluster")
    c1, c2 = [r[0] for r in conn.execute("SELECT id FROM content_item ORDER BY id")]
    overlay = "POV when you're working really hard and remember your salary"
    td = teardown_reply()
    td["text_overlay"]["english"] = overlay
    fake_llm.queue(Teardown=[td, teardown_reply()], IdeaReply=[{
        "blends": [idea("blend", insp=[f"c{c1}"], brief=[bid[:-1]], plates=[0, 1])],  # clipped id is repaired
        "from_inspiration": [idea("insp", insp=[f"c{c2}"], plates=[0], line='"quoted line"')],
        "from_brief": [idea("own", brief=[bid], plates=[1, 7])],
        "new": [idea("made up refs", insp=["c999"], brief=["b-fffff"]),
                idea("copier", insp=[f"c{c1}"], line="me working really hard and remember your salary lol")],
        "shoot_list": [plate("desk wide at night"), plate("you deadpan", person=True), plate("unused")]}])

    text, path = service.ideate(conn, settings, deps, n=5)

    call = [c for c in fake_llm.calls() if c["response_format"]["json_schema"]["name"] == "IdeaReply"][0]
    assert "blends 2, from_inspiration 2, from_brief 1" in call["prompt"]  # the asked mix for n=5
    assert f"c{c1}:" in call["prompt"] and f"{bid}:" in call["prompt"]  # both sources, each with its own ids
    assert "INSPIRATION (saved posts)" in call["prompt"] and "BRIEF (the creator own words)" in call["prompt"]
    data = json.loads((path.parent / "ideas.json").read_text())
    assert [i["origin"] for i in data["ideas"]] == ["blend", "inspiration", "own", "new", "inspiration"]
    assert data["ideas"][3]["inspiration"] == [] and data["ideas"][3]["brief"] == []  # unknown refs dropped
    assert data["ideas"][2]["plates"] == [1]  # no plate 7
    assert data["ideas"][4]["line"] is None  # copied a saved post's overlay
    assert data["ideas"][0]["brief"] == [bid] and data["ideas"][1]["line"] == "quoted line"
    assert data["shoot_list"][0]["serves"] == [0, 1, 3, 4]
    md = path.read_text()
    assert md.index("## Blends") < md.index("## From inspiration only") < md.index("## From your brief only")
    assert "UNKNOWN_REF" in md and "IDEA_COPY" in md and "| P0 | 0, 1, 3, 4 |" in md
    assert md.index("| P0 |") < md.index("| P1 |") < md.index("| P2 | - |")  # most reused plate first
    assert "Uses 2 of 2 inspiration posts and 1 of 3 brief items" in md
    assert "Origins: blend 1, inspiration 2, own 1, new 1" in text


def test_ideate_without_a_brief_still_runs(conn, settings, fake_llm, tiny_video, tiny_jpg, local_calls):
    deps = _store(conn, settings, fake_llm, tiny_video, tiny_jpg)
    fake_llm.queue(Teardown=[teardown_reply(), teardown_reply()],
                   IdeaReply=[{"from_brief": [idea("x", brief=["b-12345"])], "shoot_list": [plate("desk")]}])
    _, path = service.ideate(conn, settings, deps, n=3)
    data = json.loads((path.parent / "ideas.json").read_text())
    assert data["ideas"][0]["origin"] == "new"  # nothing in the brief to cite


def test_local_engine_sizes_the_window_and_refuses_overflow(settings, monkeypatch):
    import pytest

    from wonhee import localllm
    sent = []
    monkeypatch.setattr(localllm, "_post", lambda s, body: sent.append(body) or
                        {"message": {"content": '{"shoot_list": []}'}, "prompt_eval_count": 100})
    localllm.chat_json(settings, ix.IdeaReply, system="s", prompt="p")
    assert sent[0]["options"]["num_ctx"] == settings.local_llm_ctx
    with pytest.raises(localllm.LocalLLMError, match="too long"):
        localllm.chat_json(settings, ix.IdeaReply, system="s", prompt="x" * settings.local_llm_ctx * 3)
    monkeypatch.setattr(localllm, "_post", lambda s, body: {"message": {"content": "{}"},
                                                            "prompt_eval_count": settings.local_llm_ctx})
    with pytest.raises(localllm.LocalLLMError, match="was cut"):
        localllm.chat_json(settings, ix.IdeaReply, system="s", prompt="p")


def test_too_few_ideas_fails_the_step(conn, settings, fake_llm, tiny_video, tiny_jpg, local_calls):
    deps = _store(conn, settings, fake_llm, tiny_video, tiny_jpg)
    fake_llm.queue(Teardown=[teardown_reply(), teardown_reply()],
                   IdeaReply=[{"new": [idea("lonely")], "shoot_list": [plate("desk")]}])
    text, path = service.ideate(conn, settings, deps, n=12)
    assert path is None and "only 1 of 12 ideas" in text
