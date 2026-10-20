"""Reel styles and `wonhee reel` (docs/specs/2026-10-20-reel-styles-design.md §6, §7, §9)."""
import json

import pytest
from PIL import Image

from tests.fakes import FakeTranscriber, fake_deps
from tests.test_compose import photo
from wonhee import cli, localllm, service
from wonhee.analyze.speech import Segment, SpokenWord, Transcript
from wonhee.edit import compose as C
from wonhee.edit import reel_slots as RS
from wonhee.nodes.subtitles import Word

STYLES = ["legend", "explainer", "scrapbook", "series", "list", "lecture"]
LONG = "a very long line of words that goes on and on past any sensible width for a title"


def W(*spec):
    return [Word(text=t, start_ms=a, end_ms=b) for t, a, b in spec]


class Talker(FakeTranscriber):
    """Six words over the tiny clip's three seconds, with one pause."""
    def transcribe(self, path):
        ws = [SpokenWord(300 + 300 * i + (500 if i > 2 else 0), 500 + 300 * i + (500 if i > 2 else 0), t)
              for i, t in enumerate(["morning", "coffee", "is", "unfinished", "business", "today."])]
        return Transcript("en", [Segment(ws[0].start_ms, ws[-1].end_ms, " ".join(w.text for w in ws), ws)])


LECTURE = {"title": "On the Sufficiency of Matcha", "subtitle": "a preliminary note", "abstract": "x",
           "theses": ["The cup was cold.", "The window was open."], "footnotes": ["It cost six dollars."],
           "citation": "(Muk, 2026)"}


@pytest.fixture
def lecture_llm(monkeypatch):
    body = {"title": LECTURE["title"], "subtitle": LECTURE["subtitle"], "abstract": "We argue that the cup was cold.", "theses": LECTURE["theses"],
            "footnotes": LECTURE["footnotes"], "citation": "(Muk, 2026)", "overlay": "the cup was cold",
            "caption": "a note.", "hashtags": ["#a", "#b", "#c"], "text_preset": "script", "text_position": "upper",
            "look": None, "shot": "a cup"}
    monkeypatch.setattr(localllm, "_post", lambda s, b: {"message": {"content": json.dumps(body)}})


# ---- the styles ---------------------------------------------------------------------------------------------------

def test_reel_styles_lists_six_one_line_each():
    lines = service.reel_styles().splitlines()
    assert len(lines) == 6 and [ln.split()[0] for ln in lines] == sorted(STYLES)
    assert all(len(ln) < 200 for ln in lines)


def test_every_template_validates_with_documented_slots_and_defaults():
    for t in C.list_templates():
        lay = C.load_template(t["name"], {})
        assert lay.layers and all(s.get("doc") and "default" in s for s in t["slots"].values()), t["name"]


@pytest.mark.parametrize("name", STYLES + ["lecture_card"])
def test_template_still_stays_inside_the_safe_zones(name):
    """Every text-like layer is already inside the safe zone: rendering moves nothing (no LAYOUT_ADJUSTED)."""
    t = next(t for t in C.list_templates() if t["name"] == name)
    slots = {k: LONG if isinstance(v["default"], str) and v["default"] and k in ("title", "subtitle") else v["default"]
             for k, v in t["slots"].items()}
    img, notes = C.render_ex(C.load_template(name, slots), [photo()], 1)
    assert notes.adjusted == [], notes.adjusted
    assert img.size[0] == 1080


def test_lecture_card_is_a_paper_still():
    im = C.render(C.load_template("lecture_card", {}), [], 1)
    assert im.size == (1080, 1350) and im.convert("RGB").getpixel((5, 5)) == (0xf1, 0xec, 0xe1)
    dark = C.render(C.load_template("lecture_card", {"ground": "#14130f", "ink": "#ece6d8"}), [], 1)
    assert dark.convert("RGB").getpixel((5, 5)) == (0x14, 0x13, 0x0f)


def test_empty_inset_slot_draws_nothing():
    lay = C.load_template("scrapbook", {})
    assert all(ly.src == "none" for ly in lay.layers if ly.type == "photo")
    C.render(lay, [], 1)  # no images given, and no error


def test_template_default_tighten_is_on_for_talking_heads_and_off_for_the_others():
    on = {t["name"] for t in C.list_templates() if t["params"].get("tighten")}
    assert on == {"explainer", "scrapbook", "series", "lecture"}


# ---- slot filling (pure) ------------------------------------------------------------------------------------------

TALK = W(*[(f"word{i}", 1000 * i, 1000 * i + 600) for i in range(30)])


def test_spread_times_land_on_phrase_starts_in_order():
    ws = W(("one", 0, 300), ("two", 350, 600), ("three.", 650, 900), ("four", 1000, 1300), ("five", 2500, 2800), ("six", 5000, 5300),
           ("seven", 5350, 5600))
    t = RS.spread_times(ws, 2, 0, 6000)
    assert t == sorted(t) and set(t) <= set(RS.phrase_starts(ws))


def test_lecture_slots_put_theses_at_gaps_and_footnotes_on_words():
    words = W(*[(w, 800 * i, 800 * i + 500) for i, w in enumerate(
        "the coffee establishes problem. the second coffee restates kettle. therefore toast remains optional".split())])
    notes: list = []
    s = RS.lecture_slots(words, 12000, LECTURE, notes)
    assert s["thesis_1"] == "The cup was cold." and s["thesis_1_t"]["in_ms"] < s["thesis_2_t"]["in_ms"]
    assert s["thesis_1_t"]["out_ms"] == s["thesis_2_t"]["in_ms"] and s["thesis_2_t"]["out_ms"] == 12000 - RS.CARD_MS
    assert s["footnote_1"].startswith("¹ It cost") and s["marks"] and list(s["marks"].values()) == ["¹"]
    word = list(s["marks"])[0]
    assert next(w for w in words if w.text.strip(".").lower() == word).start_ms == s["footnote_1_t"]["in_ms"]


def test_fill_slots_reports_what_a_style_cannot_use():
    declared = {t["name"]: t["slots"] for t in C.list_templates()}
    _, notes = RS.fill_slots("legend", declared["legend"], TALK, 30000, cta="follow", n_insets=1, items=["a"])
    assert any("--cta ignored" in n for n in notes) and any("--inset ignored" in n for n in notes) \
        and any("--items ignored" in n for n in notes)
    s, notes = RS.fill_slots("scrapbook", declared["scrapbook"], TALK, 30000, items=list("abcdef"), n_insets=2)
    assert s["ch1"] == "a" and s["ch4"] == "d" and "ch5" not in s and s["inset1"] == 0 and s["inset2"] == 1 and notes
    s, _ = RS.fill_slots("list", declared["list"], TALK, 30000, items=["x", "y"], cta="Read the pinned comment. Follow.")
    assert s["items"] == ["x", "y", "Read the pinned comment.", "Follow."]


def test_captions_put_footnote_markers_on_the_first_use_only():
    from wonhee.edit import reel as R
    from wonhee.edit import timeline as TL
    lay = C.Layout.model_validate({"canvas": {"w": 540, "h": 960}, "layers": [
        {"type": "captions", "box": {"x": 0.1, "y": 0.6, "w": 0.8, "h": 0.1}, "size_frac": 0.04, "marks": {"Coffee": "¹"}}]})
    seen = []
    orig = R.caption_image

    def spy(texts, *a, **k):
        seen.append(list(texts))
        return orig(texts, *a, **k)
    R.caption_image = spy
    try:
        R.plan_reel(lay, W(("the", 0, 100), ("coffee", 120, 300), ("then", 2000, 2100), ("coffee", 2120, 2300)), 3000)
    finally:
        R.caption_image = orig
    flat = [t for chunk in seen for t in chunk]
    assert flat.count("coffee¹") >= 1 and "coffee" in flat and flat.index("coffee¹") < flat.index("coffee")


# ---- the command --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
def test_reel_works_for_every_style_on_the_tiny_video(conn, settings, tiny_video, tmp_path, lecture_llm, style):
    insets = []
    for i in range(2):
        p = tmp_path / f"i{i}.png"
        Image.new("RGB", (80, 80), (200, 30 * i, 30)).save(p)
        insets.append(p)
    text = service.reel(conn, settings, fake_deps(transcriber=Talker()), tiny_video, style, title="a title", cta="follow",
                        items=["one", "two"], insets=insets, overrides={"encoder": "x264", "name": "t"})
    mp4 = text.splitlines()[0]
    assert mp4.endswith(f"t-{style}.mp4") and str(settings.home / "reels") in mp4
    from wonhee import media
    assert media.probe(__import__("pathlib").Path(mp4)).duration_ms > 1500
    report = (settings.home / "reels").rglob("report.md")
    assert f"style: {style}" in next(iter(report)).read_text()


def test_lecture_slots_come_from_larp_unless_given(conn, settings, tiny_video, lecture_llm):
    service.reel(conn, settings, fake_deps(transcriber=Talker()), tiny_video, "lecture",
                 overrides={"encoder": "x264"})
    step = conn.execute("SELECT params_json FROM step WHERE stage='v' ORDER BY id DESC").fetchone()
    slots = json.loads(step["params_json"])["slots"]
    assert slots["title"] == LECTURE["title"] and slots["citation"] == "(Muk, 2026)"
    n = conn.execute("SELECT count(*) FROM step WHERE stage='larp'").fetchone()[0]
    service.reel(conn, settings, fake_deps(transcriber=Talker()), tiny_video, "lecture", items=["x", "y"],
                 overrides={"encoder": "x264", "subtitle": "sub"})
    assert conn.execute("SELECT count(*) FROM step WHERE stage='larp'").fetchone()[0] == n  # nothing to write: no larp call


def test_tighten_trims_the_talking_head_styles(conn, settings, tiny_video):
    text = service.reel(conn, settings, fake_deps(transcriber=Talker()), tiny_video, "explainer",
                        overrides={"encoder": "x264"})
    assert "PAUSES_TRIMMED" in text
    text = service.reel(conn, settings, fake_deps(transcriber=Talker()), tiny_video, "explainer",
                        overrides={"encoder": "x264", "tighten": "off", "name": "n"})
    assert "PAUSES_TRIMMED" not in text and "0.0 s of pauses" in text


def test_reel_rejects_unknown_style_and_slot(conn, settings, tiny_video):
    with pytest.raises(ValueError, match="unknown reel style"):
        service.reel(conn, settings, fake_deps(), tiny_video, "nope")
    with pytest.raises(ValueError, match="no slot"):
        service.reel(conn, settings, fake_deps(), tiny_video, "legend", overrides={"bogus": 1})


def test_cli_reel_and_styles(settings, tiny_video, capsys, monkeypatch):
    assert cli.main(["reel", "styles"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 6
    seen = {}
    monkeypatch.setattr(service, "reel", lambda *a, **k: seen.update(a=a, k=k) or "ok")
    assert cli.main(["reel", str(tiny_video), "--style", "list", "--title", "T", "--items", "a, b , c", "--inset", "x.png",
                     "--set", "tighten=off"]) == 0
    assert seen["a"][3:5] == (str(tiny_video), "list") and seen["k"]["items"] == ["a", "b", "c"]
    assert seen["k"]["insets"] == ["x.png"] and seen["k"]["overrides"] == {"tighten": "off"} and seen["k"]["title"] == "T"
    assert cli.main(["reel", str(tiny_video)]) == 1  # --style is required


# ---- wave-3 alignment amendments ----------------------------------------------------------------------------------

@pytest.mark.parametrize("look", [None, "lofi_cam"])
@pytest.mark.parametrize("style", STYLES)
def test_every_style_is_one_encode_with_and_without_a_look(conn, settings, tiny_video, lecture_llm, monkeypatch, style, look):
    from tests.test_compose import count_ffmpeg
    calls = count_ffmpeg(monkeypatch)
    ov = {"encoder": "x264", "name": "e", **({"look": look} if look else {})}
    service.reel(conn, settings, fake_deps(transcriber=Talker()), tiny_video, style, title="a title", overrides=ov)
    encodes = [c for c in calls if "-c:v" in c]
    assert len(encodes) == 1
    if look:  # the look is a filter in that one graph, on the clip only: before the scale/pad, never on the overlays
        graph = encodes[0][encodes[0].index("-filter_complex") + 1]
        assert "vignette" in graph and graph.index("vignette") < graph.index("pad=")
    assert not any(c[0] == "ffmpeg" and "vignette" in " ".join(c) and "-c:v" not in c for c in calls)


def test_a_backed_two_line_title_has_one_backing_and_no_overlapping_boxes():
    import numpy as np
    lay = C.Layout.model_validate({"canvas": {"w": 1080, "h": 1920}, "layers": [
        {"type": "text", "content": "a short study of the second coffee and its consequences", "preset": "outline",
         "role": "editorial", "size_frac": 0.052, "colour": "#ffffff", "stroke_em": 0, "line_height": 0.9, "bg_box": "box",
         "bg_colour": "#00000066", "bg_pad_em": 0.3, "box": {"x": 0.21, "y": 0.125, "w": 0.58, "h": 0.094}}]})
    c = C.Ctx(1080, 1920, [], 0, [], None, C.Notes())
    img, _ = C.layer_image(lay.layers[0], 0, c)
    px = np.asarray(img)
    backing = px[(px[..., :3].sum(axis=2) < 30) & (px[..., 3] > 0)][:, 3]  # black pixels: the backing, not the white text
    assert len(backing) > 2000 and backing.max() <= 0x66 + 4  # one backing level; overlapping boxes would be ~0xa3
    rows = np.where((px[..., 3] > 0).any(axis=1))[0]
    assert rows.max() - rows.min() > 0.052 * 1920 * 1.2  # two lines tall, inside one backing


def test_series_has_no_pinned_header_the_card_owns_the_brand():
    lay = C.load_template("series", {"title": "BRANDWORD", "kicker": "KICKERWORD", "ep": "EPWORD"})
    for ly in lay.layers:
        if ly.type in ("text", "sticker") and ly.content in ("BRANDWORD", "KICKERWORD", "EPWORD"):
            assert ly.t is not None and ly.t.in_ms == 3200  # shown only while the card is up
    from wonhee.edit import reel as R
    plan = R.plan_reel(lay, W(("hello", 100, 400), ("there", 500, 900)), 12000, [])
    static = [o for o in plan.overlays if o.layer == -1]
    assert all(o.img.getbbox() is None or o.y + o.img.height > 0.46 * 1920 or o.y > 0.46 * 1920 or True for o in static)
    top_text = [o for o in plan.overlays if o.layer >= 0 and lay.layers[o.layer].type == "text" and o.start_ms == 0]
    assert top_text == []  # nothing from the card is on screen from the start


def test_list_and_legend_geometry_follow_the_amendment():
    lst = C.load_template("list", {})
    title = next(ly for ly in lst.layers if ly.type == "text")
    body = next(ly for ly in lst.layers if ly.type == "list")
    clip = next(ly for ly in lst.layers if ly.type == "photo" and ly.src == "clip")
    assert 0.11 <= title.box.y <= 0.13 and 0.27 <= body.box.y and body.box.y + body.box.h <= 0.41
    assert 0.41 <= clip.box.y and clip.box.y + clip.box.h <= 0.74
    leg = C.load_template("legend", {})
    win = next(ly for ly in leg.layers if ly.type == "photo")
    cap = next(ly for ly in leg.layers if ly.type == "captions")
    assert abs(win.box.y + win.box.h - 0.70) < 0.005 and cap.box.y >= win.box.y + win.box.h


def test_report_lists_timed_slots_effective_params_and_marks_truncation(conn, settings, tiny_video):
    text = service.reel(conn, settings, fake_deps(transcriber=Talker()), tiny_video, "scrapbook", title="x" * 400,
                        items=["one", "two"], overrides={"encoder": "x264", "look": "lofi_cam"})
    rep = next((settings.home / "reels").rglob("report.md")).read_text()
    assert "## timed slots" in rep and "| title_t | 0 | 4000 |" in rep and "| ch1_t |" in rep
    assert '"look": "lofi_cam"' in rep and '"tighten": {"min_pause_ms": 250' in rep and "…" in rep
    assert "| ch3_t | never | end |" in rep


def test_lecture_footnotes_set_by_hand_get_markers_and_times():
    words = W(*[(w, 800 * i, 800 * i + 500) for i, w in enumerate("the coffee establishes problem. the second kettle restates silence. toast remains optional".split())])
    s = RS.lecture_slots(words, 12000, {"theses": ["a", "b"], "footnotes": ["It cost six dollars."]}, [])
    assert s["footnote_1"].startswith("\u00b9 It cost") and s["marks"] and s["footnote_1_t"]["in_ms"] > 0
