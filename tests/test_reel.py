"""compose_video: timed layers, word-driven kinds and pause trimming (docs/specs/2026-10-20-reel-styles-design.md §2, §3, §9)."""
import re
import subprocess

import numpy as np
import pytest
from PIL import Image

from tests.fakes import fake_deps
from tests.test_compose import count_ffmpeg, photo, store_images
from tests.test_nodes_subs import flags, out, run_chain, store_video
from wonhee import media
from wonhee.edit import compose as C
from wonhee.edit import reel as R
from wonhee.edit import timeline as TL
from wonhee.nodes.subtitles import Line, Subtitles, Word, save_subs

FRAME_MS = 34  # one frame at 30 fps (tiny_video)


def W(*spec):
    """Words from (text, start_ms, end_ms) triples."""
    return [Word(text=t, start_ms=a, end_ms=b) for t, a, b in spec]


TALK = W(("alpha", 0, 400), ("beta", 500, 900), ("gamma", 1700, 2100), ("delta", 2200, 2600))  # a 800 ms pause after beta


def lay(layers, w=540, h=960, bg="#000000", **kw):
    return C.Layout.model_validate({"canvas": {"w": w, "h": h, "bg": bg}, "layers": layers, **kw})


def subs_of(conn, settings, words):
    line = Line(id=0, start_ms=words[0].start_ms, end_ms=words[-1].end_ms, text=" ".join(w.text for w in words),
                source="speech", words=words)
    return save_subs(conn, settings.media_dir, Subtitles(lines=[line]))


def frame_at(video, t, path):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(video), "-frames:v", "1", str(path)], check=True)
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def colour_mask(a, rgb, tol=40):
    return np.abs(a - np.array(rgb)).sum(axis=2) < tol


# ---- chunking and pause finding (pure) ---------------------------------------------------------------------------

def test_chunks_break_on_pauses_punctuation_and_max_words():
    ws = W(("one", 0, 100), ("two", 120, 200), ("three.", 220, 300), ("four", 320, 400), ("five", 1200, 1300), ("six", 1320, 1400))
    chunks = [[w.text for w in c] for c in TL.chunk_words(ws, 3, 350)]
    assert chunks == [["one", "two", "three."], ["four"], ["five", "six"]]
    assert all(len(c) <= 2 for c in TL.chunk_words(ws, 2, 350))


def test_find_pauses_cuts_known_gaps_and_timeline_retimes():
    cuts = TL.find_pauses(TALK, 250, 80)
    assert cuts == ((940, 1660),)  # only the 800 ms gap; the 100 ms one stays; 80 ms of it is kept
    tl = TL.Timeline(cuts)
    assert tl.map(900) == 900 and tl.map(1700) == 980 and tl.map(2600) == 1880 and tl.map(1000) == 940
    assert tl.removed_ms() == 720 and tl.kept(3000) == [(0, 940), (1660, 3000)]


@pytest.mark.parametrize("anim", ["fade", "pop", "rise", "type"])
def test_animation_ladders_have_at_most_ten_states(anim):
    st = TL.ladder(anim, 250)
    assert 2 <= len(st) <= TL.MAX_STATES and st[-1].ms == 0
    assert TL.ladder(anim, 3000)[-1].ms == 0 and len(TL.ladder(anim, 3000)) <= TL.MAX_STATES
    assert len(TL.ladder("none")) == 1


# ---- captions ----------------------------------------------------------------------------------------------------

def caption_layer(**kw):
    return {"type": "captions", "box": {"x": 0.1, "y": 0.6, "w": 0.8, "h": 0.1}, "size_frac": 0.04, "highlight": "#ff00ff",
            "colour": "#ffffff", **kw}


def test_karaoke_has_one_state_per_word_at_its_time_and_never_more_than_max_words():
    layout = lay([caption_layer(mode="karaoke", max_words=3, bg_box="pill", bg_colour="#202020")])
    plan = R.plan_reel(layout, TALK, 3000)
    ov = [o for o in plan.overlays if o.layer == 0]
    assert [o.start_ms for o in ov] == [w.start_ms for w in TALK]  # a state per word, at that word's time (exact)
    chunks = TL.chunk_words(TALK, 3, 350)
    assert max(len(c) for c in chunks) <= 3
    # contiguous inside a chunk: the pill moves when the next word starts, no gap and no overlap
    for a, b in zip(ov, ov[1:]):
        assert a.end_ms <= b.start_ms + FRAME_MS
    pill = lambda o: colour_mask(np.asarray(o.img.convert("RGB")).astype(int) * (np.asarray(o.img)[..., 3:] > 200), (255, 0, 255), 30)
    cx = [o.x + np.where(pill(o))[1].mean() for o in ov]
    assert cx[0] < cx[1] and cx[2] < cx[3]  # the pill walks across each chunk (alpha beta | gamma delta) word by word


def test_word_mode_shows_words_one_by_one_and_chunk_mode_all_at_once():
    word = R.plan_reel(lay([caption_layer(mode="word")]), TALK, 3000).overlays
    chunk = R.plan_reel(lay([caption_layer(mode="chunk")]), TALK, 3000).overlays
    assert len(word) == 4 and len(chunk) == 2
    ink = [np.asarray(o.img)[..., 3].astype(bool).sum() for o in word]
    assert ink[0] < ink[1] and ink[2] < ink[3]  # each chunk's words accumulate (alpha beta | gamma delta)


def test_text_obeys_the_safe_zone_on_tall_canvases():
    """A caption box placed in the caption zone is moved to 12%..76% of the height, as text.py does."""
    W_, H_ = 1080, 1920
    low = {"x": 0.05, "y": 0.85, "w": 0.9, "h": 0.1}
    for layer in [caption_layer(box=low, mode="chunk"), {"type": "stack", "box": low, "size_frac": 0.05},
                  {"type": "list", "box": low, "size_frac": 0.03, "window": 3}]:
        plan = R.plan_reel(lay([layer], W_, H_), TALK, 3000)
        assert plan.overlays and plan.notes.adjusted
        for o in plan.overlays:
            assert o.y >= H_ * 0.12 - 2 and o.y + o.img.height <= H_ * 0.76 + 2
            if o.y + o.img.height > H_ * 0.35:  # reaching the buttons: inside the centred 72%
                assert o.x >= W_ * 0.14 - 2 and o.x + o.img.width <= W_ * 0.86 + 2


# ---- stack and list ----------------------------------------------------------------------------------------------

def test_stack_changes_only_at_word_times_and_builds_a_line_per_word():
    layout = lay([{"type": "stack", "box": {"x": 0.1, "y": 0.2, "w": 0.8, "h": 0.4}, "size_frac": 0.06, "max_lines": 4}])
    ov = R.plan_reel(layout, TALK, 3000).overlays
    assert [o.start_ms for o in ov] == [w.start_ms for w in TALK]
    heights = [o.img.height for o in ov]
    assert heights[0] < heights[1]  # beta joins alpha in the same phrase; the pause starts a new phrase after it
    assert heights[2] < heights[1] or heights[2] <= heights[0] + 5  # gamma starts a fresh stack


def runs(alpha_col):
    rows = np.where(alpha_col > 0)[0]
    if not len(rows):
        return []
    groups = np.split(rows, np.where(np.diff(rows) > 8)[0] + 1)
    return [(g[0], g[-1]) for g in groups]


def test_list_shows_at_most_n_lines_with_the_active_one_brightest_and_changes_at_entries():
    items = ["one", "two", "three", "four", "five", "six"]
    at = [0, 400, 900, 1500, 2100, 2500]
    layout = lay([{"type": "list", "items": items, "at_ms": at, "window": 5, "size_frac": 0.05, "scroll_ms": 0,
                   "box": {"x": 0.1, "y": 0.3, "w": 0.8, "h": 0.35}, "colour": "#ffffff", "stroke_em": 0}])
    plan = R.plan_reel(layout, [], 3000)
    ov = plan.overlays
    assert [o.start_ms for o in ov] == at  # without a scroll the list changes only at entry times
    for k, o in enumerate(ov):
        a = np.asarray(o.img)[..., 3]
        rs = runs(a.max(axis=1))
        assert len(rs) <= 5
        peaks = [a[r0:r1 + 1].max() for r0, r1 in rs]
        active = max(range(len(peaks)), key=lambda j: peaks[j])
        assert peaks[active] == 255 and sorted(peaks, reverse=True)[1] <= 0.26 * 255 + 1  # neighbours at 25%
        # the active entry is the k-th: slots before it exist only when earlier entries do
        assert active == min(k, 2)


# ---- timed ordinary layers, insets -------------------------------------------------------------------------------

def test_a_timed_layer_with_pop_has_few_states_and_ends_at_out():
    layout = lay([{"type": "text", "content": "HEADLINE", "box": {"x": 0.05, "y": 0.2, "w": 0.9, "h": 0.1},
                   "t": {"in_ms": 1000, "out_ms": 2400}, "anim": "pop", "size_frac": 0.05}])
    ov = R.plan_reel(layout, [], 3000).overlays
    assert 2 <= len(ov) <= TL.MAX_STATES
    assert ov[0].start_ms == 1000 and ov[-1].end_ms == 2400
    assert all(a.end_ms == b.start_ms for a, b in zip(ov, ov[1:]))
    widths = [o.img.width for o in ov]
    assert widths[0] < max(widths) and widths[-1] < max(widths)  # 0.85 -> 1.06 -> 1.0


def test_type_anim_reveals_characters_and_fade_ramps_alpha():
    layout = lay([{"type": "text", "content": "typing a sentence here", "box": {"x": 0.05, "y": 0.2, "w": 0.9, "h": 0.1},
                   "anim": "type", "dur_ms": 900, "size_frac": 0.05, "cursor": True},
                  {"type": "shape", "kind": "rect", "fill": "#ff0000", "box": {"x": 0.1, "y": 0.5, "w": 0.2, "h": 0.1},
                   "anim": "fade", "dur_ms": 300, "t": {"in_ms": 500}}])
    ov = R.plan_reel(layout, [], 3000).overlays
    typed = [o for o in ov if o.layer == 0]
    assert len(typed) <= TL.MAX_STATES and typed[0].img.width < typed[-1].img.width
    fade = [o for o in ov if o.layer == 1]
    a = [np.asarray(o.img)[..., 3].max() for o in fade]
    assert a == sorted(a) and a[0] < a[-1] == 255 and fade[0].start_ms == 500 and fade[-1].end_ms is None


def test_inset_is_a_timed_photo_layer_from_the_images_input():
    layout = lay([{"type": "photo", "src": 0, "shape": "rounded", "box": {"x": 0.1, "y": 0.05, "w": 0.8, "h": 0.2},
                   "t": {"in_ms": 800, "out_ms": 1800}, "anim": "fade",
                   "outlines": [{"width_frac": 0.01, "colour": "#ffffff"}]}])
    ov = R.plan_reel(layout, [], 3000, [photo()]).overlays
    assert ov[0].start_ms == 800 and ov[-1].end_ms == 1800 and len(ov) <= TL.MAX_STATES


def test_typography_fields_reach_the_text_layer():
    base = {"type": "text", "content": "Hello there", "box": {"x": 0, "y": 0, "w": 1, "h": 1}, "size_frac": 0.1}
    plain = C.render(lay([base], 600, 300))
    serif = C.render(lay([{**base, "role": "editorial", "tracking_em": 0.1, "stroke_em": 0, "bg_box": "pill",
                           "bg_colour": "#ff0000"}], 600, 300))
    assert plain.tobytes() != serif.tobytes()
    assert (np.asarray(serif.convert("RGB")).astype(int) == [255, 0, 0]).all(axis=2).sum() > 500
    with pytest.raises(ValueError):
        C.render(lay([{**base, "role": "no-such-role"}], 600, 300))


# ---- the node: one encode, duration, tighten ----------------------------------------------------------------------

def run_video(conn, settings, tiny_video, layout, words, monkeypatch, extra=None, images=None):
    sha = store_video(conn, settings, tiny_video)
    calls = count_ffmpeg(monkeypatch)
    ins = {"video": sha, "subtitles": subs_of(conn, settings, words)}
    if images:
        ins["images"] = images
    rid, status = run_chain(conn, settings, [{"id": "v", "node": "compose_video", "in": ins,
                                              "params": {"layout": layout.model_dump(mode="json"), "encoder": "x264", **(extra or {})}}],
                            fake_deps())
    assert status == "done", flags(conn, rid, "v")
    return rid, calls, media.path_for(settings.media_dir, out(conn, rid, "v", "video"))


EVERYTHING = [
    {"type": "photo", "src": "video", "box": {"x": 0, "y": 0.25, "w": 1, "h": 0.4}, "clip_fit": "cover"},
    {"type": "text", "content": "TITLE", "box": {"x": 0.05, "y": 0.13, "w": 0.9, "h": 0.1}, "t": {"in_ms": 300}, "anim": "pop"},
    {"type": "captions", "mode": "karaoke", "box": {"x": 0.1, "y": 0.66, "w": 0.8, "h": 0.08}, "highlight": "#ff00ff",
     "size_frac": 0.04, "bg_box": "box", "bg_colour": "#202020"},
    {"type": "stack", "box": {"x": 0.1, "y": 0.14, "w": 0.8, "h": 0.1}, "size_frac": 0.04, "t": {"out_ms": 900}},
    {"type": "list", "items": ["a", "b", "c"], "box": {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.1}, "size_frac": 0.03, "window": 3},
    {"type": "photo", "src": 0, "box": {"x": 0.7, "y": 0.13, "w": 0.2, "h": 0.1}, "t": {"in_ms": 1000, "out_ms": 2000}, "anim": "fade"},
]


def test_compose_video_is_one_encode_whatever_the_layers_and_keeps_the_duration(conn, settings, tiny_video, monkeypatch, tmp_path):
    layout = lay(EVERYTHING, 540, 960)
    imgs = store_images(conn, settings, tmp_path, photo())
    rid, calls, dst = run_video(conn, settings, tiny_video, layout, TALK, monkeypatch, images=imgs)
    assert len(calls) == 1  # exactly one ffmpeg call: no silencedetect without tighten, no intermediate renders
    info = media.probe(dst)
    assert (info.width, info.height) == (540, 960) and abs(info.duration_ms - 3000) < 150 and media.has_audio(dst)
    assert "PAUSES_TRIMMED" not in flags(conn, rid, "v")


def test_tighten_removes_known_pauses_and_the_captions_move_with_them(conn, settings, tiny_video, monkeypatch, tmp_path):
    cap = {"type": "captions", "mode": "karaoke", "box": {"x": 0.1, "y": 0.66, "w": 0.8, "h": 0.08}, "highlight": "#ff00ff",
           "size_frac": 0.04, "colour": "#ffffff", "bg_box": "box", "bg_colour": "#202020"}
    layout = lay([{"type": "photo", "src": "video", "box": {"x": 0, "y": 0.2, "w": 1, "h": 0.4}}, cap], 540, 960)
    plan = R.plan_reel(layout, TALK, 3000, timeline=TL.Timeline(TL.find_pauses(TALK)))
    starts = [o.start_ms for o in plan.overlays if o.layer == 1]
    assert starts == [0, 500, 980, 1480]  # gamma and delta moved 720 ms earlier (exact; one frame is 34 ms)
    assert plan.duration_ms == 2280
    calls = []
    rid, calls, dst = run_video(conn, settings, tiny_video, layout, TALK, monkeypatch, extra={"tighten": {}})
    assert len(calls) == 1
    assert abs(media.probe(dst).duration_ms - 2266) < 120 and media.has_audio(dst)
    f = [x for x in flags(conn, rid, "v")]
    assert "PAUSES_TRIMMED" in f
    from wonhee.db import queries
    detail = [fl for fl in queries.step_flags(conn, queries.current_step(conn, rid, "v")["id"]) if fl["code"] == "PAUSES_TRIMMED"][0]["detail"]
    assert "0.73 s" in detail  # the 720 ms cut, snapped to the 30 fps frame grid (734 ms)
    # gamma is active at its new time (1.0 s), not at its old one: the pill is on the third word
    a1 = frame_at(dst, 1.02, tmp_path / "a.png")
    pill = colour_mask(a1[int(0.66 * 960):int(0.74 * 960)], (255, 0, 255), 60)
    assert pill.sum() > 150
    a0 = frame_at(dst, 1.6, tmp_path / "b.png")  # delta, the chunk's second word (gamma's chunk starts at 0.97 s)
    p0 = colour_mask(a0[int(0.66 * 960):int(0.74 * 960)], (255, 0, 255), 60)
    assert p0.sum() > 100 and np.where(p0)[1].mean() > np.where(pill)[1].mean()  # the pill moved right


def test_tighten_without_subtitles_uses_silencedetect_in_a_separate_pass(conn, settings, tiny_video, monkeypatch):
    sha = store_video(conn, settings, tiny_video)  # the fixture's tone never goes silent: nothing to cut
    calls = count_ffmpeg(monkeypatch)
    layout = lay([{"type": "photo", "src": "video", "box": {"x": 0, "y": 0.2, "w": 1, "h": 0.4}}], 540, 960)
    rid, status = run_chain(conn, settings, [{"id": "v", "node": "compose_video", "in": {"video": sha},
                                              "params": {"layout": layout.model_dump(mode="json"), "encoder": "x264",
                                                         "tighten": {}}}], fake_deps())
    assert status == "done" and len(calls) == 2  # one analysis pass (decode to null) and one encode
    assert sum(1 for c in calls if "libx264" in c) == 1


def test_frame_video_alias_still_works_and_versions_are_bumped(conn, settings, tiny_video, monkeypatch):
    from wonhee.nodes import registry
    assert registry.get("frame_video").version >= 3 and registry.get("compose_video").version >= 1
    assert registry.get("compose").version >= 3 and registry.get("compose_video").salt(settings)["templates"]
    sha = store_video(conn, settings, tiny_video)
    calls = count_ffmpeg(monkeypatch)
    rid, status = run_chain(conn, settings, [{"id": "f", "node": "frame_video", "in": {"video": sha},
                                              "params": {"encoder": "x264"}}], fake_deps())
    assert status == "done" and len(calls) == 1


def test_words_estimated_is_flagged_for_word_driven_layers_on_line_only_subtitles(conn, settings, tiny_video, monkeypatch):
    sha = store_video(conn, settings, tiny_video)
    subs = save_subs(conn, settings.media_dir, Subtitles(lines=[Line(id=0, start_ms=0, end_ms=2000, text="a b c d", source="speech")]))
    layout = lay([{"type": "photo", "src": "video", "box": {"x": 0, "y": 0.2, "w": 1, "h": 0.4}},
                  {"type": "captions", "box": {"x": 0.1, "y": 0.66, "w": 0.8, "h": 0.08}}], 540, 960)
    rid, status = run_chain(conn, settings, [{"id": "v", "node": "compose_video", "in": {"video": sha, "subtitles": subs},
                                              "params": {"layout": layout.model_dump(mode="json"), "encoder": "x264"}}], fake_deps())
    assert status == "done" and "WORDS_ESTIMATED" in flags(conn, rid, "v")


# ---- wave-2 alignment amendments (reel-styles spec §2 and §3 amendment blocks) ------------------------------------

def test_chunks_are_decided_on_the_source_timeline_then_mapped_through_the_cuts():
    tl = TL.Timeline(TL.find_pauses(TALK))  # beta..gamma: 800 ms -> 80 ms kept after cutting
    ov = R.plan_reel(lay([caption_layer(mode="chunk")]), TALK, 3000, timeline=tl).overlays
    assert len(ov) == 2 and ov[0].end_ms <= ov[1].start_ms  # the pause still breaks the chunk (not 3 words in one)
    assert ov[1].start_ms == 980
    stack = R.plan_reel(lay([{"type": "stack", "box": {"x": .1, "y": .2, "w": .8, "h": .4}, "size_frac": .06}]), TALK, 3000,
                        timeline=tl).overlays
    assert stack[2].img.height < stack[1].img.height  # gamma starts a fresh stack


def test_cuts_snap_to_the_frame_grid_and_stay_inside_the_clip():
    cuts = TL.snap_cuts(((940, 1660), (2990, 5000)), 30.0, 3000)
    assert cuts == ((933, 1667),) or all(abs(a / (1000 / 30) - round(a / (1000 / 30))) < 0.01 for c in cuts for a in c)
    for a, b in cuts:
        assert abs(a / (1000 / 30) - round(a / (1000 / 30))) < 0.02 and b <= 3000
    tl = TL.Timeline(((900, 2000),))
    assert tl.kept(1000) == [(0, 900)]


def test_anim_is_refused_on_word_driven_layers_and_clip_layer_takes_no_shape():
    for layer in [caption_layer(anim="fade"), {"type": "stack", "anim": "pop"}, {"type": "list", "anim": "rise"}]:
        with pytest.raises(ValueError, match="entrances"):
            lay([layer])
    lay([caption_layer(t={"in_ms": 100})])  # t still clips them
    for bad in [{"shape": "heart"}, {"rotate": 5}, {"outlines": [{"width_frac": 0.01, "colour": "#fff"}]}]:
        with pytest.raises(ValueError, match="clip layer"):
            lay([{"type": "photo", "src": "video", **bad}])


def test_list_scrolls_with_at_most_eight_intermediate_states_per_entry():
    at = [0, 1000, 2000]
    layout = lay([{"type": "list", "items": ["one", "two", "three"], "at_ms": at, "window": 3, "size_frac": 0.05, "scroll_ms": 470,
                   "box": {"x": 0.1, "y": 0.3, "w": 0.8, "h": 0.35}}])
    ov = R.plan_reel(layout, [], 3000).overlays
    starts = [o.start_ms for o in ov]
    for k, a in enumerate(at):
        inside = [t for t in starts if a < t < (at[k + 1] if k + 1 < len(at) else 3000)]
        scroll = [t for t in inside if t - a < 470]
        assert len(scroll) <= 8  # only the scroll's offset states between entry times (then the entry at rest)
        if k == 0:
            assert not inside  # the first entry just appears
        else:
            assert scroll and a + 470 in inside and a in starts
    assert [t for t in starts if t in at] == at
    assert all(x.end_ms == y.start_ms for x, y in zip(ov, ov[1:]) if x.end_ms is not None and y.start_ms < 2000 + 470)
    assert R.plan_reel(lay([{**layout.layers[0].model_dump(mode="json"), "scroll_ms": 0}]), [], 3000).overlays.__len__() == 3


def test_overlay_cap_is_1500():
    assert R.MAX_OVERLAYS == 1500


def test_paper_backing_is_textpys_paper_and_grows_the_layer_image(monkeypatch):
    layer = caption_layer(bg_box="paper", bg_colour="#f4f2ee", colour="#111111", stroke_em=0, size_frac=0.05,
                          box={"x": 0.1, "y": 0.6, "w": 0.8, "h": 0.01})  # the layer is far shorter than its backing
    ov = R.plan_reel(lay([layer]), TALK, 3000).overlays
    assert ov
    layer_h = round(0.01 * 960)
    assert ov[0].img.height > layer_h * 3  # grew instead of clipping
    arr = np.asarray(ov[0].img).astype(int)
    assert (colour_mask(arr[..., :3], (0xf4, 0xf2, 0xee), 12) & (arr[..., 3] > 250)).sum() > 500  # the paper colour
    assert ov[0].img.getbbox() == (0, 0, ov[0].img.width, ov[0].img.height)


def two_pass(calls):
    return [c for c in calls if "-vn" in c]


def test_silencedetect_pass_is_audio_only_and_no_audio_means_untrimmed_and_flagged(conn, settings, tiny_video, monkeypatch, tmp_path):
    layout = lay([{"type": "photo", "src": "video", "box": {"x": 0, "y": 0.2, "w": 1, "h": 0.4}}], 540, 960)
    params = {"layout": layout.model_dump(mode="json"), "encoder": "x264", "tighten": {}}
    sha = store_video(conn, settings, tiny_video)
    calls = count_ffmpeg(monkeypatch)
    rid, status = run_chain(conn, settings, [{"id": "v", "node": "compose_video", "in": {"video": sha}, "params": params}], fake_deps())
    assert status == "done" and len(two_pass(calls)) == 1  # the analysis pass skips video decoding (-vn)
    mute = tmp_path / "mute.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tiny_video), "-an", "-c:v", "copy", str(mute)], check=True)
    msha = store_video(conn, settings, mute)
    calls.clear()
    rid, status = run_chain(conn, settings, [{"id": "m", "node": "compose_video", "in": {"video": msha}, "params": params}], fake_deps())
    assert status == "done" and "TIGHTEN_SKIPPED" in flags(conn, rid, "m") and "PAUSES_TRIMMED" not in flags(conn, rid, "m")
    assert len(calls) == 1 and not two_pass(calls)  # no analysis possible, so just the encode
    assert abs(media.probe(media.path_for(settings.media_dir, out(conn, rid, "m", "video"))).duration_ms - 3000) < 150


def stream_durations(path):
    import json as _json
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)],
                          capture_output=True, text=True, check=True)
    return {s["codec_type"]: float(s["duration"]) for s in _json.loads(proc.stdout)["streams"] if "duration" in s}


def test_many_cuts_keep_audio_and_video_in_step(conn, settings, tmp_path, monkeypatch):
    clip = tmp_path / "long.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x568:r=30:d=12", "-f", "lavfi",
                    "-i", "sine=frequency=330:duration=12", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(clip)], check=True)
    words = []
    t = 200
    for i in range(10):  # ten words, each followed by a 700 ms gap that gets cut
        words.append(Word(text=f"w{i}", start_ms=t, end_ms=t + 400))
        t += 400 + 700
    sha = store_video(conn, settings, clip)
    subs = subs_of(conn, settings, words)
    layout = lay([{"type": "photo", "src": "video", "box": {"x": 0, "y": 0.2, "w": 1, "h": 0.4}}], 540, 960)
    calls = count_ffmpeg(monkeypatch)
    rid, status = run_chain(conn, settings, [{"id": "v", "node": "compose_video", "in": {"video": sha, "subtitles": subs},
                                              "params": {"layout": layout.model_dump(mode="json"), "encoder": "x264", "tighten": {}}}],
                            fake_deps())
    assert status == "done" and len(calls) == 1
    dst = media.path_for(settings.media_dir, out(conn, rid, "v", "video"))
    d = stream_durations(dst)
    assert abs(d["video"] - d["audio"]) <= 1 / 30 + 0.012  # within one frame after nine cuts
    expect = 12.0 - 9 * (700 - 80) / 1000
    assert abs(d["video"] - expect) < 9 * 1 / 30  # each cut lands within a frame of its exact length


def test_the_clip_is_the_bottom_layer_and_everything_else_draws_over_it(conn, settings, tiny_video, monkeypatch, tmp_path):
    # a red square listed BEFORE the clip layer still ends up over the clip; a timed green one too
    layout = lay([{"type": "shape", "kind": "rect", "fill": "#ff0000", "box": {"x": 0.1, "y": 0.25, "w": 0.2, "h": 0.1}},
                  {"type": "photo", "src": "video", "box": {"x": 0, "y": 0.2, "w": 1, "h": 0.4}, "clip_fit": "cover"},
                  {"type": "shape", "kind": "rect", "fill": "#00ff00", "box": {"x": 0.6, "y": 0.25, "w": 0.2, "h": 0.1},
                   "t": {"in_ms": 500}}], 540, 960, bg="#202020")
    plan = R.plan_reel(layout, [], 3000)
    assert plan.window == (0, 192, 540, 576) and plan.pad_colour == "0x202020" and len(plan.overlays) == 2
    rid, calls, dst = run_video(conn, settings, tiny_video, layout, [Word(text="x", start_ms=0, end_ms=100)], monkeypatch)
    a = frame_at(dst, 1.0, tmp_path / "f.png")
    assert colour_mask(a[250:300, 80:100], (255, 0, 0), 30).mean() > 0.9 and colour_mask(a[250:300, 340:380], (0, 255, 0), 30).mean() > 0.9
    # with no clip layer the clip fills the canvas (cover) and the bg is unused
    p2 = R.plan_reel(lay([{"type": "text", "content": "hi", "box": {"x": 0, "y": .2, "w": 1, "h": .1}}]), [], 3000)
    assert p2.window is None
