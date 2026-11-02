"""take_pick (post-assistant spec §2.2): engine ranking on synthetic transcripts, plus the node's flag and output."""
import json

from wonhee import media
from wonhee.analyze.speech import SpokenWord
from wonhee.edit import takes as T
from wonhee.nodes import takes as N
from wonhee.nodes.subtitles import Line, Subtitles, Word, save_subs


def say(text, t0=0, wlen=300, gap=60, gaps=None):
    """Words of `text` back to back; `gaps` maps a word index to the silence (ms) after it."""
    gaps, out, t = gaps or {}, [], t0
    for i, tok in enumerate(text.split()):
        out.append(SpokenWord(t, t + wlen, tok))
        t += wlen + gaps.get(i, gap)
    return out


CLEAN = "The pump hits at midnight. Nobody notices until the morning. Then everyone pretends it never happened."


def test_constants_match_tighten_defaults():
    from wonhee.nodes.compose import Tighten
    assert T.TIGHTEN_MIN_PAUSE_MS == Tighten().min_pause_ms
    assert T.TIGHTEN_KEEP_MS == Tighten().keep_ms
    assert T.DEAD_AIR_ALLOWANCE_MS > T.TIGHTEN_MIN_PAUSE_MS


def test_clean_take_beats_stumbly_take():
    stumbly = ("The the pump hits um at midnight. Nobody no- Nobody notices until the morning. "
               "Then everyone you know pretends it never happened.")
    plans = T.rank_takes([say(stumbly), say(CLEAN)])
    assert plans[0].take == 1
    assert plans[0].score > max(p.score for p in plans if p.take == 0)
    assert any("stumbles: none" in w for w in plans[0].why)
    s = T.find_stumbles(say(stumbly))
    assert sorted(s.values()) == ["false_start", "filler", "filler", "repeat"]


def test_phrase_repeat_is_one_stumble():
    assert list(T.find_stumbles(say("so the pump the pump hits")).values()) == ["repeat"]


def test_deliberate_beat_survives_long_gap_penalised():
    beat = say("Wait for it. And then nothing happened at all.", gaps={2: 500})   # 500 ms < 650 allowance
    gap = say("Wait for it. And then nothing happened at all.", gaps={2: 4000})
    pb, pg = T.rank_takes([beat])[0], T.rank_takes([gap])[0]
    assert pb.components["dead_air"] == 0
    assert len(pb.spans) == 1 and pb.dropped == []          # the beat is kept inside the span
    # the long gap either costs points or is cut around; either way it scores below the beat take
    assert pg.score < pb.score
    gapped_whole = T._score(*_prep(gap), list(range(2)), 8, None)[1]["dead_air"]
    assert gapped_whole < 0


def _prep(words):
    return words, T.find_stumbles(words), T._sentences(words)


def test_long_clean_deadpan_take_not_penalised_for_flat():
    deadpan = say(" ".join(["The meeting was fine."] * 6), wlen=400, gap=120)   # slow, even, no stumbles
    plan = T.rank_takes([deadpan])[0]
    assert plan.components["stumbles"] == 0 and plan.components["dead_air"] == 0
    assert plan.components["point_early"] == 0
    assert len(plan.spans) == 1 and plan.dropped == []
    assert plan.score == T.COVERAGE_WEIGHT


def _sentences_with_gaps(sents, gap=300):
    words, t = [], 0
    for sent in sents:
        ws = say(sent, t0=t)
        words += ws
        t = ws[-1].end_ms + gap
    return words


def test_contiguous_preferred_over_spliced():
    good_a = "The pump hits at midnight and nobody notices until the morning light."
    bad = "Um you know like it was bad."
    good_b = "Then everyone pretends it never happened and they all go back to sleep."
    plans = T.rank_takes([_sentences_with_gaps([good_a, bad, good_b])], top_n=10)
    spliced = [p for p in plans if p.spliced]
    assert spliced and spliced[0].components["splice"] == -T.SPLICE_WEIGHT
    assert plans[0].spliced is False                      # a single clean sentence beats the spliced pair
    assert plans[0].score > spliced[0].score
    assert len(plans[0].spans) == 1
    # a clean take offers no splice at all
    assert all(not p.spliced for p in T.rank_takes([say(CLEAN)], top_n=10))


def test_splice_refused_when_join_has_no_room():
    # sentences run into each other with 0 ms of silence: cutting between them would clip speech
    t, words = 0, []
    for sent in ["Pump hits at midnight.", "Um you know like it was bad.", "Nobody notices until morning."]:
        ws = say(sent, t0=t, gap=0)
        words += ws
        t = ws[-1].end_ms
    assert all(not p.spliced for p in T.rank_takes([words], top_n=10))


def test_missing_audio_stats_is_none_not_zero():
    p = T.rank_takes([say(CLEAN)])[0]
    assert p.components["loudness"] is None
    assert "loudness" not in " ".join(p.why)
    q = T.rank_takes([say(CLEAN)], loudness_db=[None])[0]
    assert q.components["loudness"] is None and q.score == p.score
    flat = T.rank_takes([say(CLEAN)], loudness_db=[3.0])[0]
    wide = T.rank_takes([say(CLEAN)], loudness_db=[16.0])[0]
    assert flat.components["loudness"] == 0 and wide.score < flat.score
    assert any("16 dB" in w for w in wide.why)


def test_deterministic_and_ties_break_on_take_index():
    takes = [say(CLEAN), say(CLEAN), say(CLEAN)]
    a = [p.to_dict() for p in T.rank_takes(takes)]
    b = [p.to_dict() for p in T.rank_takes(takes)]
    assert a == b and [p["take"] for p in a] == [0, 1, 2]
    assert T.rank_takes([], top_n=3) == [] and T.rank_takes([[]]) == []


def test_plan_reports_numbers_and_dropped_stumbly_lead():
    words = say("Um um okay so. The pump hits at midnight. Nobody notices until the morning.")
    plan = T.rank_takes([words])[0]
    assert plan.start_ms > 0 and plan.dropped and plan.dropped[0][0] == 0
    assert "filler" in plan.dropped[0][2] or "repeat" in plan.dropped[0][2]
    assert plan.dropped[0][1] <= plan.start_ms
    assert all(isinstance(w, str) for w in plan.why) and any("ms" in w for w in plan.why)


class Ctx:
    def __init__(self, conn, settings):
        self.conn, self.settings, self.flags = conn, settings, []

    def flag(self, code, detail=""):
        self.flags.append(code)


def _subs(conn, settings, words=None, text=None):
    if words:
        ln = Line(id=0, start_ms=words[0].start_ms, end_ms=words[-1].end_ms, text=" ".join(w.text for w in words),
                  source="speech", words=[Word(start_ms=w.start_ms, end_ms=w.end_ms, text=w.text) for w in words])
    else:
        ln = Line(id=0, start_ms=0, end_ms=3000, text=text, source="speech")
    return save_subs(conn, settings.media_dir, Subtitles(lines=[ln]))


def test_node_flags_estimated_words_and_emits_json(conn, settings):
    ctx = Ctx(conn, settings)
    ref = _subs(conn, settings, text="The pump hits at midnight. Nobody notices until morning.")
    out = N.pick_fn(ctx, {"subtitles": ref}, N.PickParams())
    assert ctx.flags == ["WORDS_ESTIMATED"]
    body = json.loads(media.path_for(settings.media_dir, out["plans"]).read_text())
    assert body["plans"] and body["plans"][0]["take"] == 0


def test_node_real_words_no_flag_two_takes(conn, settings):
    ctx = Ctx(conn, settings)
    a, b = _subs(conn, settings, words=say(CLEAN)), _subs(conn, settings, words=say("Um " + CLEAN))
    out = N.pick_fn(ctx, {"subtitles": a, "subtitles_b": b}, N.PickParams(top_n=1))
    assert ctx.flags == []
    body = json.loads(media.path_for(settings.media_dir, out["plans"]).read_text())
    assert len(body["plans"]) == 1 and body["takes"] == ["subtitles", "subtitles_b"]


def test_node_params_forbid_extras():
    import pytest
    with pytest.raises(Exception):
        N.PickParams(weights={})
