"""Montage + fx + sfx together: a seeded plan with fx and sound cues renders to one video with an audio track.

The three engines were ported separately (edit/montage, edit/montage/fx, edit/sfx); these tests are the seams
between them: plan validation against both catalogs, the fx Ctx the renderer builds, and cue times against the
frames the renderer actually cuts. Cue unit tests are ported from slopcut's tests/test_audio.py.
"""
import json
import shutil
import wave

import numpy as np
import pytest
from PIL import Image

from wonhee.edit.montage.fx import FX, Ctx, expand, slot_seed
from wonhee.edit.montage.plan import Still, build_plan, dump_plan, load_plan, total_length
from wonhee.edit.montage.render import render, timeline
from wonhee.edit.montage.transitions import ALL
from wonhee.edit.sfx.cues import FX_RULES, _hash2, cues_for, mix, render_track, transition_sound
from wonhee.edit.sfx.dsp import SR
from wonhee.edit.sfx.sounds import SOUNDS
from wonhee.media import has_audio, probe

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
TINY = (90, 160)


@pytest.fixture(autouse=True)
def montage_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("WONHEE_MONTAGE_CACHE_DIR", str(tmp_path / "segcache"))


@pytest.fixture
def stills4(tmp_path):
    out = []
    for k, colour in enumerate([(200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40)]):
        p = tmp_path / f"s{k}.png"
        Image.new("RGB", (120, 200), colour).save(p)
        out.append(Still(p))
    return out


def hype_plan(stills, **kw):
    opts = dict(length=4.0, beat=None, size=TINY, fps=30, style="hype", seed=11, fx=["nuke"])
    opts.update(kw)
    return build_plan(stills, **opts)


def read_wav(path):
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getframerate()) == (2, SR)
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def _plan(items, **kw):
    return {"size": [270, 480], "fps": 30, "seed": 1, "items": items, **kw}


# --- cues (ported from slopcut) --------------------------------------------------------------------------------

def test_rules_only_name_real_fx_sounds_and_transitions_map():
    assert set(FX_RULES) <= set(FX)
    assert transition_sound("wipeleft")[0] == "whoosh"
    assert transition_sound("boom_cut")[0] == "boom" and transition_sound("fade") is None
    assert all(transition_sound(t) is None or transition_sound(t)[0] in SOUNDS for t in ALL)
    # every sound a rule can emit exists (lightning is a function rule; its sound is "zap")
    named = {c.sound for n in FX_RULES for c in cues_for(_plan(
        [{"src": "a", "effect": "zoom_in", "duration": 3.0, "fx": [n]}], sfx_max=1))}
    assert named <= set(SOUNDS)


def test_cue_timing_tracks_the_timeline():
    items = [{"src": "a", "effect": "zoom_in", "duration": 2.0, "transition": {"type": "wipeleft", "duration": 0.4}},
             {"src": "b", "effect": "punch", "duration": 2.0, "fx": ["nuke", "bass_boost"]}]
    cues = cues_for(_plan(items))
    assert next(c for c in cues if c.sound == "whoosh").t == pytest.approx(1.6 + 0.2)   # transition midpoint
    assert next(c for c in cues if c.sound == "nuke").t == pytest.approx(1.6)           # clip 2 starts with the xfade
    thumps = [c.t for c in cues if c.sound == "sub_drop"]
    assert thumps[:4] == pytest.approx([1.6, 1.6, 2.1, 2.6])                            # punch + bass_boost every 0.5 s


def test_fx_sounds_capped_by_priority_and_item_override():
    items = [{"src": "a", "effect": "zoom_in", "duration": 2.0, "fx": ["sparkles", "fire", "nuke", "debris"]}]
    assert {c.sound for c in cues_for(_plan(items))} == {"nuke", "debris"}
    items[0]["sfx"] = ["pop", "glass"]
    assert [c.sound for c in cues_for(_plan(items))] == ["pop", "glass"]


def test_sfx_off_and_silent_plans_make_no_track(tmp_path):
    items = [{"src": "a", "effect": "zoom_in", "duration": 2.0, "fx": ["nuke"]}]
    assert render_track(_plan(items, sfx=False), tmp_path / "a.wav") is None
    assert render_track(_plan([dict(items[0], fx=[])]), tmp_path / "b.wav") is None
    assert render_track(_plan(items), tmp_path / "c.wav").exists()


def test_mix_is_limited_and_sized():
    items = [{"src": "a", "effect": "punch", "duration": 3.0, "fx": ["nuke", "explosion", "bass_boost"]}]
    m = mix(_plan(items), gain=4.0)
    assert m.shape == (2, 3 * SR) and np.abs(m).max() <= 0.9


@needs_ffmpeg
def test_lightning_zaps_land_on_ffmpeg_strike_frames():
    import subprocess
    fps, n, seed = 30, 120, 123457
    frag = expand("lightning", Ctx(32, 32, fps, n, seed), "0:v", "out", "p")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=gray:s=32x32:r={fps}:d={n / fps}",
                          "-filter_complex", frag, "-map", "[out]", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    lum = np.frombuffer(raw, np.uint8).reshape(-1, 32 * 32).mean(1)
    seen = {f // 2 for f in np.flatnonzero(np.abs(lum - np.median(lum)) > 5)}
    assert seen == {p for p in range(n // 2) if _hash2(p, seed % 997) > 0.88}


# --- plan time -------------------------------------------------------------------------------------------------

def test_unknown_sound_or_fx_fails_at_plan_time(stills4, tmp_path):
    with pytest.raises(ValueError, match="unknown sound"):
        build_plan([Still(stills4[0].path, {"sfx": ["kazoo"]})], length=2.0, beat=None, size=TINY, fps=30,
                   style="hype", seed=1)
    with pytest.raises(ValueError, match="unknown fx"):
        hype_plan(stills4, fx=["nope"])
    plan = hype_plan(stills4)
    plan["items"][0]["sfx"] = ["kazoo"]
    path = tmp_path / "p.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="unknown sound"):
        load_plan(path)
    with pytest.raises(ValueError, match="unknown sound"):
        render(plan, tmp_path / "x.mp4")          # checked before any ffmpeg starts
    plan["items"][0]["sfx"] = []
    plan["items"][1]["fx"] = ["nope"]
    with pytest.raises(ValueError, match="unknown fx"):
        render(plan, tmp_path / "y.mp4")
    assert not (tmp_path / "x.mp4").exists() and not (tmp_path / "y.mp4").exists()


def test_slot_seed_is_what_the_renderer_and_cues_share():
    # Ctx's 4th field is the exact frame count the clip is cut to (render passes n); slot_seed is the 5th.
    assert Ctx(90, 160, 30, 60, 5).dur == 2.0
    assert slot_seed(3, 1, 2) == slot_seed(3, 1, 2) != slot_seed(3, 1, 3)


# --- render ----------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_hype_plan_renders_video_with_audio_and_replays(stills4, tmp_path):
    plan = hype_plan(stills4)
    assert any(it.get("fx") for it in plan["items"]) and plan["sfx"] is True
    out = render(plan, tmp_path / "a.mp4")
    info = probe(out)
    assert (info.width, info.height) == TINY
    assert has_audio(out)
    assert info.duration_ms / 1000 == pytest.approx(total_length(plan), abs=0.1)

    again = hype_plan(stills4)
    assert dump_plan(again) == dump_plan(plan)                       # same seed, same plan text
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    assert render_track(plan, a) is not None and render_track(again, b) is not None
    assert np.array_equal(read_wav(a), read_wav(b))                  # same seed, same audio samples
    assert read_wav(a).any()


@needs_ffmpeg
def test_different_seed_changes_the_audio(stills4, tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    render_track(hype_plan(stills4, seed=11), a)
    render_track(hype_plan(stills4, seed=12), b)
    assert not np.array_equal(read_wav(a), read_wav(b))


@needs_ffmpeg
def test_sfx_off_renders_no_audio_stream(stills4, tmp_path):
    out = render(hype_plan(stills4, sfx=False), tmp_path / "q.mp4")
    assert not has_audio(out)


@needs_ffmpeg
def test_cues_land_within_a_frame_of_the_visual_hit(stills4, tmp_path):
    """Clip i's picture starts at frame sum(n[:i]) - sum(k[:i]); a clip's fx hit sound must land within one frame."""
    plan = hype_plan(stills4, fx=["nuke"])
    for it in plan["items"]:
        it.pop("sfx", None)
    n, k = timeline(plan)
    fps = plan["fps"]
    cues = cues_for(plan)
    start_frame = 0
    for i, it in enumerate(plan["items"]):
        if "nuke" in (it.get("fx") or []):
            hit_frame = start_frame            # nuke's flash is its first frame (offset 0)
            near = [c for c in cues if c.sound == "nuke" and abs(c.t * fps - hit_frame) <= 1.0]
            assert near, (i, hit_frame, [c.t * fps for c in cues if c.sound == "nuke"])
        start_frame += n[i] - (k[i] if i < len(k) else 0)
    # and the wav the renderer mixes in is as long as the picture, so the cues keep their place
    wav = tmp_path / "t.wav"
    render_track(plan, wav)
    assert len(read_wav(wav)) / 2 / SR == pytest.approx(total_length(plan), abs=1 / fps)
