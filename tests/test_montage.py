"""Montage engine (wonhee.edit.montage): seeded plans, hand-editable JSON, xfade rendering.

Ported from slopcut's tests/test_video.py. Tests that run ffmpeg skip when it is missing; frames are tiny and
clips short so the whole transition table stays cheap.
"""
import json
import shutil
import subprocess

import pytest
from PIL import Image

from wonhee.edit.montage.effects import EFFECTS, STYLES
from wonhee.edit.montage.plan import Still, build_plan, dump_plan, load_plan, total_length
from wonhee.edit.montage.render import render, timeline
from wonhee.edit.montage.transitions import ALL, BUILTIN, POOLS
from wonhee.edit.montage.transitions_custom import CUSTOM
from wonhee.media import probe

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
SMALL = (270, 480)
TINY = (90, 160)   # 9:16, small enough that ~70 transitions render in seconds


@pytest.fixture(autouse=True)
def montage_cache(tmp_path, monkeypatch):
    """Segments cache under the test's tmp dir, never the real home."""
    monkeypatch.setenv("WONHEE_MONTAGE_CACHE_DIR", str(tmp_path / "segcache"))


@pytest.fixture
def imgs(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for name, colour in (("a", (200, 40, 40)), ("b", (40, 200, 40)), ("c", (40, 40, 200))):
        Image.new("RGB", (120, 200), colour).save(d / f"{name}.png")
    return d


def stills(imgs, overrides=None):
    overrides = overrides or {}
    return [Still(p, overrides.get(p.name, {})) for p in sorted(imgs.glob("*.png"))]


def plan_for(imgs, overrides=None, **kw):
    opts = dict(length=6.0, beat=None, size=SMALL, fps=30, style="hype", seed=7)
    opts.update(kw)
    return build_plan(stills(imgs, overrides), **opts)


# --- plan ---------------------------------------------------------------------------------------------------

def test_plan_is_deterministic_per_seed(imgs):
    assert plan_for(imgs) == plan_for(imgs)
    assert dump_plan(plan_for(imgs)) == dump_plan(plan_for(imgs))   # byte-identical text, not just equal dicts
    assert plan_for(imgs, seed=7) != plan_for(imgs, seed=8)


@pytest.mark.parametrize("style", sorted(STYLES))
def test_every_style_is_deterministic(imgs, style):
    assert dump_plan(plan_for(imgs, style=style, seed=3)) == dump_plan(plan_for(imgs, style=style, seed=3))


def test_plan_hits_requested_length(imgs):
    for length in (4.0, 6.0, 15.0):
        assert total_length(plan_for(imgs, length=length)) == pytest.approx(length, abs=0.02)


def test_beat_mode_and_too_short(imgs):
    p = plan_for(imgs, length=None, beat=2.0)
    assert all(i["duration"] == 2.0 for i in p["items"])
    with pytest.raises(ValueError, match="too short"):
        plan_for(imgs, length=0.5)
    with pytest.raises(ValueError):
        plan_for(imgs, length=None, beat=None)


def test_duration_and_effect_overrides(imgs):
    p = plan_for(imgs, overrides={"a.png": {"duration": 3, "effect": "shake"}}, length=8.0)
    assert p["items"][0]["duration"] == 3 and p["items"][0]["effect"] == "shake"
    assert total_length(p) == pytest.approx(8.0, abs=0.02)


def test_unknown_style_effect_transition_fx_raise(imgs, tmp_path):
    with pytest.raises(ValueError, match="unknown style"):
        plan_for(imgs, style="nope")
    with pytest.raises(ValueError, match="unknown effect"):
        plan_for(imgs, overrides={"a.png": {"effect": "nope"}})
    with pytest.raises(ValueError, match="unknown fx"):
        plan_for(imgs, fx=["nope"])
    plan = plan_for(imgs)
    plan["items"][0]["transition"]["type"] = "nope"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="unknown transition"):
        load_plan(path)
    with pytest.raises(ValueError, match="unknown transition"):
        render(plan, tmp_path / "x.mp4")   # a hand-built plan is checked before ffmpeg starts


def test_pools_only_name_known_transitions():
    for style, names in POOLS.items():
        assert names and set(names) <= set(ALL), style
    assert set(CUSTOM) <= set(ALL) and set(BUILTIN) <= set(ALL)


def test_caption_is_normalised_and_checked(imgs):
    p = plan_for(imgs, texts=["hello"])
    assert p["items"][0]["text"] == {"content": "hello", "position": "bottom"}
    with pytest.raises(ValueError):
        plan_for(imgs, texts=[{"content": "x", "layer": "sideways"}])
    with pytest.raises(ValueError):
        plan_for(imgs, texts=[{"content": "x", "nokey": 1}])


# --- render -------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_render_matches_plan(imgs, tmp_path):
    out = render(plan_for(imgs, sfx=False), tmp_path / "r.mp4")   # no sound cues: video-only output
    info = probe(out)
    assert (info.width, info.height) == SMALL
    assert info.fps == pytest.approx(30)
    assert info.duration_ms / 1000 == pytest.approx(6.0, abs=0.1)
    codec = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                            str(out)], capture_output=True, text=True).stdout.split()
    assert codec == ["h264"]


@needs_ffmpeg
def test_plan_json_round_trips_and_rerenders(imgs, tmp_path):
    plan = plan_for(imgs, length=3.0, size=TINY, texts=["hi", None, "there"])
    path = tmp_path / "plan.json"
    path.write_text(dump_plan(plan))
    loaded = load_plan(path)
    assert loaded == plan
    assert dump_plan(loaded) == path.read_text()
    out = render(loaded, tmp_path / "again.mp4")
    assert probe(out).duration_ms / 1000 == pytest.approx(3.0, abs=0.1)
    # unchanged plan, warm cache: the second render reuses every segment
    logs = []
    render(loaded, tmp_path / "third.mp4", log=logs.append)
    assert any("3 cached, 0 to render" in line for line in logs)


@needs_ffmpeg
def test_every_effect_renders(imgs, tmp_path):
    effects = sorted(EFFECTS)
    for start in range(0, len(effects), 3):
        chunk = effects[start:start + 3]
        # Fewer images than effects: build a plan by hand so every name gets exercised.
        base = plan_for(imgs, length=3.0, size=TINY)
        items = [dict(base["items"][k % 3], effect=e, duration=0.6) for k, e in enumerate(chunk)]
        for it in items[:-1]:
            it["transition"] = {"type": "fade", "duration": 0.2}
        items[-1].pop("transition", None)
        assert render(dict(base, items=items), tmp_path / f"e{start}.mp4").exists()


@needs_ffmpeg
@pytest.mark.parametrize("name", ALL)
def test_transition_renders_two_clips(imgs, tmp_path, name):
    base = plan_for(imgs, length=1.0, size=TINY)
    items = [dict(base["items"][0], duration=0.5, transition={"type": name, "duration": 0.3}),
             dict(base["items"][1], duration=0.5)]
    items[1].pop("transition", None)
    plan = dict(base, items=items)
    out = render(plan, tmp_path / "t.mp4")
    assert probe(out).duration_ms / 1000 == pytest.approx(total_length(plan), abs=0.1)


@needs_ffmpeg
def test_audio_is_muxed_and_trimmed(imgs, tmp_path):
    tone = tmp_path / "tone.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                    str(tone)], check=True)
    out = render(plan_for(imgs, size=TINY, sfx=False), tmp_path / "a.mp4", audio=tone)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(out)], capture_output=True, text=True).stdout.split()
    assert sorted(streams) == ["audio", "video"]
    assert probe(out).duration_ms / 1000 == pytest.approx(6.0, abs=0.1)


@needs_ffmpeg
def test_clips_land_where_the_plan_says(tmp_path):
    # Solid-colour clips through custom + builtin transitions: the middle of each clip must show
    # its own colour at the plan's time. (Transition segments once came out yuv444p and the concat
    # silently dropped them, sliding every later clip earlier.)
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    items = []
    for k, c in enumerate(colours):
        src = tmp_path / f"{k}.png"
        Image.new("RGB", (64, 64), c).save(src)
        items.append({"src": str(src), "effect": "zoom_in", "duration": 0.6 + 0.1 * k,
                      "transition": {"type": ["spin", "wipeleft", "glitch_slice", "fade"][k % 4], "duration": 0.22}})
    items[-1].pop("transition")
    plan = {"size": [180, 320], "fps": 30, "seed": 1, "sfx": False, "items": items}
    out = render(plan, tmp_path / "c.mp4")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = [raw[i:i + 180 * 320 * 3] for i in range(0, len(raw), 180 * 320 * 3)]
    n, k = timeline(plan)
    start = 0
    for i, c in enumerate(colours):
        k_in, k_out = (k[i - 1] if i else 0), (k[i] if i < len(k) else 0)
        mid = frames[start + (k_in + n[i] - k_out) // 2]
        avg = [sum(mid[j::3]) / (180 * 320) for j in range(3)]
        assert all(abs(a - b) < 40 for a, b in zip(avg, c)), (i, avg, c)
        start += n[i] - k_out


@needs_ffmpeg
def test_caption_is_drawn(imgs, tmp_path):
    plan = plan_for(imgs, length=2.0, size=SMALL, style="calm", texts=[None, None, None])
    plan["items"][0]["text"] = {"content": "HELLO", "position": "center", "fade": 0}
    plain = tmp_path / "plain.mp4"
    capt = tmp_path / "capt.mp4"
    render(dict(plan, items=[{k: v for k, v in plan["items"][0].items() if k != "text"}] + plan["items"][1:]),
           plain)
    render(plan, capt)

    def first_frame(path):
        return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout

    assert first_frame(plain) != first_frame(capt)


@needs_ffmpeg
def test_inputs_use_the_local_only_whitelist(imgs, tmp_path, monkeypatch):
    from wonhee.edit.montage import render as render_mod
    seen = []
    real = render_mod.run_ffmpeg

    def spy(cmd, strict=False):
        seen.append(list(cmd))
        return real(cmd, strict)

    monkeypatch.setattr(render_mod, "run_ffmpeg", spy)
    render(plan_for(imgs, length=2.0, size=TINY, texts=["x", None, None]), tmp_path / "w.mp4")
    assert seen
    for cmd in seen:
        # the whitelist is an input option: it must appear once per input, ahead of that input's -i
        assert cmd.count("-protocol_whitelist") == cmd.count("-i"), cmd
        first = cmd.index("-protocol_whitelist")
        assert cmd[first:first + 2] == ["-protocol_whitelist", "file,pipe"] and first < cmd.index("-i"), cmd
