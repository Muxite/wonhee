"""Synthesized sound effects: every sound renders clean, deterministically, at the requested rate, and round-trips
through a 16-bit WAV. Ported from slopcut's tests/test_audio.py (the sound-only tests; cue and ffmpeg tests come
with the cues port)."""
import wave

import numpy as np
import pytest

from wonhee.edit.sfx.dsp import SR
from wonhee.edit.sfx.sounds import SOUNDS, make, names, render, write_wav

EXPECTED = {"boom", "nuke", "shockwave", "glass", "debris", "sparks", "zap", "crackle", "rumble", "stutter",
            "whoosh", "riser", "suck", "rewind", "glitch", "tv_off", "shutter", "chime", "pop", "sub_drop"}
SUSTAINED = ("crackle", "rumble", "stutter")
RMS_FLOOR = 1e-3


def test_twenty_sounds():
    assert len(SOUNDS) == 20
    assert set(names()) == EXPECTED


@pytest.mark.parametrize("name", sorted(SOUNDS))
def test_sound_renders_clean(name):
    x, anchor = make(name, seed=3, dur=1.5)
    assert np.isfinite(x).all() and 0.05 * SR < len(x) < 8 * SR
    assert 0.3 < np.abs(x).max() <= 1.0
    assert 0 <= anchor <= len(x) / SR
    assert abs(x[-1]) < 0.02                       # faded out, no click at the end
    assert np.array_equal(x, make(name, seed=3, dur=1.5)[0])
    assert not np.array_equal(x, make(name, seed=4, dur=1.5)[0])


@pytest.mark.parametrize("name", sorted(SOUNDS))
def test_render_contract(name):
    y = render(name, seed=1)
    assert y.dtype == np.float32 and y.ndim == 1
    assert np.isfinite(y).all()
    assert float(np.abs(y).max()) <= 1.0
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    assert rms > RMS_FLOOR
    duration = len(y) / SR
    assert isinstance(duration, float) and 0.05 < duration < 8.0


@pytest.mark.parametrize("name", sorted(SOUNDS))
def test_render_is_deterministic(name):
    a, b = render(name, seed=42), render(name, seed=42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, render(name, seed=43))


def test_sustained_sounds_follow_requested_length():
    for name in SUSTAINED:
        assert len(make(name, 0, 3.0)[0]) / SR == pytest.approx(3.0, abs=0.05)
        assert len(render(name, 0, dur=3.0)) / SR == pytest.approx(3.0, abs=0.05)


@pytest.mark.parametrize("name", sorted(SOUNDS))
def test_other_sample_rate_keeps_duration(name):
    hi, lo = render(name, seed=5, sr=SR), render(name, seed=5, sr=22050)
    assert np.isfinite(lo).all() and float(np.abs(lo).max()) <= 1.0
    assert len(lo) / 22050 == pytest.approx(len(hi) / SR, abs=0.01)


def test_unknown_sound():
    with pytest.raises(ValueError, match="unknown sound") as err:
        make("kazoo")
    assert "boom" in str(err.value) and "sub_drop" in str(err.value)
    with pytest.raises(ValueError, match="unknown sound"):
        render("kazoo")


@pytest.mark.parametrize("name", sorted(SOUNDS))
def test_wav_round_trip(name, tmp_path):
    sr = 22050
    y = render(name, seed=9, sr=sr)
    path = tmp_path / f"{name}.wav"
    write_wav(path, y, sr)
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == sr
        assert w.getnframes() == len(y)
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32767
    assert np.allclose(pcm, np.clip(y.astype(np.float64), -1, 1), atol=1 / 32767 + 1e-9)
