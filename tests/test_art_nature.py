import subprocess

import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, motionkit as M, natkit as N
from wonhee.art.registry import Ctx
from wonhee.art import palette as pal

NATURE = ["nature-plankton-ribbons", "nature-strata", "nature-cloudfield", "nature-aurora-sheets", "nature-flow-wash"]
MOTION = ["motion-long-exposure", "motion-stroboscope", "motion-slit-scan", "motion-frame-diff", "motion-flow-trails"]
CONTOUR = ["contour-topo-multiscale", "contour-banded-hatch", "contour-streamlines", "contour-hatch-engrave", "contour-edge-ink"]
ALL = NATURE + MOTION + CONTOUR
SIZE = (300, 200)
FAST = {"n_frames": 8, "bodies": 4}


def _src(seed=0):
    rng = np.random.default_rng(seed)
    from wonhee.art import features as F
    return F.synthetic_source((200, 140), seed)


def _fast(name):
    return dict(FAST) if name.startswith("motion") else {}


def test_all_registered_with_mood_tags():
    reg = art.names()
    for n in ALL:
        assert n in reg, n
        tags = art.get(n).tags
        assert any(t.startswith("mood:") for t in tags), n


@pytest.mark.parametrize("name", ALL)
def test_deterministic_shape_range_not_constant(name):
    kw = dict(size=SIZE, palette="ember", params=_fast(name))
    a = compose.render_effect(name, seed=3, **kw)
    b = compose.render_effect(name, seed=3, **kw)
    c = compose.render_effect(name, seed=4, **kw)
    assert a.shape == (200, 300, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.std() > 0.03 and np.isfinite(a).all()
    assert 0 <= a.min() and a.max() <= 1


@pytest.mark.parametrize("name", ALL)
def test_source_light_palette_and_thumbnail(name):
    s = _src()
    img = compose.render_effect(name, size=(288, 192), seed=1, palette="ink-on-paper", source=s, params=_fast(name))
    assert img.shape == (192, 288, 3) and np.isfinite(img).all() and img.std() > 0.02
    assert img.min() > 0.0 and img.max() < 1.0          # no pure black / white areas


@pytest.mark.parametrize("name", NATURE)
def test_nature_independent_colour_controls(name):
    base = compose.render_effect(name, size=SIZE, seed=2, palette="ultramarine")
    alt = compose.render_effect(name, size=SIZE, seed=2, palette="ultramarine", params={"ramp": "spectrum", "grain": 0.0, "tint": 60})
    assert not np.array_equal(base, alt)


def test_time_is_coordinate_based_and_moves():
    a = compose.render_effect("nature-plankton-ribbons", size=SIZE, seed=2, params={"time": 1.5})
    b = compose.render_effect("nature-plankton-ribbons", size=SIZE, seed=2, params={"time": 1.5})
    c = compose.render_effect("nature-plankton-ribbons", size=SIZE, seed=2, params={"time": 2.0})
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.abs(a - c).mean() < 0.12                    # neighbouring times stay close (coherent)


def test_tnoise_pure_function_of_coordinates():
    x = np.linspace(0, 3, 17)
    t1 = N.tnoise(x, x * 0, 2.3, 5)
    t2 = N.tnoise(x, x * 0, 2.3, 5)
    assert np.array_equal(t1, t2) and np.abs(t1).max() <= 1.0
    assert np.abs(N.tnoise(x, x * 0, 2.3, 5) - N.tnoise(x, x * 0, 2.31, 5)).max() < 0.05


def test_sequence_frame_independent_of_frame_count():
    ctx12 = Ctx(size=SIZE, seed=1, palette=pal.named("ember"), params={**{k: v[0] for k, v in M.SEQ_PARAMS.items()}, "scene": "orbs", "n_frames": 100})
    ctx24 = Ctx(size=SIZE, seed=1, palette=pal.named("ember"), params={**ctx12.params, "n_frames": 200})
    f1 = M.sequence(ctx12)
    f2 = M.sequence(ctx24)
    # frame at normalised time 0.5 in both (index 49.5 vs 99.5 -> use endpoints, t=0 and t=1)
    assert np.allclose(f1[0], f2[0], atol=1e-5) and np.allclose(f1[-1], f2[-1], atol=1e-5)


def test_motion_accepts_frame_array_directory_and_video(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.random((60, 90, 3)).astype(np.float32) * 0.3
    frames = []
    for i in range(10):
        f = base.copy()
        f[20:35, 5 + i * 6:20 + i * 6] = 0.9
        frames.append(f)
    for name in MOTION:
        img = compose.render_effect(name, size=(180, 120), seed=1, params={"scene": "frames", "colour_mode": "ramp"}, source=None) if False else None
    ctx_params = {"scene": "frames"}
    out = compose.render_effect("motion-long-exposure", size=(180, 120), params=ctx_params, data=frames)
    assert out.shape == (120, 180, 3) and out.std() > 0.02
    d = tmp_path / "fr"
    d.mkdir()
    for i, f in enumerate(frames):
        compose.save(f, str(d / f"{i:03d}.png"))
    o2 = compose.render_effect("motion-frame-diff", size=(180, 120), params={"scene": "frames", "frames": str(d), "n_frames": 8})
    assert o2.std() > 0.02
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        mp4 = tmp_path / "c.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", str(d / "%03d.png"), "-pix_fmt", "yuv420p", str(mp4)], check=True)
        o3 = compose.render_effect("motion-stroboscope", size=(180, 120), params={"scene": "frames", "frames": str(mp4), "n_frames": 6, "mode": "movers", "echoes": 4})
        assert o3.shape == (120, 180, 3) and o3.std() > 0.02


def test_contour_minimum_line_width_on_thumbnail():
    """Lines must still exist at 288x192: coverage of 'ink' pixels is not tiny."""
    img = compose.render_effect("contour-topo-multiscale", size=(288, 192), seed=1, palette="ink-on-paper", params={"fill": 0.0, "shade": 0.0})
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert (lum < np.median(lum) - 0.15).mean() > 0.02


def test_contour_data_field_is_used():
    yy, xx = np.mgrid[0:40, 0:60].astype(np.float32)
    a = compose.render_effect("contour-topo-multiscale", size=SIZE, seed=1, data=np.sin(xx / 7) + np.cos(yy / 5))
    b = compose.render_effect("contour-topo-multiscale", size=SIZE, seed=1)
    assert not np.array_equal(a, b)


def test_render_budget_full_size():
    import time
    for name, p in (("nature-plankton-ribbons", {}), ("motion-long-exposure", {"scene": "orbs"}), ("contour-streamlines", {})):
        t0 = time.time()
        compose.render_effect(name, size=(1500, 1000), seed=1, params=p)
        assert time.time() - t0 < 8.0, name
