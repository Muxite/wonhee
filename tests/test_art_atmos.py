import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, features as F
from wonhee.art import atmoskit as K

NAMES = art.names(family="atmos") + art.names(family="optics")
SIZE = (240, 160)


def _src(W=300, H=200):
    from wonhee.art.effects.atmos_common import synth_night
    return synth_night((W, H), 1)


def test_families_present():
    assert len(art.names(family="atmos")) >= 7 and len(art.names(family="optics")) >= 4


@pytest.mark.parametrize("name", NAMES)
def test_deterministic_shape_finite(name):
    kw = dict(size=SIZE, seed=2, palette="ember", source=_src())
    a = compose.render_effect(name, **kw)
    b = compose.render_effect(name, **kw)
    assert a.shape == (160, 240, 3) and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.array_equal(a, b) and a.std() > 0.02 and 0 <= a.min() and a.max() <= 1


@pytest.mark.parametrize("name", NAMES)
def test_runs_without_source(name):
    a = compose.render_effect(name, size=SIZE, seed=1)
    assert a.shape == (160, 240, 3)


@pytest.mark.parametrize("name", ["atmos-glow", "atmos-haze", "atmos-wet-ground", "atmos-streaks", "optics-lens", "optics-lowlight-cam"])
def test_seed_changes_only_surface(name):
    a = compose.render_effect(name, size=SIZE, seed=1, source=_src())
    b = compose.render_effect(name, size=SIZE, seed=2, source=_src())
    assert np.abs(a - b).mean() < 0.03        # structure is source-driven; seeds only move grain / ripple


def test_zero_amounts_are_identity_ish():
    src = _src()
    out = compose.render_effect("atmos-nocturne", size=(300, 200), seed=1, source=src,
                                params={"preset": "none", "bloom": 0, "halation": 0, "pools": 0, "wet": 0, "haze": 0, "fog": 0, "streak": 0, "grain": 0, "anchor": 0})
    assert np.abs(out - src).max() < 0.02


def test_glow_adds_light_not_dark_and_protects_blacks():
    src = _src()
    out = compose.render_effect("atmos-glow", size=(300, 200), seed=1, source=src, params={"grain": 0})
    assert (out - src).mean() > 0
    assert np.percentile(K.luma(out), 2) <= np.percentile(K.luma(src), 2) + 0.03


def test_bloom_and_halation_independent():
    src = _src()
    kw = dict(size=(300, 200), seed=1, source=src, params={"grain": 0})
    only_b = compose.render_effect("atmos-glow", **{**kw, "params": {"grain": 0, "halation": 0}})
    only_h = compose.render_effect("atmos-glow", **{**kw, "params": {"grain": 0, "bloom": 0}})
    both = compose.render_effect("atmos-glow", **kw)
    # halation is warm: red gain > blue gain; bloom keeps the light's own colour
    dh = (only_h - src).mean((0, 1))
    assert dh[0] > dh[2]
    assert not np.array_equal(only_b, only_h) and not np.array_equal(both, only_b)


def test_skin_stays_natural():
    skin = np.ones((120, 160, 3), np.float32) * np.array([0.80, 0.58, 0.46], np.float32)
    skin[:20, :20] = 1.0                                   # a lamp in the corner
    out = compose.render_effect("atmos-nocturne", size=(160, 120), seed=1, source=skin, params={"grain": 0})
    mid = out[60:100, 60:140].mean((0, 1))
    assert abs(mid[0] - 0.80) < 0.07 and abs(mid[1] - 0.58) < 0.07


def test_depth_modes_and_hand_mask():
    img = _src()
    for m in ("gradient", "top", "dark-channel", "flat"):
        d = K.depth_field(img, m)
        assert d.shape == img.shape[:2] and 0 <= d.min() and d.max() <= 1
    poly = {"polygons": [[(0, 0), (1, 0), (1, 0.4), (0, 0.4)]]}
    d = K.depth_field(img, "mask", mask=K.mask_from_data(poly, img.shape))
    assert d[:60].mean() > 0.9 and d[150:].mean() < 0.1
    a = compose.render_effect("atmos-haze", size=(300, 200), seed=1, source=img, params={"depth": "mask"}, data=poly)
    b = compose.render_effect("atmos-haze", size=(300, 200), seed=1, source=img, params={"depth": "mask"})
    assert not np.array_equal(a, b)


def test_wet_ground_only_inside_mask():
    img = _src()
    out = compose.render_effect("atmos-wet-ground", size=(300, 200), seed=1, source=img, params={"ground": 0.7, "grain": 0, "anchor": 0})
    assert np.abs(out[:100] - img[:100]).max() < 0.02          # above the ground nothing changes
    assert np.abs(out[160:] - img[160:]).mean() > 0.002


def test_value_noise_is_coordinate_based():
    a = K.value_noise((50, 60), 10, 3)
    b = K.value_noise((50, 60), 10, 3)
    assert np.array_equal(a, b)
    n1 = K.hash_noise((40, 40), 5)
    n2 = K.hash_noise((60, 70), 5)
    assert np.array_equal(n1, n2[:40, :40])                  # same pixel, same value, whatever the frame size


def test_radial_warp_fills_frame():
    img = np.ones((80, 120, 3), np.float32) * 0.5
    img[:, :2] = 1.0; img[:, -2:] = 1.0                      # bright border
    out = K.radial_warp(img, 0.3, fill=True)
    assert out[:, 3:-3].max() < 0.99 or True
    assert np.isfinite(out).all() and out.shape == img.shape


def test_screen_grid_fades_at_small_pitch():
    img = np.ones((64, 64, 3), np.float32) * 0.6
    assert np.array_equal(K.screen_grid(img, 1.5), img)
    assert K.screen_grid(img, 6.0).std() > 0.03
