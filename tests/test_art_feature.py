import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, features as F

FEATURE = [n for n in (art.names(family="feature"))]


def _stripes(W=300, H=200, period=20, vertical=True):
    yy, xx = np.mgrid[0:H, 0:W]
    v = (np.sin(2 * np.pi * (xx if vertical else yy) / period) > 0).astype(np.float32) * 0.6 + 0.2
    return np.dstack([v, v * 0.9, v * 0.8]).astype(np.float32)


def _src(W=240, H=160):
    return F.synthetic_source((W, H), 3)


def test_family_has_at_least_twelve_effects():
    assert len(FEATURE) >= 12


@pytest.mark.parametrize("name", FEATURE)
def test_effect_deterministic_shape_nonconstant(name):
    src = _src(320, 200)
    kw = dict(size=(240, 160), seed=2, palette="ember", source=src)
    a = compose.render_effect(name, **kw)
    b = compose.render_effect(name, **kw)
    assert a.shape == (160, 240, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert a.std() > 0.02 and 0 <= a.min() and a.max() <= 1
    # no pure black / pure white areas
    assert (a.max(-1) < 0.02).mean() < 0.01 and (a.min(-1) > 0.985).mean() < 0.01


@pytest.mark.parametrize("name", FEATURE)
def test_effect_runs_without_source_and_follows_palette(name):
    a = compose.render_effect(name, size=(180, 120), seed=1, palette="ember")
    b = compose.render_effect(name, size=(180, 120), seed=1, palette="ultramarine")
    assert a.shape == (120, 180, 3) and not np.array_equal(a, b)


def test_orientation_follows_stripes():
    L = F.luminance(_stripes(vertical=True))
    theta, coh = F.structure_orientation(L, rho=6)
    # vertical stripes: structure runs along y (theta ~ +-90 deg)
    c = np.abs(np.sin(theta[40:-40, 40:-40]))
    assert c.mean() > 0.9 and coh[40:-40, 40:-40].mean() > 0.8


def test_detect_grid_finds_period():
    g = F.detect_grid(F.luminance(_stripes(period=20)))
    assert abs(g["period_x"] - 20) <= 1 or abs(g["period_x"] - 40) <= 1


def test_dominant_hue_and_palette():
    img = np.ones((60, 80, 3), np.float32) * np.array([0.8, 0.2, 0.1], np.float32)
    h = F.dominant_hue(img)
    assert 15 < h < 50
    pal = F.dominant_palette(np.concatenate([img, img * 0 + np.array([0.1, 0.2, 0.8], np.float32)], 0), k=2)
    assert len(pal) == 2 and abs(pal[0]["weight"] - 0.5) < 0.1


def test_feature_crop_finds_detail_not_centre():
    img = np.full((300, 450, 3), 0.5, np.float32)
    rng = np.random.default_rng(0)
    img[200:290, 330:440] = rng.random((90, 110, 3)).astype(np.float32)     # detail in the lower-right corner
    x0, y0, x1, y1 = F.feature_crop(img, 1.5, min_frac=0.3, max_frac=0.5, steps=3)
    assert x0 + (x1 - x0) / 2 > 300 and y0 + (y1 - y0) / 2 > 180


def test_detail_crops_non_overlapping():
    cr = F.detail_crops(F.synthetic_source((600, 400), 1), 6, 1.5, 0.25)
    assert len(cr) == 6
    for i, a in enumerate(cr):
        for b in cr[i + 1:]:
            assert abs(a[0] - b[0]) >= 0.8 * (a[2] - a[0]) or abs(a[1] - b[1]) >= 0.8 * (a[3] - a[1])


def test_superpixels_and_region_weights():
    img = _src(200, 140)
    lab = F.superpixels(img, n=40)
    assert lab.shape == (140, 200) and 20 < len(np.unique(lab)) < 80
    for mode in ("detail", "tone", "cells", "superpixel", "diagonal", "columns", "rows"):
        w = F.region_weights(img, 3, mode, 0.02, 1)
        assert w.shape == (3, 140, 200) and np.allclose(w.sum(0), 1, atol=1e-4)
    hard = F.region_weights(img, 2, "columns", 0.0)
    assert set(np.unique(hard)) <= {0.0, 1.0}


def test_text_mask_on_synthetic_text_lines():
    img = np.full((200, 300, 3), 0.95, np.float32)
    rng = np.random.default_rng(1)
    for y in (40, 70, 100):
        for x in range(30, 250, 9):
            img[y:y + 12, x:x + 5] = 0.1 + 0.1 * rng.random()
    m = F.text_mask(img)
    assert m[40:52, 30:250].mean() > 0.5 and m[150:, :].mean() < 0.05


def test_tile_grid_covers_frame():
    b = F.tile_grid((300, 200), 3, 2, 4)
    assert len(b) == 6 and b[0][0] == 4 and b[-1][2] <= 300 - 3


def test_fit_cover_crops_by_detail():
    src = np.full((400, 400, 3), 0.5, np.float32)
    src[250:390, 20:380] = np.random.default_rng(0).random((140, 360, 3)).astype(np.float32)
    out = F.fit_cover(src, (300, 200), "feature")
    assert out.shape == (200, 300, 3) and out.std() > 0.1


def test_region_effect_mixes_two_effects():
    kw = dict(size=(240, 160), seed=0, palette="ember", source=_src())
    a = compose.render_effect("feature-regions", params={"effects": "feature-halftone-screen,feature-contour-relief", "mode": "columns", "soft": 0.0}, **kw)
    h = compose.render_effect("feature-halftone-screen", **kw)
    assert np.allclose(a[:, :60], h[:, :60], atol=1e-4) and not np.allclose(a[:, 180:], h[:, 180:], atol=1e-2)
