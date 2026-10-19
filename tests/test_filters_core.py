import json

import cv2
import numpy as np
import pytest

from tests.test_filters import LEAF, SKIN, SKY, chroma, hue, patches
from tests.test_vibe import day, night
from wonhee.edit import filters as F
from wonhee.edit import fx
from wonhee.edit.skin import Undertone, apply_undertone


def scene(seed=0, size=(160, 240)) -> np.ndarray:
    """A balanced scene with full tonal range and some colour."""
    rng = np.random.default_rng(seed)
    h, w = size
    g = np.tile(np.linspace(5, 250, w, dtype=np.float32), (h, 1))
    im = np.stack([g * 0.95, g, g * 1.02], -1) + rng.normal(0, 3, (h, w, 3))
    im[: h // 6, w // 2: w // 2 + w // 8] = (60, 150, 70)  # a leaf; the gradient keeps its full tonal range
    return np.clip(im, 0, 255).astype(np.uint8)


def cast_b(im) -> float:
    lab = cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return float(lab[..., 2][im.shape[0] // 3:].mean())


def test_normalize_is_near_identity_on_a_balanced_photo():
    im = scene()
    out, notes = F.normalize(im, 1.0)
    assert np.abs(out.astype(int) - im).mean() < 3 and notes["graded"] is False
    json.dumps(notes)


def test_normalize_removes_a_strong_cast():
    im = scene()
    warm = np.clip(im.astype(np.float32) * (0.7, 1.0, 1.25), 0, 255).astype(np.uint8)  # BGR: orange cast
    out, notes = F.normalize(warm, 1.0)
    assert abs(cast_b(out)) < abs(cast_b(warm)) - 3
    assert notes["wb_shift"][1] < -2 and abs(notes["wb_shift"][1]) <= 6.001


def test_low_key_is_left_dark_but_underexposure_is_lifted():
    im = scene()
    low_key = np.full_like(im, 25)  # a dark studio backdrop with a bright face: meant to be dark
    low_key[40:120, 60:140] = SKIN
    _, notes = F.normalize(low_key, 1.0)
    assert notes["exposure"] == pytest.approx(0, abs=0.02)
    under = (im.astype(np.float32) * 0.35).astype(np.uint8)  # the whole frame dull, highlights included
    _, notes = F.normalize(under, 1.0)
    assert notes["exposure"] > 0.1


def test_graded_photo_is_corrected_less():
    im = scene()
    faded = (im.astype(np.float32) * 0.7 + 60).astype(np.uint8)  # lifted blacks
    faded = np.clip(faded.astype(np.float32) * (0.8, 1.0, 1.2), 0, 255).astype(np.uint8)
    plain = np.clip(im.astype(np.float32) * (0.8, 1.0, 1.2), 0, 255).astype(np.uint8)
    out, notes = F.normalize(faded, 1.0)
    _, pnotes = F.normalize(plain, 1.0)
    assert notes["graded"] is True and pnotes["graded"] is False
    assert abs(notes["wb_shift"][1]) < abs(pnotes["wb_shift"][1])


@pytest.mark.parametrize("im", [np.zeros((50, 60, 3), np.uint8), np.full((50, 60, 3), 255, np.uint8),
                                np.full((1, 1, 3), 120, np.uint8), np.random.default_rng(1).integers(
                                    0, 255, (3, 3, 3), dtype=np.uint8),
                                np.full((40, 40), 100, np.uint8)])
def test_normalize_is_safe_on_degenerate_images(im):
    out, notes = F.normalize(im, 1.0)
    assert out.shape == im.shape and out.dtype == np.uint8
    assert np.isfinite(notes["exposure"]) and all(np.isfinite(notes["wb_shift"]))
    assert F.apply(im, F.Filter(), auto=1.0).shape == im.shape if im.ndim == 3 else True


def test_apply_auto_zero_matches_the_old_pipeline():
    im = night(3)
    f = F.Filter(contrast=0.3, glow=0.4, halation=0.3, vignette=0.3, light_leak=0.4, aberration=0.3, clarity=0.2)
    x = F.point(im.astype(np.float32) / 255, f)
    x = F._aberration(F._halation(F._glow(F._sharpen(F._clarity(x, f.clarity), 0), f.glow), f.halation), f.aberration)
    x = F._leak(x, f.light_leak, f.leak_hue, f.leak_side)
    old = fx.vignette(np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8), f.vignette)
    assert np.array_equal(F.apply(im, f), old) and np.array_equal(F.apply(im, f, auto=0.0), old)


def test_ranges_only_touch_their_own_hue():
    f = F.Filter(skin_protect=0, ranges={"green": F.RangeAdj(sat=-1)})
    out = F.apply(patches(), f)
    assert chroma(out[60, 300]) < chroma(LEAF) * 0.5
    assert chroma(out[60, 180]) == pytest.approx(chroma(SKY), rel=0.1)
    rot = F.apply(patches(), F.Filter(skin_protect=0, ranges={"blue": {"hue": 1}}))
    assert abs(hue(rot[60, 180]) - hue(SKY)) > 8 and abs(hue(rot[60, 300]) - hue(LEAF)) < 3
    grey = np.full((8, 8, 3), 128, np.uint8)
    lum = F.apply(grey, F.Filter(ranges={c: {"lum": 1, "hue": 1} for c in F.RANGE_DEG}))
    assert np.abs(lum.astype(int) - grey).max() <= 2  # greys untouched
    bright = F.apply(patches(), F.Filter(skin_protect=0, ranges={"green": {"lum": 1}}))
    assert bright[60, 300].mean() > patches()[60, 300].mean() + 5


def test_greens_alias_matches_the_green_range():
    a = F.apply(patches(), F.Filter(greens=-0.8, blues=0.6, skin_protect=0))
    b = F.apply(patches(), F.Filter(ranges={"green": {"sat": -0.8}, "blue": {"sat": 0.6}}, skin_protect=0))
    assert np.array_equal(a, b)
    assert "ranges" not in F.NUMERIC and F.Filter(ranges={"red": {"hue": 0.2}}).model_dump()["ranges"]
    with pytest.raises(Exception):
        F.RangeAdj(bogus=1)


def test_scaled_scales_ranges_mono_and_bounds():
    f = F.Filter(mono=True, ranges={"red": {"sat": 0.8}}, contrast=0.6)
    assert not F.scaled(f, 0.2).mono and F.scaled(f, 1.0).mono
    assert F.scaled(f, 0.5).ranges["red"].sat == pytest.approx(0.4)
    assert F.scaled(f, 2).ranges["red"].sat == 1.0
    assert F._bounds("tint") == (-1.0, 1.0)
    im = night(1)
    assert np.abs(F.apply(im, f, intensity=0).astype(int) - im).max() <= 2


def test_cube_picks_up_ranges():
    plain = F.cube(F.Filter(skin_protect=0)).splitlines()
    ranged = F.cube(F.Filter(skin_protect=0, ranges={"green": {"sat": -1}})).splitlines()
    assert len(plain) == len(ranged) and plain != ranged


def test_grain_differs_between_photos_and_is_reproducible():
    a, b = day(), night(2)
    na, nb = (fx.grain(x, 0.5).astype(int) - x for x in (a, b))
    assert np.array_equal(fx.grain(a, 0.5), fx.grain(a, 0.5))
    assert not np.array_equal(na[:100, :100], nb[:100, :100])
    assert np.array_equal(fx.grain(a, 0.5, seed=7), fx.grain(a, 0.5, seed=7))
    assert not np.array_equal(fx.grain(a, 0.5, seed=7), fx.grain(a, 0.5, seed=8))


def test_grain_is_weaker_in_shadows_and_highlights_than_mids():
    h = 200
    std = {}
    for name, v in (("shadow", 6), ("mid", 128), ("high", 252)):
        im = np.full((h, h, 3), v, np.uint8)
        std[name] = (fx.grain(im, 0.8, seed=5).astype(int) - im).std()
    assert std["shadow"] < std["mid"] * 0.6 and std["high"] < std["mid"] * 0.6


def test_undertone_moves_skin_not_sky_or_leaves():
    im = patches()
    for tone in ("cool", "warm", "rose", "gold", "neutral"):
        out = apply_undertone(im, Undertone(tone=tone, strength=1))
        assert np.array_equal(out[:, 120:], im[:, 120:])
        assert np.abs(out[:, :120].astype(int) - im[:, :120]).max() > 0 or tone == "neutral"
    warm, cool = (apply_undertone(im, Undertone(tone=t, strength=1))[60, 60] for t in ("warm", "cool"))
    assert hue(warm) > hue(cool) + 3
    assert np.array_equal(apply_undertone(im, Undertone(strength=0)), im)
    assert not np.array_equal(F.apply(im, F.Filter(), undertone=Undertone(tone="rose", strength=1)), F.apply(im, F.Filter()))


def test_checks_survives_extreme_aspect_and_aberration_runs_on_views():
    src = np.full((2, 3000, 3), 100, np.uint8)
    assert F.checks(src, src)["clipped"] == [0.0, 0.0]
    assert F.apply(night(1, (60, 80)), F.Filter(aberration=1)).shape == (60, 80, 3)


def test_big_images_take_the_downscaled_path(monkeypatch):
    im = night(4, (200, 300))
    f = F.Filter(glow=0.5, halation=0.3, vignette=0.4, light_leak=0.3)
    ref = F.apply(im, f)
    monkeypatch.setattr(F, "BIG_PX", 1000)
    monkeypatch.setattr(F, "STRIP_PX", 5000)
    big = F.apply(im, f)
    assert big.shape == ref.shape and np.abs(big.astype(int) - ref).mean() < 4
    out, _ = F.normalize(scene(), 1.0)
    assert np.array_equal(out, F.normalize(scene(), 1.0)[0])
