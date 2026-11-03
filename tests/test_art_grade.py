import time

import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, grading as G, features as F

G.ensure_loaded()
NAMES = sorted(G.LOOKS)


def _photo(w=240, h=160):
    return F.synthetic_source((w, h), 3)


def test_library_size_and_registration():
    assert len(NAMES) >= 30
    for n in NAMES:
        assert "grade-" + n in art.names("grade")


@pytest.mark.parametrize("name", NAMES)
def test_deterministic_shape_range(name):
    src = _photo()
    a = compose.render_effect("grade-" + name, size=(160, 106), seed=2, source=src)
    b = compose.render_effect("grade-" + name, size=(160, 106), seed=2, source=src)
    assert a.shape == (106, 160, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert 0 <= a.min() and a.max() <= 1 and a.std() > 0.02


@pytest.mark.parametrize("name", NAMES)
def test_identity_at_intensity_zero(name):
    src = _photo()
    out = compose.render_effect("grade-" + name, size=(160, 106), source=src, params={"intensity": 0})
    ref = F.fit_cover(src, (160, 106), "center")
    assert np.abs(out - ref).max() < 1e-6


def test_identity_lut_roundtrip():
    img = np.random.default_rng(0).random((40, 50, 3)).astype(np.float32)
    assert np.abs(G.apply_lut(img, G.identity_lut()) - img).max() < 1e-5


def test_lut_matches_exact_grade():
    rng = np.random.default_rng(1)
    s = rng.random((5000, 1, 3)).astype(np.float32)
    for n in NAMES:
        g = G.LOOKS[n].grade(G.look_params(n, None))
        err = np.sqrt(((g.apply(s[:, 0]) - G.apply_lut(s, g.lut(33))[:, 0]) ** 2).mean()) * 255
        assert err < 6.5, (n, err)
        if n != "infrared-false-colour":
            assert err < 2.5, (n, err)


def test_cube_write_read(tmp_path):
    lut = G.look_lut("teal-orange")
    p = str(tmp_path / "a.cube")
    G.write_cube(lut, p, "t")
    back = G.read_cube(p)
    assert back.shape == (33, 33, 33, 3) and np.abs(back - lut).max() < 1e-5
    txt = open(p).read()
    assert "LUT_3D_SIZE 33" in txt and txt.count("\n") > 33 ** 3
    # red varies fastest in a .cube: first rows change r only
    first = [list(map(float, l.split())) for l in txt.splitlines()[5:7]]
    assert first[0] != first[1]


def test_look_intensity_is_monotone():
    src = _photo()
    ref = F.fit_cover(src, (120, 80), "center")
    d = [np.abs(compose.render_effect("grade-teal-orange", size=(120, 80), source=src, params={"intensity": k, "finish": 0}) - ref).mean() for k in (0.25, 0.5, 1.0)]
    assert d[0] < d[1] < d[2]


def test_skin_protection():
    skin = np.tile(G.ink(0.7, 0.07, 52), (30, 30, 1)).astype(np.float32)
    h = lambda rgb: G.lab_to_lch(G.rgb_to_oklab(rgb.mean((0, 1))))[2]  # noqa: E731
    a = compose.render_effect("grade-teal-orange", size=(30, 30), source=skin, params={"skin": 1.0, "finish": 0})
    b = compose.render_effect("grade-teal-orange", size=(30, 30), source=skin, params={"skin": 0.0, "finish": 0})
    assert abs(G.hue_diff(h(a), 52)) < abs(G.hue_diff(h(b), 52)) + 1e-6 and abs(G.hue_diff(h(a), 52)) < 12


def test_describe_and_nearest():
    d = G.describe(_photo())
    for k in G.VIBE_FIELDS:
        assert k in d
    chart = G.test_chart()
    for n in ("cyanotype", "cold-blue-flash", "sepia-albumen"):
        out = G.render_look(chart, n, G.CHART_SIZE, 0, {"crop": "center"})
        assert G.nearest_looks(out, 1)[0][0] == n
    assert len(G.nearest_looks(_photo(), 5)) == 5


def test_web_exports():
    svg, fid = G.svg_filter("sepia-albumen")
    assert "feColorMatrix" in svg and "feComponentTransfer" in svg and fid["rms_8bit"] < 10
    css, fc = G.css_filter("high-key-editorial")
    assert css.startswith("filter:")


def test_speed_full_size():
    src = F.synthetic_source((1500, 1000), 1)
    for n in ("cold-blue-flash", "vhs", "cinestill-800t"):
        t = time.time(); G.render_look(src, n, (1500, 1000)); assert time.time() - t < 3.0, n
