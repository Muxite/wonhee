import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, dotkit as dk

DOTS = [n for n in art.names() if n.startswith("dots-")]


def test_at_least_ten_dot_effects():
    assert len(DOTS) >= 10


@pytest.mark.parametrize("name", DOTS)
def test_effect_deterministic_shape_nonconstant(name):
    a = compose.render_effect(name, size=(300, 200), seed=1)
    b = compose.render_effect(name, size=(300, 200), seed=1)
    c = compose.render_effect(name, size=(300, 200), seed=2)
    assert a.shape == (200, 300, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert a.std() > 0.02
    if name != "dots-word":          # the text mosaic's letters are seed independent
        assert not np.array_equal(a, c)
    assert 0 <= a.min() and a.max() <= 1


def test_data_and_source_hooks():
    m = np.random.default_rng(0).random((9, 38))
    img = compose.render_effect("dots-bitmatrix", size=(300, 200), data=m)
    assert img.shape == (200, 300, 3)
    w = compose.render_effect("dots-word", size=(300, 200), data=["NEXUS", "42"])
    assert w.std() > 0.02
    src = np.random.default_rng(1).random((50, 70, 3)).astype(np.float32)
    for n in ("dots-stipple", "dots-pixel-waves", "dots-hex-field", "dots-dash-flow", "dots-halftone-lines"):
        assert compose.render_effect(n, size=(300, 200), source=src).shape == (200, 300, 3)


def test_lattices():
    size = (300, 200)
    rng = np.random.default_rng(0)
    sq = dk.square(size, 20)
    assert 150 <= len(sq) <= 176 and abs(np.diff(np.unique(sq.x)).min() - 20) < 1e-3
    hx = dk.hexagonal(size, 20)
    d = np.hypot(hx.x[:, None] - hx.x[None], hx.y[:, None] - hx.y[None])
    d[d == 0] = 1e9
    assert abs(d.min() - 20) < 0.5
    pl = dk.polar(size, (150, 100), 20, 20)
    assert len(pl) > 50
    po = dk.poisson(size, 12, rng)
    dd = np.hypot(po.x[:, None] - po.x[None], po.y[:, None] - po.y[None])
    dd[np.diag_indices(len(po))] = 1e9
    assert dd.min() >= 12 * 0.99
    f = np.zeros((200, 300), np.float32); f[:, 150:] = 1
    st = dk.stipple(f, size, 600, rng)
    assert (st.x > 150).sum() > 2 * (st.x <= 150).sum()
    qt = dk.quadtree(f, size, 64, 8, rng)
    assert qt.cell[qt.x > 150].min() < qt.cell[qt.x <= 150].max()
    g = dk.grid(3, 4, size)
    assert len(g) == 12 and g.grid_shape == (3, 4)


@pytest.mark.parametrize("shape", dk.SHAPES)
def test_render_dots_shapes_and_two_fields(shape):
    f = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (60, 1))
    ramp = np.linspace([0.1, 0.1, 0.1], [0.9, 0.8, 0.5], 64, dtype=np.float32)
    img = dk.render_dots(f, (100, 60), "square", pitch=10, bg=(0, 0, 0), ramp=ramp, shape=shape, size_by=f)
    assert img.shape == (60, 100, 3) and img.max() > 0.3
    # size grows left to right: more ink on the right half
    assert img[:, 50:].sum() > img[:, :50].sum()


def test_led_look_has_off_dots_and_glow():
    m = np.zeros((4, 6), np.float32); m[1, 1] = 1
    g = dk.grid(4, 6, (120, 80))
    img = dk.render_dots(m, (120, 80), g, bg=(0, 0, 0), color=(1, 0.5, 0.2), size_const=True, off=(0.1, 0.1, 0.1), threshold=0.5,
                         glow_sigma=6, normalize=False)
    assert img[30, 30, 0] > 0.9              # lit dot (row 1, col 1 -> centre 30,30)
    assert img[50, 110].sum() > 0.05         # an unlit site still shows its dim dot
    assert img[30, 44, 0] > 0.12             # glow bleeds past the lit dot's edge


def test_halftone_modes():
    f = np.tile(np.linspace(0, 1, 120, dtype=np.float32), (80, 1))
    for mode in ("round", "line", "hex", "bayer", "fs", "atkinson"):
        m = dk.halftone_mask(f, (120, 80), mode, cell=8 if mode in ("round", "line", "hex") else 2)
        assert m.shape == (80, 120)
        assert m[:, 90:].mean() > m[:, :30].mean() + 0.1, mode


def test_text_rasteriser():
    m = dk.text_matrix("HI")
    assert m.shape == (7, 11) and m.dtype == bool
    assert m[:, 0].all() and not m[:, 5].any()          # H's left stem, then the letter gap
    assert dk.text_matrix("a").shape == dk.text_matrix("A").shape
    two = dk.text_matrix("A\nB")
    assert two.shape[0] == 16
    band = dk.tile_text("AB", 9, 50, row0=1)
    assert band.shape == (9, 50) and band[1:8].any() and not band[0].any()
    t = dk.text_matrix("Wonhee", font="ttf", rows=9)
    assert t.shape[0] == 9 and t.any()


def test_proj_ledger_spans_deterministic():
    from wonhee.art import compose
    d = {"lines": [{"words": [[4, 0], [6, 1], [3, 1], [8, 0]] * 8}]}
    a = compose.render_effect("proj-ledger-spans", size=(300, 200), seed=1, palette="green-gold", data=d)
    b = compose.render_effect("proj-ledger-spans", size=(300, 200), seed=1, palette="green-gold", data=d)
    assert a.shape == (200, 300, 3) and np.allclose(a, b)
