import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose

LIGHT = ["light-ribbons", "lightning-dbm", "lightning-bolts", "dla-crystal", "aurora-curtains", "star-trails",
         "laser-billiards", "caustic-pool", "flow-streaks", "firework-bursts", "bokeh-field",
         "plasma-arcs", "neon-circuit", "rain-shutter"]
SIZE = (300, 200)


def test_all_light_effects_registered():
    for n in LIGHT:
        assert n in art.names(family="light"), n


@pytest.mark.parametrize("name", LIGHT)
def test_light_effect_deterministic_shape_not_constant(name):
    a = compose.render_effect(name, size=SIZE, seed=3, palette="ember")
    b = compose.render_effect(name, size=SIZE, seed=3, palette="ember")
    c = compose.render_effect(name, size=SIZE, seed=4, palette="ember")
    assert a.shape == (200, 300, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.std() > 0.03 and np.isfinite(a).all()
    assert a.min() >= 0 and a.max() <= 1
    # never pure black ground
    assert a.mean(axis=2).min() > 0.0


def test_data_and_source_hooks():
    rng = np.random.default_rng(0)
    pts = rng.random((9, 2)).tolist()
    graph = {"nodes": [{"id": i} for i in range(6)], "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3]]}
    src = np.tile(np.linspace(0.1, 0.9, SIZE[0], dtype=np.float32)[None, :, None], (SIZE[1], 1, 3))
    for name in LIGHT:
        base = compose.render_effect(name, size=SIZE, seed=1)
        with_src = compose.render_effect(name, size=SIZE, seed=1, source=src)
        assert with_src.shape == base.shape
        for d in (pts, graph):
            out = compose.render_effect(name, size=SIZE, seed=1, data=d)
            assert out.shape == base.shape and np.isfinite(out).all()
    # data is not ignored where documented
    for name in ("plasma-arcs", "neon-circuit", "firework-bursts", "bokeh-field", "lightning-dbm"):
        a = compose.render_effect(name, size=SIZE, seed=1)
        b = compose.render_effect(name, size=SIZE, seed=1, data=graph)
        assert not np.array_equal(a, b), name


def test_no_second_hue_in_complement_palette():
    from wonhee.art.palette import rgb_to_oklch
    img = compose.render_effect("light-ribbons", size=SIZE, seed=1, palette="ultramarine")
    hs = []
    for px in img.reshape(-1, 3)[::37]:
        L, C, h = rgb_to_oklch(px)
        if C > 0.05:
            hs.append(h)
    hs = np.array(hs)
    assert len(hs) and (np.abs(((hs - 265 + 180) % 360) - 180) < 70).mean() > 0.9


def test_proj_nightcall_fleet_renders():
    from wonhee.art.registry import discover
    discover()
    a = compose.render_effect("proj-nightcall-fleet", size=(300, 200), seed=1, palette="hue:225:dark:analogous", params={"ships": 40, "steps": 80})
    assert a.shape == (200, 300, 3) and a.std() > 0.02 and np.isfinite(a).all()
