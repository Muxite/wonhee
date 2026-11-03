import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose

NAMES = [n for n in art.names() if n.startswith(("micro-", "arch-"))]


def test_effect_count():
    assert len([n for n in NAMES if n.startswith("micro-")]) >= 6
    assert len([n for n in NAMES if n.startswith("arch-")]) >= 7


@pytest.mark.parametrize("name", NAMES)
def test_deterministic_shape_nonconstant(name):
    a = compose.render_effect(name, size=(300, 200), seed=1)
    b = compose.render_effect(name, size=(300, 200), seed=1)
    c = compose.render_effect(name, size=(300, 200), seed=2)
    assert a.shape == (200, 300, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.std() > 0.03 and 0 <= a.min() and a.max() <= 1


@pytest.mark.parametrize("name", NAMES)
def test_tags_and_light_variant(name):
    e = art.registry.get(name)
    assert any(t in e.tags for t in ("engineered", "tactile", "precise", "urban", "monumental", "nocturnal"))
    light = compose.render_effect(name, size=(300, 200), palette="ink-on-paper")
    dark = compose.render_effect(name, size=(300, 200), palette="teal-coral")
    assert light.mean() > dark.mean()


def test_source_data_hooks():
    src = np.random.default_rng(0).random((60, 90, 3)).astype(np.float32)
    for n in ("micro-plates-source", "arch-night-photo", "arch-wall", "micro-plates"):
        a = compose.render_effect(n, size=(300, 200), source=src)
        b = compose.render_effect(n, size=(300, 200))
        assert a.shape == (200, 300, 3)
        assert not np.array_equal(a, b) or n == "arch-wall"
    field = np.random.default_rng(1).random((30, 40)).astype(np.float32)
    assert compose.render_effect("micro-plates-data", size=(300, 200), data=field).std() > 0.03
    labels = (np.arange(40)[None, :] // 10 + np.arange(30)[:, None] // 10 * 4).astype(np.int32)
    assert compose.render_effect("micro-plates-data", size=(300, 200), data=labels).std() > 0.03
    pts = np.random.default_rng(2).random((7, 2)).astype(np.float32)
    assert compose.render_effect("micro-plates-data", size=(300, 200), data=pts, params={"boundary": "data"}).std() > 0.03


def test_independent_controls():
    base = compose.render_effect("micro-plates", size=(300, 200), seed=3)
    grain = compose.render_effect("micro-plates", size=(300, 200), seed=3, params={"grain": 0.0, "wear": 0.0, "glow": 0.0})
    assert not np.array_equal(base, grain)
    # colour controls leave the plate layout (luminance structure) recognisable but change the hues
    hue = compose.render_effect("micro-plates", size=(300, 200), seed=3, params={"hue_spread": 70.0})
    assert not np.array_equal(base, hue)
    lit = compose.render_effect("arch-wall", size=(300, 200), seed=3, params={"lit": 0.05})
    lit2 = compose.render_effect("arch-wall", size=(300, 200), seed=3, params={"lit": 0.9})
    assert lit2.mean() > lit.mean()
