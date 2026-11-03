import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, metrics

art.discover()
PHYS = sorted(n for n in art.names() if n.startswith("physics-"))

SMALL = {  # cheaper params for the hook tests only; the main tests use defaults
    "physics-spiral": {"steps": 60}, "physics-grayscott": {"steps": 400}, "physics-swift-hohenberg": {"steps": 60},
    "physics-vortex": {"steps": 10}, "physics-wavemedium": {"steps": 120}, "physics-grains": {"steps": 8},
}


def test_physics_effects_registered():
    assert len(PHYS) >= 10


@pytest.mark.parametrize("name", PHYS)
def test_shape_dtype_deterministic_not_constant(name):
    a = compose.render_effect(name, size=(300, 200), seed=1)
    b = compose.render_effect(name, size=(300, 200), seed=1)
    c = compose.render_effect(name, size=(300, 200), seed=2)
    assert a.shape == (200, 300, 3) and a.dtype == np.float32
    assert 0.0 <= a.min() and a.max() <= 1.0
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.std() > 0.03          # not constant


@pytest.mark.parametrize("name", PHYS)
def test_palette_changes_output(name):
    a = compose.render_effect(name, size=(240, 160), seed=0, palette="ember", params=SMALL.get(name))
    b = compose.render_effect(name, size=(240, 160), seed=0, palette="ultramarine", params=SMALL.get(name))
    assert not np.allclose(a, b)


@pytest.mark.parametrize("name", PHYS)
def test_no_focal_point_at_card_size(name):
    img = compose.render_effect(name, size=(450, 300), seed=0)
    r = metrics.analyze(img)
    assert r.coverage >= 0.85 and r.center_bias <= 1.6 and r.focal_ratio <= 3.0, r.reasons


def _points(n=6):
    rng = np.random.default_rng(5)
    return [(float(x), float(y), float(w)) for x, y, w in zip(rng.random(n) * 0.8 + 0.1, rng.random(n) * 0.8 + 0.1, rng.normal(1, 0.3, n))]


@pytest.mark.parametrize("name", ["physics-chladni", "physics-interference", "physics-fieldlines", "physics-marbling", "physics-wavemedium", "physics-vortex"])
def test_point_data_hook(name):
    base = compose.render_effect(name, size=(240, 160), seed=0, params=SMALL.get(name))
    withd = compose.render_effect(name, size=(240, 160), seed=0, params=SMALL.get(name), data=_points())
    again = compose.render_effect(name, size=(240, 160), seed=0, params=SMALL.get(name), data=_points())
    assert not np.array_equal(base, withd) and np.array_equal(withd, again)


@pytest.mark.parametrize("name", ["physics-harmonograph", "physics-attractor"])
def test_series_data_hook(name):
    base = compose.render_effect(name, size=(240, 160), seed=0, params={"samples": 8000, "points": 40000})
    withd = compose.render_effect(name, size=(240, 160), seed=0, params={"samples": 8000, "points": 40000}, data=[0.3, 0.9, 0.1, 0.5, 0.7, 0.2])
    assert not np.array_equal(base, withd)


@pytest.mark.parametrize("name", ["physics-chladni", "physics-spiral", "physics-grayscott", "physics-swift-hohenberg", "physics-marbling", "physics-moire",
                                  "physics-caustics", "physics-grains", "physics-wavemedium", "physics-vortex"])
def test_source_hook(name):
    yy, xx = np.mgrid[0:120, 0:180]
    src = np.clip(0.5 + 0.4 * np.sin(xx / 17.0) * np.cos(yy / 11.0), 0, 1)[..., None].repeat(3, 2).astype(np.float32)
    base = compose.render_effect(name, size=(240, 160), seed=0, params=SMALL.get(name))
    with_s = compose.render_effect(name, size=(240, 160), seed=0, params=SMALL.get(name), source=src)
    assert with_s.shape == base.shape and not np.array_equal(base, with_s)


def test_proj_wavesim_field_deterministic():
    a = compose.render_effect("proj-wavesim-field", size=(240, 160), seed=3, params={"steps": 150})
    b = compose.render_effect("proj-wavesim-field", size=(240, 160), seed=3, params={"steps": 150})
    assert np.array_equal(a, b) and a.std() > 0.03
