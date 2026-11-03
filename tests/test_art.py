import numpy as np
from wonhee import art
from wonhee.art import metrics, compose


def test_registry_and_determinism():
    assert "wave-ripple" in art.names()
    a = compose.render_effect("wave-ripple", size=(300, 200), seed=1, params={"steps": 60, "scale": 4})
    b = compose.render_effect("wave-ripple", size=(300, 200), seed=1, params={"steps": 60, "scale": 4})
    c = compose.render_effect("wave-ripple", size=(300, 200), seed=2, params={"steps": 60, "scale": 4})
    assert a.shape == (200, 300, 3) and a.dtype == np.float32
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_palette_roles_contrast():
    p = art.make(40, polarity="dark")
    from wonhee.art.palette import rgb_to_oklch
    Lc, Ls, La = rgb_to_oklch(p.canvas)[0], rgb_to_oklch(p.structure)[0], rgb_to_oklch(p.accent)[0]
    assert Ls - Lc >= 0.25 and La - Lc >= 0.30


def test_metrics_flag_a_focal_blob_and_pass_even_detail():
    H, W = 400, 600
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-(((xx - W / 2) / 60) ** 2 + ((yy - H / 2) / 60) ** 2))[..., None].repeat(3, 2).astype(np.float32)
    assert not metrics.analyze(blob).ok
    rng = np.random.default_rng(0)
    even = np.clip(0.5 + 0.4 * np.sin(xx / 7.0) * np.sin(yy / 9.0) + rng.normal(0, 0.02, (H, W)), 0, 1)[..., None].repeat(3, 2).astype(np.float32)
    assert metrics.analyze(even).ok


def test_recipe_layers():
    img = compose.render_recipe({"size": [200, 140], "palette": "ember", "layers": [{"effect": "wave-ripple", "params": {"steps": 40, "scale": 4}, "opacity": 0.8}], "post": [{"op": "grain", "amount": 0.2}]})
    assert img.shape == (140, 200, 3)


def test_provenance_meta_flags_synthetic_data():
    from wonhee.art import compose
    img, meta = compose.render_effect("data-heat-bleed", size=(96, 64), seed=0, with_meta=True)
    assert meta["synthetic"] is True and meta["family"] in ("data", "multichan")
    _, meta2 = compose.render_effect("data-heat-bleed", size=(96, 64), seed=0, data=[[0.1, 0.9], [0.5, 0.3]], with_meta=True)
    assert meta2["synthetic"] is False
    _, meta3 = compose.render_effect("wave-ripple", size=(96, 64), seed=0, params={"steps": 20}, with_meta=True)
    assert "synthetic" not in meta3
