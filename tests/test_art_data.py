
import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, datakit as dk, mckit as mk
from wonhee.art.effects.multichan_composite import channel_meta

DATA = [n for n in art.names() if n.startswith("data-")]
MULTI = [n for n in art.names() if n.startswith("multichan-")]


def test_counts():
    assert len(DATA) >= 6 and len(MULTI) >= 5


@pytest.mark.parametrize("name", DATA + MULTI)
@pytest.mark.parametrize("pal", ["ember", "red-on-cream"])
def test_deterministic_shape_nonconstant(name, pal):
    a = compose.render_effect(name, size=(300, 200), seed=1, palette=pal)
    b = compose.render_effect(name, size=(300, 200), seed=1, palette=pal)
    c = compose.render_effect(name, size=(300, 200), seed=2, palette=pal)
    assert a.shape == (200, 300, 3) and a.dtype == np.float32
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.std() > 0.04 and 0 <= a.min() and a.max() <= 1


@pytest.mark.parametrize("name", DATA)
def test_full_mode_and_card_small(name):
    full = compose.render_effect(name, size=(600, 400), params={"mode": "full"})
    card = compose.render_effect(name, size=(288, 192))
    assert full.shape == (400, 600, 3) and card.shape == (192, 288, 3)
    assert card.std() > 0.05


def test_scales_monotone_and_in_gamut():
    for pol in ("dark", "light"):
        for n in dk.SEQ:
            r = dk.check_scale(dk.lut(n, None, polarity=pol))
            assert r["monotone"] and r["in_gamut"] and r["range"] > 0.55, (pol, n, r)
        for n in dk.DIV:
            r = dk.check_scale(dk.lut(n, None, polarity=pol), "diverging")
            assert r["symmetric"] and r["in_gamut"], (pol, n, r)


def test_categorical_distinct():
    c = dk.categorical(6, "dark")
    d = np.linalg.norm(c[:, None] - c[None], axis=-1) + np.eye(6)
    assert d.min() > 0.18


def test_normalize_explicit_range_reported():
    m = np.array([[0, 5], [10, 20]], np.float32)
    v, used = dk.normalize(m, 0, 10)
    assert used == (0.0, 10.0) and v[1, 1] == 1.0 and v[0, 1] == 0.5
    v, used = dk.normalize(m, None, None, "robust", center=5)
    assert abs((used[0] + used[1]) / 2 - 5) < 1e-6


def test_cluster_order_groups_blocks():
    rng = np.random.default_rng(0)
    lab = np.repeat([0, 1, 2], 6)
    base = np.eye(3)[lab] @ np.eye(3)[lab].T
    perm = rng.permutation(18)
    M = base[np.ix_(perm, perm)] + rng.normal(0, 0.05, (18, 18))
    o, Z = dk.cluster_order(M, 0)
    ordered = lab[perm][o]
    assert (np.diff(ordered) != 0).sum() <= 3          # three contiguous groups (up to the order of groups)


def test_aggregate_nan_safe():
    m = np.arange(36, dtype=np.float32).reshape(6, 6); m[0, 0] = np.nan
    a = dk.aggregate(m, 3, 3)
    assert a.shape == (3, 3) and np.isfinite(a).all()


def test_provenance_flags():
    assert dk.provenance("x", None)["synthetic"] is True
    assert dk.provenance("x", np.ones((2, 2)))["synthetic"] is False
    assert channel_meta("multichan-composite")["standin"] is True
    assert channel_meta("multichan-composite", source=np.zeros((2, 2, 3)))["aesthetic"] is True
    assert channel_meta("multichan-composite", data={"a": 1})["synthetic"] is False


def test_real_matrix_changes_output():
    rng = np.random.default_rng(3)
    m = rng.random((12, 7))
    a = compose.render_effect("data-heat-bleed", size=(300, 200), data=m, params={"scale": "magma"})
    b = compose.render_effect("data-heat-bleed", size=(300, 200), data=1 - m, params={"scale": "magma"})
    assert not np.array_equal(a, b)
    # dict form with labels and runs
    s3 = rng.random((10, 5, 3))
    out = compose.render_effect("data-stripe-matrix", size=(300, 200), data={"scores": s3})
    assert out.std() > 0.04
    corr = np.corrcoef(rng.normal(size=(10, 50)))
    compose.render_effect("data-cluster-matrix", size=(300, 200), data=corr)
    ser = {"a": rng.random(60), "b": rng.random(60), "c": rng.random(60)}
    compose.render_effect("data-terminal-panels", size=(300, 200), data=ser)
    compose.render_effect("data-pixel-field", size=(300, 200), data=m)
    compose.render_effect("data-annotated-grow", size=(300, 200), data=m)


def test_multichannel_real_channels_and_ranges():
    yy, xx = np.mgrid[0:60, 0:90]
    chans = {"a": np.sin(xx / 9.0), "b": np.cos(yy / 7.0), "c": xx * 1.0}
    img = compose.render_effect("multichan-composite", size=(300, 200), data=chans, params={"ranges": {"a": [-1, 1], "b": [-1, 1], "c": [0, 89]}})
    img2 = compose.render_effect("multichan-composite", size=(300, 200), data=chans, params={"ranges": {"a": [-1, 1], "b": [-1, 1], "c": [0, 400]}})
    assert not np.array_equal(img, img2)                       # explicit range matters
    it = mk.intake(type("C", (), dict(data=chans, source=None, size=(300, 200), seed=0))(), 3, None, {"c": [0, 400]})
    assert it.kind == "data" and not it.meta["standin"] and it.meta["ranges"]["c"] == [0.0, 400.0]
    for n in ("multichan-relief", "multichan-diff", "multichan-fluorescence"):
        assert compose.render_effect(n, size=(300, 200), data=chans).std() > 0.03


def test_channel_mix_source_and_aesthetic_label():
    src = np.random.default_rng(0).random((60, 90, 3)).astype(np.float32)
    img = compose.render_effect("multichan-channel-mix", size=(300, 200), source=src)
    assert img.shape == (200, 300, 3) and img.std() > 0.05
    it = mk.intake(type("C", (), dict(data=None, source=src, size=(300, 200), seed=0))(), 3, ["red", "green", "blue"])
    assert it.kind == "aesthetic" and it.meta["aesthetic"] and "not infrared" in it.meta["note"]


def test_budget_1500():
    import time
    for n in ("data-heat-bleed", "data-terminal-panels", "multichan-composite"):
        t = time.time(); compose.render_effect(n, size=(1500, 1000)); assert time.time() - t < 8


def test_data_effects_survive_tiny_sizes():
    from wonhee.art.compose import render_effect
    for name in ("data-cluster-matrix", "data-annotated-grow", "data-terminal-panels"):
        img = render_effect(name, size=(96, 64), seed=0, params={"mode": "full"})
        assert img.shape == (64, 96, 3)
