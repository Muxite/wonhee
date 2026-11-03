import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, features as F, layerkit as K, regionkit as R

MINE = sorted(art.names(family="region") + art.names(family="layers") + art.names(family="collage"))


def _src(W=360, H=240, seed=3):
    return F.synthetic_source((W, H), seed)


def _two_surface(W=300, H=200):
    """left half smooth cyan-ish, right half rough brown: two obvious materials."""
    rng = np.random.default_rng(0)
    img = np.zeros((H, W, 3), np.float32)
    img[:, :W // 2] = (0.2, 0.55, 0.6)
    img[:, W // 2:] = (0.5, 0.35, 0.25) + rng.normal(0, 0.06, (H, W - W // 2, 1))
    return np.clip(img, 0, 1)


def test_families_have_effects_and_mood_tags():
    assert len(MINE) >= 9
    for n in MINE:
        tags = art.registry.REGISTRY[n].tags
        assert any(t.startswith("mood:") for t in tags), n


@pytest.mark.parametrize("name", MINE)
def test_deterministic_shape_nonconstant_no_pure_extremes(name):
    kw = dict(size=(240, 160), seed=2, palette="ember", source=_src())
    a = compose.render_effect(name, **kw)
    b = compose.render_effect(name, **kw)
    assert a.shape == (160, 240, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert a.std() > 0.02
    assert (a.max(-1) < 0.02).mean() < 0.01 and (a.min(-1) > 0.985).mean() < 0.01


@pytest.mark.parametrize("name", MINE)
def test_runs_without_source_and_light_palette(name):
    a = compose.render_effect(name, size=(180, 120), seed=1, palette="ember")
    b = compose.render_effect(name, size=(180, 120), seed=1, palette="ink-on-paper", source=_src())
    assert a.shape == b.shape == (120, 180, 3)
    assert b.mean() > 0.2          # light palette gives a lighter image than the dark default would


@pytest.mark.parametrize("name", MINE)
def test_palette_and_seed_change_output(name):
    s = _src()
    a = compose.render_effect(name, size=(200, 130), seed=1, palette="ember", source=s)
    b = compose.render_effect(name, size=(200, 130), seed=1, palette="ultramarine", source=s)
    assert np.abs(a - b).mean() > 0.01


def test_segment_finds_two_materials_and_orders_dark_to_light():
    img = _two_surface()
    rs = R.segment(img, n=4, by="colour", seed=0)
    assert 2 <= rs.n <= 3
    assert rs.labels.shape == img.shape[:2]
    # left and right halves land in different regions
    assert rs.labels[100, 40] != rs.labels[100, 260]
    L = F.luminance(img)
    means = [L[rs.labels == i].mean() for i in range(rs.n)]
    assert means == sorted(means)
    assert np.allclose(rs.weights.sum(0), 1, atol=1e-3)


def test_merge_collapses_similar_regions():
    rng = np.random.default_rng(1)
    g = np.linspace(0.45, 0.55, 300)[None, :, None] * np.ones((200, 1, 3), np.float32)
    img = np.clip(g + rng.normal(0, 0.003, g.shape), 0, 1).astype(np.float32)
    assert R.segment(img, n=5, merge=0.5).n <= 2


def test_polygon_and_mask_inputs_define_regions():
    img = _two_surface()
    poly = [{"region": 1, "points": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]}]
    rs = R.from_polygons(poly, (300, 200))
    assert rs.n == 2 and rs.labels[100, 150] == 1 and rs.labels[5, 5] == 0
    m = np.zeros((100, 150), np.uint8); m[:, 75:] = 255
    rs2 = R.from_mask(m, (300, 200))
    assert rs2.n == 2 and rs2.labels[100, 250] == 1 and rs2.labels[100, 50] == 0
    # grey-level mask with many levels is quantised
    g = np.tile(np.linspace(0, 1, 120, dtype=np.float32), (80, 1))
    assert R.from_mask(g, (240, 160), n=3).n == 3


def test_user_mask_via_ctx_data_overrides_segmentation():
    src = _two_surface()
    m = np.zeros((200, 300), np.uint8); m[:100] = 255        # top / bottom split, not left / right
    a = compose.render_effect("region-mask-overlay", size=(300, 200), seed=0, palette="ember", source=src, data={"mask": m},
                              params={"snap": 0.0, "smooth": 0.0})
    b = compose.render_effect("region-mask-overlay", size=(300, 200), seed=0, palette="ember", source=src,
                              params={"snap": 0.0, "smooth": 0.0})
    assert np.abs(a - b).mean() > 0.02
    # the seam of the mask is horizontal: rows far above / below differ strongly in the overlay colour
    assert np.abs(a[30, 150] - a[170, 150]).max() > 0.05


def test_mask_follows_the_source_crop():
    # source is 2x wider than the frame; a mask in source space must be cropped identically
    src = np.concatenate([_two_surface(300, 200)] * 2, 1)        # 600 x 200 (3:1), frame 3:2
    m = np.zeros((200, 600), np.uint8); m[:, 300:] = 255
    ctx = type("C", (), {})()
    fake = R.frame_source(type("X", (), {"size": (300, 200), "seed": 0, "palette": art.palette.named("ember"), "source": src,
                                          "data": None, "p": lambda self, k, d=None: {"crop": "center"}.get(k, d)})())
    rs = R.from_mask(m, (300, 200), fake.box)
    assert rs.n == 2
    assert 0.2 < rs.area()[1] < 0.8


def test_frequency_bands_sum_to_signal():
    L = F.luminance(_src(200, 130))
    lo, mid, hi = R.band_split(L, 12.0, 3.0)
    assert np.allclose(lo + mid + hi, L, atol=1e-4)


def test_restyle_texture_retention_scales_with_texture_gain():
    img = _two_surface()
    rs = R.segment(img, n=3, seed=0)
    L0 = F.luminance(img)

    def hf(tex):
        st = [R.RegionStyle(flat=1.0, mid=0.0, high=tex, L_mean=0.5) for _ in range(rs.n)]
        out = R.restyle(img, rs, st, s_low=20, s_tex=3, dark=True)
        L = F.luminance(out)
        return float(np.abs(L - R.lowpass(L, 3)).mean())

    assert hf(1.0) > 1.8 * hf(0.3) > 0


def test_region_poster_flattens_shading_but_keeps_texture():
    src = _two_surface()
    flat = compose.render_effect("region-material-poster", size=(300, 200), seed=0, palette="ember", source=src,
                                 params={"flat": 1.0, "texture": 1.0, "grain": 0})
    smooth = compose.render_effect("region-material-poster", size=(300, 200), seed=0, palette="ember", source=src,
                                   params={"flat": 1.0, "texture": 0.0, "grain": 0})
    hf = lambda a: np.abs(F.luminance(a) - R.lowpass(F.luminance(a), 2)).mean()
    assert hf(flat) > 1.5 * hf(smooth)


def test_depth_proxy_is_unit_range_and_documented_approximation():
    d = K.depth_proxy(_src(200, 130))
    assert d.shape == (130, 200) and 0 <= d.min() and d.max() <= 1
    assert "approximation" in K.__doc__


def test_exploded_planes_cumulative_masks_and_split_modes():
    src = _src(240, 160)
    for split in ("tone", "colour", "depth", "band", "channel"):
        a = compose.render_effect("layers-exploded-planes", size=(240, 160), seed=0, palette="ember", source=src, params={"split": split})
        assert a.std() > 0.02, split
    # a real height map in ctx.data replaces the heuristic depth
    h = np.tile(np.linspace(0, 1, 160, dtype=np.float32)[:, None], (1, 240))
    a = compose.render_effect("layers-exploded-planes", size=(240, 160), seed=0, palette="ember", source=src, data={"depth": h},
                              params={"split": "depth"})
    b = compose.render_effect("layers-exploded-planes", size=(240, 160), seed=0, palette="ember", source=src, params={"split": "depth"})
    assert np.abs(a - b).mean() > 0.005


def test_terrain_slabs_uses_data_field():
    f = np.add.outer(np.linspace(0, 1, 60), np.linspace(0, 1, 90)).astype(np.float32)
    a = compose.render_effect("layers-terrain-slabs", size=(240, 160), seed=0, palette="ember", data=f)
    b = compose.render_effect("layers-terrain-slabs", size=(240, 160), seed=0, palette="ember", data=f[::-1].copy())
    assert np.abs(a - b).mean() > 0.01


def test_projection_spacing_moves_planes():
    s = _src(240, 160)
    a = compose.render_effect("layers-exploded-planes", size=(240, 160), seed=0, palette="ember", source=s, params={"spacing": 0.05})
    b = compose.render_effect("layers-exploded-planes", size=(240, 160), seed=0, palette="ember", source=s, params={"spacing": 0.25})
    assert np.abs(a - b).mean() > 0.01


def test_collage_modes_differ_and_generated_surface_is_used():
    s = _src(300, 200)
    p = compose.render_effect("collage-paper-cutouts", size=(240, 160), seed=0, palette="ember", source=s, params={"mode": "paper"})
    c = compose.render_effect("collage-paper-cutouts", size=(240, 160), seed=0, palette="ember", source=s, params={"mode": "clean"})
    assert np.abs(p - c).mean() > 0.005
    a = compose.render_effect("collage-digital-panels", size=(240, 160), seed=0, palette="ember", source=s, params={"fills": "physics-moire"})
    b = compose.render_effect("collage-digital-panels", size=(240, 160), seed=0, palette="ember", source=s, params={"fills": "crop"})
    assert np.abs(a - b).mean() > 0.02
    g1 = compose.render_effect("collage-subject-sticker", size=(240, 160), seed=0, palette="ember", source=s, params={"surface": "physics-moire"})
    g2 = compose.render_effect("collage-subject-sticker", size=(240, 160), seed=0, palette="ember", source=s, params={"surface": "dots-hex-field"})
    assert np.abs(g1 - g2).mean() > 0.02


def test_sticker_mask_from_polygon_places_subject():
    s = _two_surface(300, 200)
    poly = [{"region": 1, "points": [[0.55, 0.2], [0.95, 0.2], [0.95, 0.85], [0.55, 0.85]]}]
    a = compose.render_effect("collage-subject-sticker", size=(300, 200), seed=0, palette="ember", source=s, data={"polygons": poly})
    assert a.shape == (200, 300, 3) and a.std() > 0.03
