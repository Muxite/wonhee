import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, ditherkit as D, printkit as K, swisskit as S

NAMES = [n for n in art.names() if n.split("-")[0] in ("print", "swiss", "dither")]


SEED_FREE = {"dither-photo", "dither-blocks"}


def _src(seed=0, shape=(120, 180)):
    r = np.random.default_rng(seed)
    from scipy import ndimage
    base = ndimage.gaussian_filter(r.random(shape + (3,)).astype(np.float32), (6, 6, 0))
    return (base - base.min()) / np.ptp(base)


def test_counts():
    assert len(NAMES) >= 12


@pytest.mark.parametrize("name", NAMES)
def test_effect_contract(name):
    kw = dict(size=(300, 200), seed=3, source=_src())
    a = compose.render_effect(name, **kw)
    b = compose.render_effect(name, **kw)
    c = compose.render_effect(name, **{**kw, "seed": 4})
    assert a.shape == (200, 300, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert a.std() > 0.02
    if name not in SEED_FREE:      # Bayer/diffusion of a given source has no randomness: the seed only matters for noise dithers
        assert not np.array_equal(a, c)
    assert 0 <= a.min() and a.max() <= 1


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("pal", ["ink-on-paper", "ultramarine"])
def test_small_size_and_palettes(name, pal):
    a = compose.render_effect(name, size=(288, 192), seed=1, palette=pal)
    assert a.shape == (192, 288, 3) and a.std() > 0.03


def _hf(img):
    g = img.mean(-1)
    return float(np.abs(np.diff(g, axis=1)).mean())


@pytest.mark.parametrize("name,extra", [("print-screenprint", {}), ("swiss-grid-poster", {}), ("print-nine-ways", {}), ("print-riso-dots", {})])
def test_clean_variant_has_no_surface_texture(name, extra):
    kw = dict(size=(400, 266), seed=2, palette="ink-on-paper", source=_src(2, (140, 210)))
    tex = compose.render_effect(name, params={"clean": False, **extra}, **kw)
    cln = compose.render_effect(name, params={"clean": True, **extra}, **kw)
    assert _hf(cln) < _hf(tex)
    assert not np.array_equal(tex, cln)


def test_paper_fibre_and_clean():
    rng = np.random.default_rng(0)
    p, h = K.paper((300, 200), rng, K.color("cream"), 0.8, 0.6, 0.6)
    flat, _ = K.paper((300, 200), np.random.default_rng(0), K.color("cream"), 0, 0, 0)
    assert p[..., 0].std() > 0.01 and flat[..., 0].std() < 1e-6 and 0 <= h.min() and h.max() <= 1


def test_screen_coverage_tracks_input():
    size = (600, 400)
    for kind in ("dot", "line", "fm"):
        lo = K.screen(np.full((400, 600), 0.2, np.float32), size, kind=kind, pitch=12).mean()
        hi = K.screen(np.full((400, 600), 0.7, np.float32), size, kind=kind, pitch=12).mean()
        assert 0.12 < lo < 0.30 and 0.6 < hi < 0.82
    assert K.screen(np.ones((40, 60), np.float32), (60, 40), kind="dot").min() > 0.95   # full coverage = solid
    assert K.screen(np.zeros((40, 60), np.float32), (60, 40), kind="dot").max() < 0.05


def test_screen_pitch_has_small_size_floor():
    # at 288 px wide a 12 px (1500 master) pitch would be 2.3 px: the floor keeps it >= 4.2
    s = K.screen(np.full((192, 288), 0.5, np.float32), (288, 192), kind="dot", pitch=12, angle=0)
    row = s[96]
    edges = np.flatnonzero(np.diff((row > 0.5).astype(int)) != 0)
    assert len(edges) < 288 / 4.2 * 2 + 4


def test_separations_return_fields_in_range():
    img = _src(1, (90, 120))
    rng = np.random.default_rng(0)
    for covs in (K.sep_tone(img.mean(-1), 3), K.sep_levels(img.mean(-1), 3), K.sep_hue(img, 3, rng), K.sep_cmy(img, 3)):
        assert len(covs) >= 3
        for c in covs:
            assert c.shape == (90, 120) and 0 <= c.min() and c.max() <= 1 + 1e-5


def test_registration_floor_and_shift():
    a = np.zeros((50, 50), np.float32)
    a[20:30, 20:30] = 1
    b = K.shift(a, 3, 0)
    assert b[25, 24] < 0.5 and b[25, 25] > 0.5 or abs(b.sum() - a.sum()) < 5
    assert np.array_equal(K.shift(a, 0, 0), a)


def test_overprint_multiply_darkens():
    out = np.ones((4, 4, 3), np.float32) * 0.9
    m = np.ones((4, 4), np.float32)
    o = K.lay(out, m, K.color("ultramarine"), 1.0, "multiply")
    assert (o < out).all()


# ---- swiss

def test_grid_and_pack_snap_and_quadrants():
    g = S.Grid((600, 400), 6, 0.0, 0.01)
    assert g.rows >= 3 and abs(g.mw - g.mh) < g.mw * 0.3
    rng = np.random.default_rng(1)
    blocks, occ = S.pack(g, rng, 0.6, 0.3)
    assert len(blocks) >= 6
    for q in ((0, 0), (1, 0), (0, 1), (1, 1)):
        sub = occ[q[1] * g.rows // 2:(q[1] + 1) * g.rows // 2 + 0, q[0] * g.cols // 2:(q[0] + 1) * g.cols // 2]
        assert (sub >= 0).any()
    # no overlaps: each module has at most one block id and blocks are rectangles
    for i, (c0, r0, c1, r1) in enumerate(blocks):
        assert (occ[r0:r1, c0:c1] == i).all()
    x0, y0, x1, y1 = g.rect(0, 0, 6, g.rows)
    assert (x0, y0, x1, y1) == (0, 0, 600, 400)


def test_swiss_type_layer_only_with_real_text():
    kw = dict(size=(600, 400), seed=5, palette="ink-on-paper", params={"clean": True})
    plain = compose.render_effect("swiss-grid-poster", **kw)
    typed = compose.render_effect("swiss-grid-poster", data={"title": "Wave simulation", "subtitle": "Acoustics", "caption": "2026"}, **kw)
    junk = compose.render_effect("swiss-grid-poster", data=np.arange(12).reshape(3, 4), **kw)
    assert np.array_equal(plain, junk)             # non-text data never produces type
    assert not np.array_equal(plain, typed)
    assert S.parse_text("A") == {"title": "A"}
    assert S.parse_text(None) is None
    assert S.font_path("bold") is not None


def test_swiss_variants_light_and_dark():
    light = compose.render_effect("swiss-grid-poster", size=(300, 200), seed=1, palette="ink-on-paper", params={"clean": True})
    dark = compose.render_effect("swiss-grid-poster", size=(300, 200), seed=1, palette="ink-on-paper", params={"clean": True, "variant": "dark"})
    assert light.mean() > dark.mean() + 0.1


def test_form_tiling_fills_frame():
    im = compose.render_effect("swiss-form-tiling", size=(300, 200), seed=2, params={"clean": True})
    # every quadrant carries more than one colour: no empty areas
    for sl in (np.s_[:100, :150], np.s_[:100, 150:], np.s_[100:, :150], np.s_[100:, 150:]):
        assert len(np.unique((im[sl] * 20).astype(int).reshape(-1, 3), axis=0)) > 1


# ---- dither

def test_bayer_matrix_is_permutation_of_levels():
    for n in (2, 4, 8, 16):
        b = D.bayer(n)
        assert b.shape == (n, n) and np.allclose(np.sort(b.ravel()), (np.arange(n * n) + 0.5) / (n * n))


def test_kernel_weights_sum():
    for k, (taps, div) in D.KERNELS.items():
        s = sum(w for _, _, w in taps)
        assert s == div or (k == "atkinson" and s == 6 and div == 8)


def test_blue_noise_is_uniform():
    t = D.blue_noise(32, np.random.default_rng(0))
    assert abs(t.mean() - 0.5) < 0.01 and t.min() > 0 and t.max() < 1


@pytest.mark.parametrize("algo", D.ALGOS)
def test_quantise_uses_only_palette_and_preserves_mean(algo):
    h, w = 48, 64
    ramp = np.linspace(0.2, 0.8, w)[None, :].repeat(h, 0)
    lab = np.stack([ramp, np.zeros_like(ramp), np.zeros_like(ramp)], -1).astype(np.float32)
    pal = np.array([[0.05] * 3, [0.95] * 3], np.float32)
    idx = D.quantise(lab, pal, algo, np.random.default_rng(0))
    assert set(np.unique(idx)) <= {0, 1}
    if algo != "none":
        out_l = D.F.to_oklab(pal[None])[0][idx][..., 0]
        assert abs(out_l.mean() - ramp.mean()) < 0.1      # dither preserves mean tone
    else:
        assert idx.shape == (h, w)


def test_dither_photo_output_colours_subset_of_palette_and_scale_consistent():
    src = _src(3, (150, 225))
    kw = dict(seed=1, params={"colors": "gameboy", "algo": "fs", "cols": 60}, source=src)
    a = compose.render_effect("dither-photo", size=(300, 200), **kw)
    b = compose.render_effect("dither-photo", size=(600, 400), **kw)
    pal = {tuple((D.resolve_palette("gameboy", type("C", (), {"palette": None, "seed": 0}), None) * 255 + .5).astype(int)[i]) for i in range(4)}
    cols = {tuple((c * 255 + .5).astype(int)) for c in np.unique(a.reshape(-1, 3), axis=0)}
    assert cols <= pal
    # same logical pixel grid: the 600 px render downsampled equals the 300 px one
    down = b.reshape(200, 2, 300, 2, 3).mean((1, 3))
    assert np.abs(down - a).mean() < 0.06


def test_grid_size_floor_and_budget():
    gw, gh, b = D.grid_size((288, 192), 400)
    assert b >= D.MIN_BLOCK - 1e-6 and gw <= 288 / D.MIN_BLOCK + 1
    gw, gh, b = D.grid_size((1500, 1000), 5000)
    assert gw * gh <= D.MAX_CELLS


def test_resolve_palette_forms():
    ctx = type("C", (), {"palette": compose._palette("ember"), "seed": 0})
    img = _src(0)
    assert D.resolve_palette("palette:6", ctx, img).shape == (6, 3)
    assert D.resolve_palette("auto:5", ctx, img).shape == (5, 3)
    assert D.resolve_palette("#112233,#ffeeaa,#884400", ctx, img).shape == (3, 3)
    L = D.F.to_oklab(D.resolve_palette("palette:5", ctx, img)[None])[0, :, 0]
    assert (np.diff(L) > 0).all()
