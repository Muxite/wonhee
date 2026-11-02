import cv2
import numpy as np
import pytest

from tests.test_nodes_subs import out, run_chain
from wonhee.db import repo
from wonhee.edit import region as rg

H, W = 200, 100


def flat(v=150):
    return np.full((H, W, 3), v, np.uint8)


def test_ramp_is_full_at_the_start_none_at_the_end_and_smooth():
    m = rg.mask(rg.Region("ramp", (0, 0, 1, 0.5)), H, W)
    assert m[0, 0] == pytest.approx(1) and m[H // 2 + 2, 0] == 0 and m[-1, 0] == 0
    assert np.all(np.diff(m[:H // 2, 0]) <= 1e-6)  # never rises going away from the start


def test_ramp_can_run_upward():
    m = rg.mask(rg.Region("ramp", (0, 1, 1, 0.5)), H, W)
    assert m[-1, 0] == pytest.approx(1) and m[0, 0] == 0


def test_rect_inside_outside_and_invert():
    r = rg.Region("rect", (0.25, 0.25, 0.75, 0.75), feather=0.02)
    m = rg.mask(r, H, W)
    assert m[H // 2, W // 2] == 1 and m[2, 2] == 0
    assert rg.mask(rg.Region("rect", r.box, 0.02, invert=True), H, W)[H // 2, W // 2] == 0


def test_darken_touches_only_the_region():
    im = flat()
    out_im = rg.apply_layers(im, [rg.Layer("darken", rg.Region("ramp", (0, 0, 1, 0.4)), 0.8)])
    assert out_im[0].mean() < 80 and np.array_equal(out_im[-1], im[-1])


def test_zero_amount_is_identity_and_later_layers_see_earlier_ones():
    im = flat()
    assert np.array_equal(rg.apply_layers(im, [rg.Layer("blur", amount=0)]), im)
    two = rg.apply_layers(im, [rg.Layer("darken", amount=0.5), rg.Layer("darken", amount=0.5)])
    assert two.mean() < rg.apply_layers(im, [rg.Layer("darken", amount=0.5)]).mean()


def test_every_effect_runs_and_keeps_shape():
    im = cv2.randu(np.zeros((H, W, 3), np.uint8), 0, 255)
    for name in rg.EFFECTS:
        opts = {"preset": "matte"} if name == "look" else {}
        r = rg.apply_layers(im, [rg.Layer(name, rg.Region("ellipse", (0.1, 0.1, 0.9, 0.9)), 0.5, opts)])
        assert r.shape == im.shape and r.dtype == np.uint8, name


@pytest.mark.parametrize("bad", [dict(effect="sparkle"), dict(amount=3), dict(effect="look")])
def test_bad_layers_are_refused(bad):
    with pytest.raises(ValueError):
        rg.Layer(**{"effect": "blur", **bad})


@pytest.mark.parametrize("args", [("cube", (0, 0, 1, 1)), ("rect", (0.6, 0, 0.4, 1)), ("rect", (0, 0, 2, 1))])
def test_bad_regions_are_refused(args):
    with pytest.raises(ValueError):
        rg.Region(*args)


def test_node_runs_and_stores_an_image(conn, settings, tiny_jpg):
    sha = repo.save_file(conn, settings.media_dir, tiny_jpg, "image", "image/jpeg")
    rid, status = run_chain(conn, settings, [{"id": "r", "node": "region_fx", "in": {"image": sha},
                                              "params": {"layers": [{"effect": "darken", "amount": 0.9,
                                                                     "region": {"shape": "ramp", "box": [0, 0, 1, 0.5]}}]}}])
    assert status == "done"
    im = cv2.imread(str(settings.media_dir / out(conn, rid, "r", "image")[:2] / out(conn, rid, "r", "image")))
    assert im is not None and im[0].mean() < im[-1].mean()


def test_look_takes_a_hand_made_filter_and_validates_it():
    im = flat()
    dark = rg.apply_layers(im, [rg.Layer("look", amount=1, options={"filter": {"exposure": -0.5, "grain": 0}})])
    assert dark.mean() < im.mean()
    with pytest.raises(ValueError):
        rg.Layer("look", options={"filter": {"exposure": 5}})
