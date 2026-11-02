import json

import numpy as np
import pytest
from PIL import Image

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain
from wonhee.db import repo
from wonhee.edit import mask as mk
from wonhee.edit import place as pl
from wonhee.edit import text as T
from wonhee.edit.text import TextSpec

W, H = 540, 960


@pytest.fixture(autouse=True)
def node():
    import wonhee.nodes.place  # noqa: F401 - registers text_place
    yield


def photo(quiet: list[tuple[float, float, float, float]]) -> np.ndarray:
    """A noisy BGR frame with flat dark patches at (x0, y0, x1, y1) fractions: the only places text can read."""
    rng = np.random.default_rng(7)
    im = np.clip(rng.normal(128, 55, (H, W, 3)), 0, 255).astype(np.uint8)
    for x0, y0, x1, y1 in quiet:
        im[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = (40, 40, 40)
    return im


def blob(x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 1.0
    return m


def inside(c: pl.Candidate, region) -> bool:
    x0, y0, x1, y1 = region
    return c.x - c.w / 2 >= x0 - 0.01 and c.x + c.w / 2 <= x1 + 0.01 and c.y - c.h / 2 >= y0 - 0.01 \
        and c.y + c.h / 2 <= y1 + 0.01


SPEC = TextSpec("still shipping")


def test_text_lands_in_the_quiet_region():
    quiet = (0.05, 0.30, 0.95, 0.46)
    best = pl.place(photo([quiet]), SPEC)[0]
    assert inside(best, quiet) and best.busyness < 0.1 and best.score > 0.8


def test_never_over_the_subject_blob():
    quiet = (0.04, 0.50, 0.96, 0.72)
    subject = blob(0.30, 0.45, 0.70, 0.76)
    cands = pl.place(photo([quiet]), SPEC, subject, n=3)
    assert cands
    for c in cands:
        assert c.subject_overlap == 0.0
        apart_x = c.x + c.w / 2 <= 0.30 or c.x - c.w / 2 >= 0.70
        apart_y = c.y + c.h / 2 <= 0.45 or c.y - c.h / 2 >= 0.76
        assert apart_x or apart_y


def test_subject_loses_to_a_busier_clear_area():
    quiet = (0.04, 0.50, 0.96, 0.72)
    subject = blob(0.0, 0.40, 1.0, 0.76)  # the quiet region is all subject
    best = pl.place(photo([quiet]), SPEC, subject)[0]
    assert best.subject_overlap == 0.0 and not inside(best, quiet)


def test_stays_inside_the_safe_zones():
    for im in (photo([(0.0, 0.0, 1.0, 1.0)]), photo([(0.05, 0.85, 0.95, 0.99)]), photo([(0.05, 0.02, 0.95, 0.10)])):
        for c in pl.place(im, TextSpec("me and bro shipping our 4th AI wrapper after saying we would never")):
            assert c.y - c.h / 2 >= T.SAFE_TOP - 1e-6 and c.y + c.h / 2 <= T.SAFE_BOTTOM + 1e-6
            assert c.x - c.w / 2 >= pl.MARGIN - 1e-6 and c.x + c.w / 2 <= 1 - pl.MARGIN + 1e-6


def test_three_distinct_ranked_candidates():
    cands = pl.place(photo([(0.05, 0.14, 0.95, 0.24), (0.05, 0.55, 0.95, 0.70)]), SPEC)
    assert len(cands) == 3
    assert [c.score for c in cands] == sorted((c.score for c in cands), reverse=True)
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            assert abs(a.x - b.x) > 0.02 or abs(a.y - b.y) > 0.02
    assert all(c.why for c in cands)


def test_deterministic():
    im, subject = photo([(0.05, 0.30, 0.95, 0.46)]), blob(0.4, 0.3, 0.6, 0.46)
    assert pl.place(im, SPEC, subject) == pl.place(im, SPEC, subject)


def test_block_size_comes_from_the_text_engine():
    bw, bh, nl = pl.measure(SPEC, (W, H))
    lines, _px, _gap, block_h, _y0 = T._place(SPEC, (W, H))
    assert nl == len(lines) and bh == block_h and bw == max(l.width for l in lines)
    c = pl.place(photo([]), SPEC)[0]
    assert c.w == pytest.approx(bw / W) and c.h == pytest.approx(bh / H)


def test_band_spans_the_frame_and_only_moves_in_y():
    cands = pl.place(photo([(0.0, 0.30, 1.0, 0.46)]), TextSpec("band line", "band"))
    assert cands and all(c.w == 1.0 and c.x == 0.5 for c in cands)


def test_tall_frame_block_over_the_button_strip_is_measured_narrow():
    c = pl.place(photo([(0.0, 0.36, 1.0, 0.74)]), TextSpec("one two three four five six seven eight nine ten eleven"))[0]
    assert c.width_frac == T.BUTTONS_WRAP and c.x + c.w / 2 <= pl.BUTTON_EDGE + 1e-6


def test_empty_text_has_no_candidates():
    assert pl.place(photo([]), TextSpec("  ")) == []


def test_accepts_pil_images():
    assert pl.place(Image.fromarray(photo([(0.05, 0.30, 0.95, 0.46)])[..., ::-1]), SPEC)


def test_node_falls_back_with_a_flag_when_no_mask(conn, settings, tiny_jpg, monkeypatch):
    from wonhee.media import path_for
    def boom(*a, **k):
        raise OSError("model not available offline")
    monkeypatch.setattr(mk, "subject", boom)
    sha = repo.save_file(conn, settings.media_dir, tiny_jpg, "image", "image/jpeg")
    rid, status = run_chain(conn, settings, [{"id": "p", "node": "text_place", "in": {"image": sha},
                                              "params": {"text": "still shipping"}}], fake_deps())
    assert status == "done"  # still produces a placement; the flag tells the reviewer face avoidance was not checked
    assert "NO_SUBJECT_MASK" in flags(conn, rid, "p")
    rep = json.loads(path_for(settings.media_dir, out(conn, rid, "p", "placement")).read_text())
    assert rep["subject_mask"] is False and rep["best"]["why"] and len(rep["candidates"]) >= 1


def test_node_uses_the_mask_when_available(conn, settings, tiny_jpg, monkeypatch):
    monkeypatch.setattr(mk, "subject", lambda im, d, kind="person": np.zeros(im.shape[:2], np.float32))
    from wonhee.media import path_for
    sha = repo.save_file(conn, settings.media_dir, tiny_jpg, "image", "image/jpeg")
    rid, status = run_chain(conn, settings, [{"id": "p", "node": "text_place", "in": {"image": sha},
                                              "params": {"text": "hi", "n": 2}}], fake_deps())
    assert status == "done" and "NO_SUBJECT_MASK" not in flags(conn, rid, "p")
    rep = json.loads(path_for(settings.media_dir, out(conn, rid, "p", "placement")).read_text())
    assert rep["subject_mask"] is True and len(rep["candidates"]) <= 2


def test_node_rejects_unknown_params():
    from wonhee.nodes.place import PlaceParams
    with pytest.raises(Exception):
        PlaceParams(text="x", colour="red")

