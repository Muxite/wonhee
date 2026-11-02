import json

import numpy as np
import pytest
from PIL import Image

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain
from wonhee.db import repo
from wonhee.edit import fit as ft
from wonhee.edit import mask as mk
from wonhee.edit.text import TextSpec

W, H = 540, 960
SPEC = TextSpec("still shipping")


@pytest.fixture(autouse=True)
def node():
    import wonhee.nodes.fit  # noqa: F401 - registers photo_fit
    yield


def wall(v: int = 120) -> np.ndarray:
    return np.full((H, W, 3), v, np.uint8)


def checker(cell: int = 24) -> np.ndarray:
    yy, xx = np.indices((H, W))
    c = (((yy // cell) + (xx // cell)) % 2 * 255).astype(np.uint8)
    return np.dstack([c, c, c])


def blob(x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 1.0
    return m


def test_clean_wall_with_the_subject_low_is_ok():
    f = ft.fit(wall(120), SPEC, blob(0.3, 0.8, 0.7, 0.95))
    assert f.verdict == "ok" and f.score >= 0.5
    assert f.best.subject_overlap == 0.0
    assert f.fixes == [] and f.subject_area == pytest.approx(0.4 * 0.15, abs=0.01)


def test_busy_checkerboard_is_rejected_with_the_busy_fix():
    f = ft.fit(checker(), SPEC)
    assert f.verdict == "reject" and f.score < 0.3
    assert any("too busy" in s for s in f.fixes)
    assert f.subject_area is None and f.best.subject_overlap is None  # no mask: unknown, not 0


def test_subject_covering_the_middle_is_rejected_with_the_looser_shot_fix():
    f = ft.fit(wall(120), SPEC, blob(0.05, 0.10, 0.95, 0.80))
    assert f.verdict == "reject"
    assert f.subject_area > ft.SUBJECT_AREA
    assert any("looser shot" in s for s in f.fixes)


def test_very_bright_photo_with_light_text_gets_the_brightness_fix():
    f = ft.fit(wall(245), SPEC)
    assert f.text_light is True and f.brightness > 0.9
    assert f.verdict == "reject"
    assert "photo is too bright for light text; use darker text or darken the photo" in f.fixes


def test_very_dark_photo_with_dark_text_gets_the_brightness_fix():
    f = ft.fit(wall(20), TextSpec("still shipping", colours=("black", "white")))
    assert f.text_light is False
    assert "photo is too dark for dark text; use lighter text or brighten the photo" in f.fixes


def test_no_mask_is_flagged_but_still_gives_a_verdict():
    f = ft.fit(wall(120), SPEC)
    assert f.verdict == "ok"  # a clean wall is judged even without a subject mask
    assert f.best.subject_overlap is None and f.subject_area is None
    assert any("no subject mask" in r for r in f.reasons)


def test_empty_line_is_rejected_without_a_candidate():
    f = ft.fit(wall(), TextSpec("   "))
    assert f.verdict == "reject" and f.best is None and f.score is None


def test_to_dict_is_json_and_rounded():
    d = ft.fit(wall(120), SPEC, blob(0.3, 0.8, 0.7, 0.95)).to_dict()
    assert json.loads(json.dumps(d))["verdict"] == "ok"
    assert d["best"]["x"] == round(d["best"]["x"], 4)


def test_node_writes_the_fit_and_flags_a_missing_mask(conn, settings, tiny_jpg, monkeypatch):
    from wonhee.media import path_for

    def boom(*a, **k):
        raise OSError("model not available offline")
    monkeypatch.setattr(mk, "subject", boom)
    sha = repo.save_file(conn, settings.media_dir, tiny_jpg, "image", "image/jpeg")
    rid, status = run_chain(conn, settings, [{"id": "f", "node": "photo_fit", "in": {"image": sha},
                                              "params": {"text": "still shipping", "slot": "slide-2"}}], fake_deps())
    assert status == "done"
    assert "NO_SUBJECT_MASK" in flags(conn, rid, "f")
    rep = json.loads(path_for(settings.media_dir, out(conn, rid, "f", "fit")).read_text())
    assert rep["verdict"] in ("ok", "tight", "reject") and rep["subject_mask"] is False
    assert rep["slot"] == "slide-2" and rep["best"]["subject_overlap"] is None


def test_node_flags_no_quiet_area_on_a_busy_photo(conn, settings, tmp_path, monkeypatch):
    from wonhee.media import path_for
    monkeypatch.setattr(mk, "subject", lambda im, d, kind="person": np.zeros(im.shape[:2], np.float32))
    p = tmp_path / "busy.jpg"
    Image.fromarray(checker()).save(p, quality=95)
    sha = repo.save_file(conn, settings.media_dir, p, "image", "image/jpeg")
    rid, status = run_chain(conn, settings, [{"id": "f", "node": "photo_fit", "in": {"image": sha},
                                              "params": {"text": "still shipping"}}], fake_deps())
    assert status == "done" and "NO_SUBJECT_MASK" not in flags(conn, rid, "f")
    assert "NO_QUIET_AREA" in flags(conn, rid, "f")
    rep = json.loads(path_for(settings.media_dir, out(conn, rid, "f", "fit")).read_text())
    assert rep["verdict"] == "reject" and any("too busy" in s for s in rep["fixes"])


def test_node_rejects_unknown_params():
    from wonhee.nodes.fit import FitParams
    with pytest.raises(Exception):
        FitParams(text="x", colour="red")
