"""The vibe grade ladder: no-GPU runs, the range step, readability checks, mask failures, empty tone bands."""
import json
import types
from contextlib import nullcontext

import cv2
import numpy as np
import pytest

from tests.test_filters import SKIN
from tests.test_vibe import night
from wonhee.edit import mask as mk
from wonhee.edit import style
from wonhee.nodes import vibe


def face_photo() -> np.ndarray:
    im = np.full((240, 320, 3), 150, np.uint8)
    im[60:180, 120:200] = SKIN
    return im


def no_mask(monkeypatch):
    monkeypatch.setattr(vibe, "subject_mask", lambda im, p, d, guard=False: (None, "none"))


HARD = dict(strength=1.0, tone=1.5, auto=0.0)  # the ladder alone; correct-first has its own tests


def test_ladder_runs_without_a_person_mask(monkeypatch, tmp_path):
    no_mask(monkeypatch)
    prof = style.profile([night(i) for i in range(3)], "night")
    _, m, notes = vibe.grade(face_photo(), prof, vibe.VibeParams(strength=0.3, tone=0.5), tmp_path)
    assert notes["step"] == "global" and m is None and notes["tried"] == ["global"]
    _, m, notes = vibe.grade(face_photo(), prof, vibe.VibeParams(**HARD), tmp_path)
    assert notes["tried"][:2] == ["global", "gentler"] and len(notes["tried"]) >= 3  # it kept climbing, no GPU
    assert not set(notes["tried"]) & {"soft mask", "mask"}  # mask rungs need a mask
    assert m is None and not notes.get("unreadable")


def test_range_step_rescues_crushed_skin_and_hands_on_no_mask(monkeypatch, tmp_path):
    no_mask(monkeypatch)
    prof = style.profile([night(i) for i in range(3)], "night")
    out, m, notes = vibe.grade(face_photo(), prof, vibe.VibeParams(**HARD), tmp_path)
    assert notes["step"] == "range" and m is None and not notes.get("unreadable")  # blur effects must not be exempted on those pixels
    glob = style.match(face_photo(), prof.stats, 1.0, tone=1.5, colour=1.0, skin_protect=0.6)
    assert out[100:140, 140:180].mean() > glob[100:140, 140:180].mean() + 3  # face brighter than the plain grade
    assert abs(float(out[10:30, 10:30].mean()) - float(glob[10:30, 10:30].mean())) < 2  # background as graded


def test_range_step_uses_the_mask_when_one_exists_only_after_it(monkeypatch, tmp_path):
    face = np.zeros((240, 320), np.float32)
    face[60:180, 120:200] = 1
    monkeypatch.setattr(vibe, "subject_mask", lambda im, p, d, guard=False: (face, "sam2:person"))
    prof = style.profile([night(i) for i in range(3)], "night")
    _, _, notes = vibe.grade(face_photo(), prof, vibe.VibeParams(**HARD), tmp_path)
    assert notes["tried"].index("range") == 2 and notes["mask"] == "sam2:person"


def test_blown_hole_is_closed():
    assert not vibe.readable((60.0, 95.0, 0.0), (60.0, 100.0, 0.0))  # 95 -> 100 used to pass (before95 >= 94)
    assert not vibe.readable((70.0, 96.5, 0.0), (80.0, 96.9, 0.05))  # clipped share up by more than 2%
    assert vibe.readable((60.0, 96.0, 0.0), (62.0, 97.4, 0.0))  # small drift is fine
    assert vibe.readable((60.0, 99.0, 0.3), (60.0, 99.5, 0.31))  # already blown stays as it was
    assert vibe.readable((50.0, 80.0), (50.0, 80.0))  # two-tuples still work
    assert not vibe.readable((57.7, 80.0, 0.0), (88.4, 95.0, 0.0))  # washed out (the studio portrait case)
    assert vibe.readable((70.0, 90.0, 0.0), (84.0, 95.0, 0.0))  # a brighter look within reason is fine


def test_dark_check_has_an_absolute_cap():
    assert not vibe.readable((70.0, 85.0, 0.0), (44.0, 85.0, 0.0))  # 0.6*70 = 42 allowed it; 25 L* drop is too much
    assert vibe.readable((70.0, 85.0, 0.0), (46.0, 85.0, 0.0))
    assert vibe.readable((20.0, 40.0, 0.0), (17.0, 40.0, 0.0))  # a dark subject only has to stay about there


def test_unreadable_flag_and_least_bad_step(monkeypatch, tmp_path):
    no_mask(monkeypatch)
    prof = style.profile([night(i) for i in range(3)], "night")
    monkeypatch.setattr(vibe, "readable", lambda b, a: False)
    seen = []
    real = vibe.badness
    monkeypatch.setattr(vibe, "badness", lambda b, a: seen.append(real(b, a)) or real(b, a))
    _, m, notes = vibe.grade(face_photo(), prof, vibe.VibeParams(**HARD), tmp_path)
    assert notes["unreadable"] is True and notes["step"] == notes["tried"][seen.index(min(seen))]
    assert m is None  # the least-bad rung here is a maskless one


def test_notes_are_plain_json(monkeypatch, tmp_path):
    face = np.zeros((240, 320), np.float32)
    face[60:180, 120:200] = 1
    monkeypatch.setattr(vibe, "subject_mask", lambda im, p, d, guard=False: (face, "sam2:person"))
    prof = style.profile([night(i) for i in range(3)], "night")
    _, notes = vibe.render_notes(face_photo(), prof, vibe.VibeParams(**HARD), tmp_path)
    json.dumps(notes)
    assert all(type(x) is float for x in notes["subject_L"])
    assert vibe._plain({"a": np.float32(1.5), "b": np.bool_(True), "c": [np.int64(3)]}) == {"a": 1.5, "b": True, "c": [3]}
    assert type(vibe._plain({"a": np.float32(1.5)})["a"]) is float


def test_empty_source_band_adds_no_cast():
    src = np.full((120, 160, 3), 100, np.uint8)  # neutral grey, no highlights
    s = style.measure(src)
    assert s.bands["high"].share == 0
    tgt = s.model_copy(update={"lum": [min(v + 45, 100) for v in s.lum],
                               "bands": {**s.bands, "high": style.Band(a=20.0, b=20.0, share=0.2)}})
    out = style.to_lab(style.match(src, tgt, 1.0))
    assert abs(float(out[..., 1].mean())) < 3 and abs(float(out[..., 2].mean())) < 3  # no pink cast
    assert out[..., 0].mean() > style.to_lab(src)[..., 0].mean() + 10  # the tone move still happened


def fake_gpu(monkeypatch, masks):
    class Batch(dict):
        def to(self, _):
            return self
    monkeypatch.setattr(mk, "boxes_for", lambda im, prompt, **k: [([0.0, 0.0, 5.0, 5.0], 0.9)])
    sp = lambda **k: Batch(original_sizes=None)  # noqa: E731
    sp.post_process_masks = lambda pred, sizes, binarize=True: [masks]
    monkeypatch.setattr(mk, "_gpu_models", lambda: {"sp": sp, "torch": types.SimpleNamespace(no_grad=nullcontext),
                                                    "sm": lambda **k: types.SimpleNamespace(pred_masks=_T())})


class _T:
    def cpu(self):
        return self


def test_all_zero_sam_mask_is_none_and_onnx_runs(monkeypatch, tmp_path):
    im = face_photo()
    fake_gpu(monkeypatch, np.zeros((1, 1, *im.shape[:2])))
    assert mk.subject_text(im, "plush toy") is None
    onnx = np.zeros(im.shape[:2], np.float32)
    onnx[60:180, 120:200] = 1
    monkeypatch.setattr(mk, "gpu_available", lambda: True)
    monkeypatch.setattr(mk, "subject", lambda im, d, kind, **k: onnx)
    m, how = vibe.subject_mask(im, vibe.VibeParams(mask="object", mask_prompt="plush toy"), tmp_path)
    assert how == "onnx:object" and m is onnx
    ok = np.zeros((1, 1, *im.shape[:2]))
    ok[..., 60:180, 120:200] = 1
    fake_gpu(monkeypatch, ok)
    assert mk.subject_text(im, "plush toy").max() > 0.5


def test_gpu_exception_falls_back_to_onnx_then_to_no_mask(monkeypatch, tmp_path):
    im = face_photo()
    onnx = np.zeros(im.shape[:2], np.float32)
    onnx[60:180, 120:200] = 1
    monkeypatch.setattr(mk, "gpu_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(mk, "subject_text", boom)
    monkeypatch.setattr(mk, "subject", lambda im, d, kind, **k: onnx)
    prof = style.profile([night(i) for i in range(3)], "night")
    _, _, notes = vibe.grade(im, prof, vibe.VibeParams(**HARD), tmp_path)  # guard path (auto)
    assert notes["mask"] == "onnx:person" and "out of memory" in notes["mask_error"]
    json.dumps(notes)

    def offline(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(mk, "subject", offline)
    _, m, notes = vibe.grade(im, prof, vibe.VibeParams(mask="person", **HARD), tmp_path)
    assert notes["mask"] == "none" and "offline" in notes["mask_error"] and m is None
    assert notes["tried"][0] == "global"  # and the ladder still ran


def test_gpu_models_load_once_and_publish_when_ready(monkeypatch):
    monkeypatch.setattr(mk, "_gpu_ready", False)
    monkeypatch.setattr(mk, "_gpu", {})
    import sys
    fake_t = types.SimpleNamespace()
    cls = type("M", (), {"from_pretrained": staticmethod(lambda *a, **k: types.SimpleNamespace(
        to=lambda self_dev: types.SimpleNamespace(eval=lambda: "model")))})
    fake_tf = types.SimpleNamespace(AutoModelForZeroShotObjectDetection=cls, AutoProcessor=cls, Sam2Model=cls,
                                    Sam2Processor=cls)
    monkeypatch.setitem(sys.modules, "torch", fake_t)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setattr(mk, "_pretrained", lambda c, i: c.from_pretrained(i))
    g = mk._gpu_models()
    assert mk._gpu_ready and set(g) == {"torch", "dp", "dm", "sp", "sm"}
    monkeypatch.setattr(mk, "_gpu_ready", False)
    monkeypatch.setattr(mk, "_gpu", {})

    def fail(c, i):
        raise OSError("offline")
    monkeypatch.setattr(mk, "_pretrained", fail)
    with pytest.raises(OSError):
        mk._gpu_models()
    assert not mk._gpu_ready and not mk._gpu  # nothing half-published


def test_correct_first_notes_and_params(monkeypatch, tmp_path):
    no_mask(monkeypatch)
    calls = []
    monkeypatch.setattr(vibe.filters, "normalize", lambda im, a: calls.append(a) or (im, {"graded": False}),
                        raising=False)
    prof = style.profile([night(i) for i in range(3)], "night")
    _, _, notes = vibe.grade(face_photo(), prof, vibe.VibeParams(strength=0.3, tone=0.5), tmp_path)
    assert calls == [0.5] and notes["correct"] == {"graded": False}
    vibe.grade(face_photo(), prof, vibe.VibeParams(strength=0.3, auto=0.0), tmp_path)
    assert calls == [0.5]  # auto=0 skips it
    p = vibe.VibeParams(undertone="rose", undertone_strength=0.3)
    assert p.undertone == "rose" and vibe.VibeParams().undertone == "none"
