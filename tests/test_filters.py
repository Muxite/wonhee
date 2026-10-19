import json

import cv2
import numpy as np
import pytest

from tests.fakes import fake_deps
from tests.test_vibe import _write, day, night
from wonhee import filterstore, localllm, service
from wonhee.edit import filters as F
from wonhee.edit import mask as mk
from wonhee.edit import style
from wonhee.media import path_for
from wonhee.nodes import vibe
from wonhee.nodes.filter import Move, apply_moves

SKIN = (120, 150, 200)  # BGR, a mid skin tone
SKY = (220, 170, 110)
LEAF = (60, 150, 70)


def patches() -> np.ndarray:
    im = np.zeros((120, 360, 3), np.uint8)
    for i, c in enumerate((SKIN, SKY, LEAF)):
        im[:, i * 120:(i + 1) * 120] = c
    return im


def hue(px) -> float:
    lab = cv2.cvtColor(np.array([[px]], np.uint8).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[0, 0]
    return float(np.degrees(np.arctan2(lab[2], lab[1])))


def chroma(px) -> float:
    lab = cv2.cvtColor(np.array([[px]], np.uint8).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[0, 0]
    return float(np.hypot(lab[1], lab[2]))


def test_neutral_filter_and_zero_intensity_leave_the_photo_alone():
    im = day()
    assert np.abs(F.apply(im, F.Filter()).astype(int) - im).max() <= 2
    assert np.abs(F.apply(im, F.PACK["disposable"][1], intensity=0).astype(int) - im).max() <= 2


@pytest.mark.parametrize("kw", [dict(contrast=1), dict(contrast=-1), dict(shadows=1, highlights=-1),
                                dict(shadows=-1, highlights=1, fade=1, rolloff=1)])
def test_tone_curve_never_reverses(kw):
    c = F.tone_curve(F.Filter(**kw))
    assert (np.diff(c) >= -1e-6).all() and 0 <= c.min() and c.max() <= 1


def test_every_pack_filter_renders_and_is_deterministic():
    im = day()
    for name, (_, f) in F.PACK.items():
        a, b = F.apply(im, f), F.apply(im, f)
        assert a.shape == im.shape and np.array_equal(a, b), name
        assert np.abs(a.astype(int) - im).mean() > 1, f"{name} does nothing"


def test_skin_protection_keeps_skin_hue_under_a_cool_grade():
    im = patches()
    cold = F.Filter(temperature=-1, skin_protect=0)
    kept = F.Filter(temperature=-1, skin_protect=1)
    drift = lambda f: abs(hue(F.apply(im, f)[60, 60]) - hue(SKIN))  # noqa: E731
    assert drift(kept) < drift(cold) / 3
    assert chroma(F.apply(im, kept)[60, 180]) > chroma(SKY) + 5  # the sky still takes the grade (bluer)


def test_greens_and_blues_only_touch_their_hues():
    out = F.apply(patches(), F.Filter(greens=-1, skin_protect=0))
    assert chroma(out[60, 300]) < chroma(LEAF) * 0.5
    assert chroma(out[60, 180]) == pytest.approx(chroma(SKY), rel=0.1)
    out = F.apply(patches(), F.Filter(blues=1, skin_protect=0))
    assert chroma(out[60, 180]) > chroma(SKY) * 1.3


def test_cube_lut_matches_the_filter_and_identity_is_identity():
    lines = F.cube(F.Filter()).splitlines()
    assert lines[1] == "LUT_3D_SIZE 33" and len(lines) == 4 + 33 ** 3
    first, red = [float(x) for x in lines[4].split()], [float(x) for x in lines[4 + 32].split()]
    assert first == pytest.approx([0, 0, 0], abs=0.01) and red == pytest.approx([1, 0, 0], abs=0.02)
    warm = F.cube(F.Filter(temperature=1, skin_protect=0)).splitlines()
    mid = 4 + 16 + 16 * 33 + 16 * 33 * 33  # grey 0.5
    r, g, b = (float(x) for x in warm[mid].split())
    assert r > b + 0.05


def test_guard_and_verdicts_read_the_checks():
    src = np.full((100, 100, 3), 200, np.uint8)
    out = np.full((100, 100, 3), 255, np.uint8)
    c = F.checks(src, out)
    assert c["clipped"][1] == 100
    fixed, notes = F.guard(F.Filter(exposure=0.5), [c])
    assert fixed.exposure == pytest.approx(0.35) and fixed.highlights < 0 and notes
    assert "highlights blowing out" in F.verdicts(c) and "skin colour natural" in F.verdicts(c)


def test_moves_are_clamped_steps_and_set_hues():
    f, done = apply_moves(F.Filter(grain=0.9), [
        Move(knob="grain", direction="up", amount="a lot"),
        Move(knob="shadow_tone", direction="up", amount="some", hue="teal"),
        Move(knob="highlight_tone", direction="up", amount="a little"),
        Move(knob="mono", direction="up", amount="a little")])
    assert f.grain == 1.0 and f.shadow_tone == 0.2 and f.shadow_hue == "teal"
    assert f.highlight_hue == "orange" and f.mono and len(done) == 4


def test_filterstore_saves_keeps_history_and_reserves_the_pack(tmp_path):
    with pytest.raises(ValueError, match="pack filter"):
        filterstore.save(tmp_path, "film-warm", {"filter": F.Filter().model_dump()})
    filterstore.save(tmp_path, "My Vibe!", {"filter": F.Filter(grain=0.2).model_dump(), "description": "a"})
    filterstore.save(tmp_path, "my-vibe", {"filter": F.Filter(grain=0.4).model_dump(), "description": "b"})
    f, rec = filterstore.load(tmp_path, "my vibe")
    assert f.grain == 0.4 and rec["source"] == "saved" and rec["previous"][-1]["filter"]["grain"] == 0.2
    assert filterstore.load(tmp_path, "matte")[1]["source"] == "pack"
    with pytest.raises(ValueError, match="no filter called"):
        filterstore.load(tmp_path, "nope")


def _fake_models(monkeypatch, replies: dict[str, list]):
    sent = []

    def post(s, body):
        name = body["format"]["title"]
        sent.append((name, body))
        return {"message": {"content": json.dumps(replies[name].pop(0))}, "prompt_eval_count": 10}
    monkeypatch.setattr(localllm, "_post", post)
    return sent


def test_filter_make_drafts_tunes_saves_and_tweaks(conn, settings, tmp_path, monkeypatch):
    photos = [_write(tmp_path, "a.jpg", day()), _write(tmp_path, "b.jpg", night(1))]
    sent = _fake_models(monkeypatch, {
        "Draft": [{"base": "film-cool", "notes": "cool and soft",
                   "moves": [{"knob": "grain", "direction": "up", "amount": "some"},
                             {"knob": "temperature", "direction": "down", "amount": "a lot"}]}],
        "Adjust": [{"score": 3, "verdict": "too cold", "moves": [{"knob": "temperature", "direction": "up",
                                                                  "amount": "a little"}]},
                   {"score": 4, "verdict": "good", "moves": []},
                   {"score": 4, "verdict": "warmer now", "moves": []}],
        "Tweak": [{"notes": "warmer", "moves": [{"knob": "temperature", "direction": "up", "amount": "a lot"}]}]})
    text, out = service.filter_make(conn, settings, fake_deps(), "rainy cafe, soft and cool", photos,
                                    name="Rainy Cafe", rounds=3)
    assert "saved" in text and "round 2: 4/5" in text and "round 3" not in text  # stopped when done
    f, rec = filterstore.load(settings.home, "rainy-cafe")
    cool = F.PACK["film-cool"][1].temperature
    assert f.temperature == pytest.approx(cool - 0.35 + 0.1) and f.grain == pytest.approx(0.45)
    assert rec["best_round"] == 2 and (out / "round-1.jpg").exists() and (out / "01-a.jpg").exists()
    review = [b for n, b in sent if n == "Adjust"][0]
    assert len(review["messages"][1]["images"]) == 1 and "skin colour natural" in review["messages"][1]["content"]

    text, _ = service.filter_make(conn, settings, fake_deps(), "a bit warmer", photos, name="rainy-cafe",
                                  rounds=1, start="rainy-cafe")
    f2, rec2 = filterstore.load(settings.home, "rainy-cafe")
    assert f2.temperature == pytest.approx(f.temperature + 0.35) and f2.grain == f.grain  # only what changed
    assert rec2["description"].startswith("rainy cafe, soft and cool; then: a bit warmer")
    assert rec2["previous"][-1]["filter"]["temperature"] == f.temperature

    text, out = service.filter_apply(conn, settings, fake_deps(), "rainy-cafe", photos, intensity=0.5)
    assert "2 of 2" in text and len(list(out.glob("*.jpg"))) == 2
    assert "rainy-cafe" in service.filter_list(settings)
    cube = tmp_path / "x.cube"
    assert "LUT written" in service.filter_show(settings, "rainy-cafe", cube=cube) and cube.exists()


def test_soft_masks_are_capped_and_have_no_hard_edge():
    im = day((200, 200))
    m = np.zeros((200, 200), np.float32)
    m[50:150, 50:150] = 1
    s = mk.soften(m, im, feather=0.02, opacity=0.6)
    assert s.max() <= 0.6 + 1e-6 and np.abs(np.diff(s, axis=1)).max() < 0.1


def test_mask_policy_tries_without_a_mask_first(monkeypatch, tmp_path):
    prof = style.profile([night(i) for i in range(3)], "night")
    im = np.full((240, 320, 3), 150, np.uint8)
    im[60:180, 120:200] = SKIN  # a "face"
    face = np.zeros(im.shape[:2], np.float32)
    face[60:180, 120:200] = 1
    monkeypatch.setattr(vibe, "subject_mask", lambda im, p, d, guard=False: (face, "sam2:person"))
    gentle = vibe.VibeParams(strength=0.3, tone=0.5)
    _, m, notes = vibe.grade(im, prof, gentle, tmp_path)
    assert notes["step"] == "global" and m is None  # readable without a mask: none blended
    hard = vibe.VibeParams(strength=1.0, tone=1.5, auto=0.0)
    _, m, notes = vibe.grade(im, prof, hard, tmp_path)
    # the range step (skin only, mask not handed on) now sits before the person-mask rungs
    assert notes["step"] in ("range", "soft mask", "mask")
    assert (m is None) == (notes["step"] == "range") and (m is None or m.max() <= 0.65 + 1e-6)
    _, m, notes = vibe.grade(im, prof, vibe.VibeParams(strength=1.0, tone=1.5, mask_policy="off"), tmp_path)
    assert m is None and notes["mask"] == "none"


# --- wave 2: store robustness, usage log, ranges, slugs, batch errors ----------------------------------------------

def _write_rec(home, name, rec):
    p = filterstore.folder(home) / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec))
    return p


def test_old_saved_json_with_unknown_key_and_no_dates_loads_and_resaves(tmp_path):
    _write_rec(tmp_path, "old", {"name": "old", "description": "d", "filter": {"grain": 0.3, "sepia": 0.5}})
    f, rec = filterstore.load(tmp_path, "old")
    assert f.grain == 0.3 and rec["dropped"] == ["sepia"]
    filterstore.save(tmp_path, "old", {"filter": F.Filter(grain=0.5).model_dump()})  # no created/updated before
    f, rec = filterstore.load(tmp_path, "old")
    assert f.grain == 0.5 and rec["previous"][-1]["updated"] is None and "dropped" not in rec


def test_corrupt_file_is_a_clear_error_saved_skips_it_and_save_recovers(tmp_path):
    p = filterstore.folder(tmp_path) / "bad.json"
    p.parent.mkdir(parents=True)
    p.write_text("{not json")
    _write_rec(tmp_path, "nofilter", {"name": "nofilter"})
    with pytest.raises(ValueError, match="saved filter bad is unreadable"):
        filterstore.load(tmp_path, "bad")
    with pytest.raises(ValueError, match="saved filter nofilter is unreadable"):
        filterstore.load(tmp_path, "nofilter")
    assert filterstore.saved(tmp_path) == []
    filterstore.save(tmp_path, "bad", {"filter": F.Filter().model_dump()})
    assert filterstore.load(tmp_path, "bad")[1]["source"] == "saved"


def test_usage_log_is_not_a_filter_and_median_of_last_ten(tmp_path):
    for x in (0.2, 0.9, 0.7, 0.7):
        filterstore.log_use(tmp_path, "My Vibe", {"intensity": x, "n": 1})
    filterstore.log_use(tmp_path, "matte", {"intensity": 0.5, "n": 1})
    filterstore.log_use(tmp_path, "film-warm", {"intensity": 1.5, "n": 1})
    assert [e["intensity"] for e in filterstore.uses(tmp_path, "my-vibe")] == [0.2, 0.9, 0.7, 0.7]
    assert filterstore.usual_intensity(tmp_path, "my-vibe") == 0.7
    assert filterstore.usual_intensity(tmp_path, "matte") == 0.5 and filterstore.usual_intensity(tmp_path, "x") is None
    assert (filterstore.folder(tmp_path) / "_usage.jsonl").exists() and filterstore.saved(tmp_path) == []


def test_slug_edge_cases(tmp_path):
    assert filterstore.slug("rainy_cafe  night") == "rainy-cafe-night"
    long = "a" * 47 + "_b"
    assert filterstore.slug(long) == "a" * 47 and not filterstore.slug(long).endswith("-")
    k = filterstore.slug("비 오는 카페")
    assert k.startswith("f-") and len(k) == 10 and k == filterstore.slug("비 오는 카페")
    assert filterstore.slug("Café Noir") == "cafe-noir"
    _write_rec(tmp_path, "matte", {"name": "matte", "filter": {}, "description": "mine"})
    assert "shadowed by pack" in service.filter_list(type("S", (), {"home": tmp_path})())


def test_apply_default_intensity_follows_history_and_survives_a_missing_photo(conn, settings, tmp_path):
    good = _write(tmp_path, "a.jpg", day())
    text, out = service.filter_apply(conn, settings, fake_deps(), "matte", [good])
    assert "intensity 1" in text and "usual" not in text and out is not None
    for x in (0.4, 0.6):
        service.filter_apply(conn, settings, fake_deps(), "matte", [good], intensity=x)
    text, out = service.filter_apply(conn, settings, fake_deps(), "matte", [tmp_path / "gone.jpg", good])
    assert "intensity 0.6 (your usual)" in text and "1 of 2" in text and "failed" in text and out is not None
    assert len(list(out.glob("*.jpg"))) == 1
    text, out = service.filter_apply(conn, settings, fake_deps(), "matte", [tmp_path / "gone.jpg"])
    assert out is None and "0 of 1" in text
    with pytest.raises(ValueError, match="between 0 and 2"):
        service.filter_apply(conn, settings, fake_deps(), "matte", [good], intensity=3)
    assert filterstore.uses(settings.home, "matte")[-1]["n"] == 1


def test_range_moves_apply_and_a_range_knob_without_range_is_ignored():
    f, done = apply_moves(F.Filter(), [Move(knob="range_sat", direction="down", amount="a lot", range="green"),
                                       Move(knob="range_hue", direction="up", amount="some", range="blue"),
                                       Move(knob="range_lum", direction="up", amount="a little")])
    assert f.ranges["green"].sat == pytest.approx(-0.35) and f.ranges["blue"].hue == pytest.approx(0.2)
    assert set(f.ranges) == {"green", "blue"} and "ignored" in done[-1]
    f, _ = apply_moves(f, [Move(knob="range_sat", direction="down", amount="a lot", range="green")] * 4)
    assert f.ranges["green"].sat == -1.0
    from wonhee.nodes.filter import describe, knob_help
    assert "green" in describe(f) and describe(F.Filter()) == "{}" and knob_help().count("range_hue") == 1


def test_filter_image_node_takes_auto_and_undertone_and_undertone_changes_skin(conn, settings, tmp_path):
    from wonhee.nodes.filter import FilterImageParams
    assert FilterImageParams().auto == 0.6
    im = np.full((80, 80, 3), 140, np.uint8)
    im[:] = SKIN
    src = _write(tmp_path, "skin.jpg", im)
    outs = {}
    for tone in ("none", "rose"):
        params = {**F.Filter().model_dump(), "auto": 0.0, "undertone": tone, "undertone_strength": 1.0}
        rid, _ = service.node_run(conn, settings, fake_deps(), "filter_image",
                                  {"image": repo_save(conn, settings, src)}, params)
        o = service._step_out(conn, rid, "filter_image")
        outs[tone] = style.load(path_for(settings.media_dir, o["image"]))
    assert np.abs(outs["none"].astype(int) - outs["rose"]).max() > 3


def repo_save(conn, settings, p):
    from wonhee.db import repo
    return repo.save_file(conn, settings.media_dir, p, "image", "image/jpeg")


def test_checks_compare_against_the_normalized_source(conn, settings, tmp_path, monkeypatch):
    # the normalized source is 40 levels brighter; a filter that leaves the photo as-is would look "dark" against it
    # but the guard must not pull exposure down because of the (raw) clipping difference
    seen = []
    real = F.checks
    monkeypatch.setattr(F, "normalize", lambda im, a=0.5: (np.clip(im.astype(int) + 40, 0, 255).astype(np.uint8), {}))
    monkeypatch.setattr(F, "checks", lambda a, b: (seen.append(a.mean()), real(a, b))[1])
    photo = _write(tmp_path, "a.jpg", np.full((90, 120, 3), 100, np.uint8))
    _fake_models(monkeypatch, {"Draft": [{"base": "none", "notes": "n", "moves": []}],
                               "Adjust": [{"score": 4, "verdict": "ok", "moves": []}]})
    text, _ = service.filter_make(conn, settings, fake_deps(), "plain", [photo], name="plain", rounds=1)
    assert seen and seen[0] == pytest.approx(140, abs=3)  # the normalized source, not the raw 100
    assert "guard" not in text
