import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from tests.fakes import fake_deps
from wonhee import localllm, service
from wonhee.edit import fx, style
from wonhee.inspire import looks


def night(seed: int, size=(240, 320)) -> np.ndarray:
    """Dark, blue-cast frame with a bright cyan 'practical' and dark corners."""
    rng = np.random.default_rng(seed)
    h, w = size
    im = np.zeros((h, w, 3), np.uint8)
    im[...] = (60, 30, 10)  # BGR: dark blue
    im[h // 3:h // 2, w // 3:w // 2] = (255, 255, 160)  # cyan-white light
    im = cv2.add(im, rng.integers(0, 12, im.shape, dtype=np.uint8))
    return fx.vignette(im, 0.8)


def day(size=(240, 320)) -> np.ndarray:
    h, w = size
    g = np.tile(np.linspace(90, 220, w, dtype=np.uint8), (h, 1))
    return cv2.merge([g, g, g])


def test_profile_reads_the_night_look_and_match_moves_toward_it():
    prof = style.profile([night(1), night(2)], "night")
    assert prof.stats.bands["shadow"].b < -5 and prof.stats.lum[3] < 30
    assert any("cool all over" in h for h in prof.hints)
    src = day()
    assert np.array_equal(style.match(src, prof.stats, 0.0), src) or \
        np.abs(style.match(src, prof.stats, 0.0).astype(int) - src).max() <= 2  # strength 0 = untouched
    out = style.measure(style.match(src, prof.stats, 0.8))
    before = style.measure(src)
    assert out.lum[3] < before.lum[3] - 10  # darker
    assert out.bands["mid"].b < before.bands["mid"].b - 3  # bluer


def test_mask_keeps_the_subject_closer_to_true_and_lifts_it():
    prof = style.profile([night(1)], "night")
    src = day()
    m = np.zeros(src.shape[:2], np.float32)
    m[60:180, 100:220] = 1
    out = style.match(src, prof.stats, 0.9, mask=m, subject_strength=0.2, subject_lift=0.3)
    diff = np.abs(out.astype(int) - src.astype(int)).mean(axis=2)
    assert diff[80:160, 120:200].mean() < diff[:, :60].mean()  # background moved more than the subject


def test_lens_and_print_effects():
    src = day()
    fe = fx.fisheye(src, 0.8)
    assert fe.shape == src.shape and np.abs(fe[120, 160].astype(int) - src[120, 160]).max() <= 3  # centre stays
    eb = fx.edge_blur(src, 1.0)
    assert np.array_equal(eb[110:130, 150:170], src[110:130, 150:170])  # centre untouched
    duo = fx.duotone(src, ink="#0000ff", paper="#ffffff", texture=0)
    assert duo[..., 1].max() == duo[..., 1].max() and (np.abs(duo[..., 0].astype(int) - 255) <= 1).all()  # B channel full
    ht = fx.halftone(src, ink="#ff0000", paper="#ffffff")
    assert (ht[..., 2] >= 254).all()  # red ink on white paper never lowers red
    assert fx.grain(src, 0.5, seed=3).tolist() == fx.grain(src, 0.5, seed=3).tolist()


def test_hints_need_dark_corners_to_call_it_fisheye():
    s = style.measure(day())
    s2 = s.model_copy(update={"edge_sharpness": 0.4, "vignette": 0.95})
    assert not any("fisheye" in h for h in style.hints(s2)) and any("soft background" in h for h in style.hints(s2))
    s3 = s.model_copy(update={"edge_sharpness": 0.5, "vignette": 0.5})
    assert any("fisheye" in h for h in style.hints(s3))


def test_looks_group_posts_with_close_numbers():
    profs = {1: style.profile([night(1)]), 2: style.profile([night(2)]), 3: style.profile([day()]),
             4: style.profile([night(3)])}
    assert looks.group(profs) == [[1, 2, 4]]


def _write(tmp_path, name, im):
    p = tmp_path / name
    cv2.imwrite(str(p), im)
    return p


def test_vibe_end_to_end_with_a_local_reviewer(conn, settings, tmp_path, monkeypatch):
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in range(3):
        _write(refs, f"r{i}.jpg", night(i))
    photo = _write(tmp_path, "mine.jpg", day())
    sent = []
    knobs = {"fit": 9, "reason": "darken and cool it", "strength": 3.0, "tone": 0.6, "colour": 1.0, "mask": "none",
             "subject_lift": 0, "fisheye": 0.4, "edge_blur": 0.3, "motion_blur": 0, "bloom": 0.5, "vignette": 0.6,
             "grain": 0.1, "crop": "4:5", "reshoot": ["shoot at night", "0.5x lens", "find a cyan light", "lower", "x"]}
    monkeypatch.setattr(localllm, "_post", lambda s, body: sent.append(body) or
                        {"message": {"content": json.dumps(knobs)}, "prompt_eval_count": 10})
    text, out = service.vibe(conn, settings, fake_deps(), [str(refs)], [photo], name="Night Test", guide=False)
    assert "3 reference(s)" in text and out.name.endswith("-night-test")
    assert len(sent[0]["messages"][1]["images"]) == 3  # the photo + 2 reference thumbnails
    md = (out / "report.md").read_text()
    assert "Fit 5/5" in md and md.count("\n- shoot at night") == 1 and "- x" not in md  # clamped, 4 tips max
    assert '"strength": 1.0' in md  # 3.0 clamped
    assert conn.execute("SELECT detail FROM step_flag WHERE code='REVIEW_CLAMPED'").fetchone()[0] == "strength 3.0 -> 1.0"
    edited = cv2.imread(str(out / "01-mine.jpg"))
    assert abs(edited.shape[0] / edited.shape[1] - 5 / 4) < 0.02  # cropped 4:5
    assert style.measure(edited).bands["mid"].b < style.measure(day()).bands["mid"].b
    assert (out / "sheet.jpg").exists()


def test_shot_guide_is_given_the_measured_facts(conn, settings, tmp_path, fake_llm):
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in range(2):
        _write(refs, f"r{i}.jpg", night(i))
    fake_llm.queue(ShotGuide=[{"summary": "dark cyan wide snapshot", "lens": "0.5x", "camera_position": "low",
                               "distance": "40 cm", "light": ["monitor"], "exposure_and_motion": "-1 stop",
                               "colour": "cyan", "composition": ["prop near"], "subject_direction": ["deadpan"],
                               "places_like_this": ["desk at night"], "avoid": ["daylight"], "edit_finish": "cool",
                               "checklist": ["lock exposure"]}])
    _, out = service.vibe(conn, settings, fake_deps(), [str(refs)], [], name="n")
    call = fake_llm.calls()[0]
    assert "L* percentiles" in call["prompt"] and "cool all over" in call["prompt"] and len(call["images"]) == 2
    assert "**dark cyan wide snapshot**" in (out / "report.md").read_text()


def test_poster_node(conn, settings, tmp_path):
    from wonhee.media import path_for
    photo = _write(tmp_path, "p.jpg", day())
    rid, _ = service.node_run(conn, settings, fake_deps(), "poster", {"image": str(photo)},
                              {"title": "Signal Shielding", "label1": "Hardware:", "value1": "foil", "body": "x " * 40})
    sha = service._step_out(conn, rid, "poster")["image"]
    im = cv2.imread(str(path_for(settings.media_dir, sha)))
    assert im.shape[:2] == (1350, 1080)
    assert im[30:110, 30:500].min() < 60  # dark title type top-left


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")


@needs_ffmpeg
def test_audio_clean_hits_the_loudness_target_and_tames_peaks(conn, settings, tmp_path, monkeypatch):
    from wonhee.edit import audio as au
    src = tmp_path / "loud.mp4"
    # a hot, clipping tone with hiss, plus a black video track
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=6",
                    "-f", "lavfi", "-i", "sine=f=220:d=6,volume=12",
                    "-f", "lavfi", "-i", "anoisesrc=color=white:amplitude=0.05:d=6",
                    "-filter_complex", "[1][2]amix=inputs=2:normalize=0[a]", "-map", "0:v", "-map", "[a]",
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)], check=True)
    monkeypatch.setattr(au, "rnn_model", lambda d: None)  # offline: falls back to afftdn
    text = service.audio(conn, settings, fake_deps(), src)
    assert "peaks hit" in text and "cleaned (afftdn)" in text
    after, _ = au.measure(src.with_name("loud.clean.mp4"))
    assert abs(after.lufs - -14) < 1.0 and after.true_peak <= -1.0
    assert conn.execute("SELECT 1 FROM step_flag WHERE code='DENOISE_FALLBACK'").fetchone()


def test_subject_mask_prefers_the_gpu_text_mask_and_falls_back(monkeypatch, tmp_path):
    from wonhee.edit import mask as mk
    from wonhee.nodes import vibe as vb
    im = day()
    calls = []
    monkeypatch.setattr(mk, "gpu_available", lambda: True)
    monkeypatch.setattr(mk, "subject_text", lambda im, p: calls.append(p) or np.ones(im.shape[:2], np.float32))
    monkeypatch.setattr(mk, "subject", lambda im, d, kind: np.zeros(im.shape[:2], np.float32))
    assert vb.subject_mask(im, vb.VibeParams(mask_prompt="plush toy"), tmp_path)[1] == "sam2:plush toy"
    assert vb.subject_mask(im, vb.VibeParams(mask="person"), tmp_path)[1] == "sam2:person"
    assert vb.subject_mask(im, vb.VibeParams(mask="object"), tmp_path)[1] == "onnx:object"  # nothing named
    monkeypatch.setattr(mk, "subject_text", lambda im, p: None)  # the prompt found nothing
    assert vb.subject_mask(im, vb.VibeParams(mask_prompt="unicorn"), tmp_path)[1] == "onnx:object"
    monkeypatch.setattr(mk, "gpu_available", lambda: False)
    assert vb.subject_mask(im, vb.VibeParams(mask="person"), tmp_path)[1] == "onnx:person"
    assert vb.subject_mask(im, vb.VibeParams(), tmp_path) == (None, "none")


@needs_ffmpeg
def test_audio_isolate_cleans_the_voice_and_ducks_the_bed(conn, settings, tmp_path, monkeypatch):
    from wonhee.edit import audio as au
    from wonhee.edit import separate
    src = tmp_path / "street.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=4,volume=4", "-c:a", "aac",
                    str(src)], check=True)
    split = []

    def fake_split(s, voice, bed, model="htdemucs"):  # voice = the input, bed = a quiet tone
        split.append(s)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(s), "-ac", "2", "-ar", "44100", str(voice)], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=f=880:d=4", "-ac", "2", "-ar",
                        "44100", str(bed)], check=True)

    monkeypatch.setattr(separate, "available", lambda: True)
    monkeypatch.setattr(separate, "split_voice", fake_split)
    monkeypatch.setattr(au, "rnn_model", lambda d: None)
    text = service.audio(conn, settings, fake_deps(), src, isolate=True, music_db=-15)
    assert len(split) == 1 and "cleaned (demucs+afftdn)" in text
    after, _ = au.measure(src.with_name("street.clean.mp4"))
    assert abs(after.lufs - -14) < 1.0
