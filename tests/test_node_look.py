import subprocess

import pytest
from PIL import Image, ImageStat

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain, store_video
from wonhee import media, procenv
from wonhee.db import repo
from wonhee.edit.looks import PRESETS, filter_chain


def test_every_preset_builds_and_ends_in_a_pixel_format():
    for p in PRESETS:
        assert filter_chain(p, video=True).endswith("format=yuv420p")
        assert filter_chain(p, video=False).endswith("format=yuvj444p")


def test_grain_is_temporal_for_video_only_and_overridable():
    assert "noise=c0s=24:c0f=t" in filter_chain("grain", video=True)
    assert "c0f=t" not in filter_chain("grain", video=False)
    assert "noise" not in filter_chain("grain", video=True, grain=0)
    assert "noise=c0s=9" in filter_chain("moody", video=True, grain=9)


def test_zero_intensity_is_neutral():
    assert filter_chain("moody", 0, video=True) == "format=yuv420p"


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown look"):
        filter_chain("vaporwave", video=True)


def _lum(path):
    return ImageStat.Stat(Image.open(path).convert("L")).mean[0]


@pytest.fixture
def gray_jpg(tmp_path):
    p = tmp_path / "in.jpg"
    Image.new("RGB", (160, 284), (128, 140, 150)).save(p, quality=95)
    return p


def test_every_preset_renders_a_still(gray_jpg, tmp_path):
    stamp = tmp_path / "s.txt"
    stamp.write_text("'25 10 17")
    for p in PRESETS:
        dst = tmp_path / f"{p}.jpg"
        vf = filter_chain(p, video=False, datestamp_file=stamp if p == "digicam" else None, height=284)
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(gray_jpg), "-vf", vf, "-frames:v", "1",
                            str(dst)], capture_output=True, text=True, env=procenv.clean_env())
        assert r.returncode == 0, (p, r.stderr)
    assert _lum(tmp_path / "moody.jpg") < _lum(gray_jpg) - 5  # underexposed


def test_look_image_node_darkens(conn, settings, gray_jpg):
    sha = repo.save_file(conn, settings.media_dir, gray_jpg, "image", "image/jpeg")
    rid, status = run_chain(conn, settings, [{"id": "l", "node": "look_image", "in": {"image": sha},
                                              "params": {"preset": "moody"}}], fake_deps())
    assert status == "done"
    new = out(conn, rid, "l", "image")
    assert new != sha and _lum(media.path_for(settings.media_dir, new)) < _lum(gray_jpg) - 5


def test_look_video_node_keeps_duration_and_audio(conn, settings, tiny_video):
    sha = store_video(conn, settings, tiny_video)
    rid, status = run_chain(conn, settings, [{"id": "l", "node": "look", "in": {"video": sha},
                                              "params": {"preset": "digicam", "encoder": "x264",
                                                         "datestamp": "'25 10 17"}}], fake_deps())
    assert status == "done" and "ENCODER_FALLBACK" not in flags(conn, rid, "l")
    src, dst = media.path_for(settings.media_dir, sha), media.path_for(settings.media_dir, out(conn, rid, "l", "video"))
    assert abs(media.probe(dst).duration_ms - media.probe(src).duration_ms) < 200
    assert media.has_audio(dst) == media.has_audio(src)
