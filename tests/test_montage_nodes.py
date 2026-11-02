"""join_videos / video_fx / montage / sfx nodes (docs/specs/2026-10-28-slopcut-port-design.md)."""
import shutil
import subprocess
import wave

import pytest
from PIL import Image
from pydantic import ValidationError

import wonhee.nodes.montage as M  # registers the nodes
from tests.fakes import fake_deps
from tests.test_nodes_subs import out, run_chain, store_video
from wonhee import media
from wonhee.db import repo
from wonhee.edit.montage import fx as fxmod
from wonhee.edit.montage.transitions import BUILTIN, CUSTOM
from wonhee.nodes.images import Images, Slide, save_images

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture(autouse=True)
def montage_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("WONHEE_MONTAGE_CACHE_DIR", str(tmp_path / "mcache"))


def probe(settings, sha):
    return media.probe(media.path_for(settings.media_dir, sha))


@pytest.fixture
def silent_video(tmp_path):
    p = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=yellow:s=320x568:d=1.5:r=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


@pytest.mark.parametrize("name", ["fade", sorted(CUSTOM)[0]])
def test_join_videos_length_and_size(conn, settings, tiny_video, silent_video, name):
    a = store_video(conn, settings, tiny_video)       # 3.0 s with audio
    b = store_video(conn, settings, silent_video)     # 1.5 s without audio
    assert name in BUILTIN or name in CUSTOM
    rid, status = run_chain(conn, settings, [{"id": "j", "node": "join_videos", "in": {"video": a, "video_b": b},
                                              "params": {"transition": name, "duration_s": 0.5}}], fake_deps())
    assert status == "done"
    sha = out(conn, rid, "j", "video")
    pr = probe(settings, sha)
    assert abs(pr.duration_ms / 1000 - (3.0 + 1.5 - 0.5)) < 0.15
    assert (pr.width, pr.height) == (320, 568)
    assert media.has_audio(media.path_for(settings.media_dir, sha))   # A's sound survives a silent B


def test_join_videos_both_silent_has_no_audio(conn, settings, silent_video):
    s = store_video(conn, settings, silent_video)
    rid, status = run_chain(conn, settings, [{"id": "j", "node": "join_videos", "in": {"video": s, "video_b": s}}],
                            fake_deps())
    assert status == "done"
    sha = out(conn, rid, "j", "video")
    assert not media.has_audio(media.path_for(settings.media_dir, sha))
    assert abs(probe(settings, sha).duration_ms / 1000 - 2.5) < 0.15


def test_join_rejects_unknown_transition_and_extra_params():
    with pytest.raises(ValidationError):
        M.JoinParams(transition="no_such_transition")
    with pytest.raises(ValidationError):
        M.JoinParams(bogus=1)


def test_join_is_deterministic(conn, settings, tiny_video, silent_video):
    a, b = store_video(conn, settings, tiny_video), store_video(conn, settings, silent_video)
    shas = []
    for seed in (1, 1):
        rid, st = run_chain(conn, settings, [{"id": "j", "node": "join_videos", "in": {"video": a, "video_b": b},
                                              "params": {"transition": "wipeleft", "seed": seed}}], fake_deps())
        assert st == "done"
        shas.append(out(conn, rid, "j", "video"))
    assert shas[0] == shas[1]


@pytest.mark.parametrize("name", ["negative", "tile_grid", "datamosh", None])
def test_video_fx_keeps_frame_size(conn, settings, tiny_video, name):
    if name is None:
        name = fxmod.names("color")[0]
    sha = store_video(conn, settings, tiny_video)
    rid, status = run_chain(conn, settings, [{"id": "f", "node": "video_fx", "in": {"video": sha},
                                              "params": {"fx": name, "seed": 3, "intensity": 1.5}}], fake_deps())
    assert status == "done"
    new = out(conn, rid, "f", "video")
    pr = probe(settings, new)
    assert (pr.width, pr.height) == (320, 568)
    assert abs(pr.duration_ms - 3000) < 200
    assert media.has_audio(media.path_for(settings.media_dir, new))


def test_video_fx_rejects_unknown_name_and_bad_intensity():
    with pytest.raises(ValidationError):
        M.FxParams(fx="not_an_fx")
    with pytest.raises(ValidationError):
        M.FxParams(fx=fxmod.names()[0], intensity=5)


def _slides(conn, settings, tmp_path, n=3):
    items = []
    for i in range(n):
        p = tmp_path / f"s{i}.jpg"
        Image.new("RGB", (240, 300), (40 + 70 * i, 90, 200 - 60 * i)).save(p, quality=90)
        items.append(Slide(sha=repo.save_file(conn, settings.media_dir, p, "image", "image/jpeg")))
    return save_images(conn, settings.media_dir, Images(items=items))


def test_montage_size_duration_and_plan_round_trip(conn, settings, tmp_path):
    imgs = _slides(conn, settings, tmp_path)
    params = {"style": "calm", "length_s": 4, "seed": 5, "fps": 24, "size": [270, 480], "sfx": False}
    rid, status = run_chain(conn, settings, [{"id": "m", "node": "montage", "in": {"images": imgs},
                                              "params": params}], fake_deps())
    assert status == "done"
    video, plan = out(conn, rid, "m", "video"), out(conn, rid, "m", "plan")
    pr = probe(settings, video)
    assert (pr.width, pr.height) == (270, 480) and abs(pr.duration_ms / 1000 - 4.0) < 0.15
    # the plan is on a text port, and feeding it back (no images, different params) gives identical bytes
    rid2, status2 = run_chain(conn, settings, [{"id": "m", "node": "montage", "in": {"plan": plan},
                                                "params": {"seed": 99}}], fake_deps())
    assert status2 == "done"
    assert out(conn, rid2, "m", "video") == video and out(conn, rid2, "m", "plan") == plan


def test_montage_same_seed_same_plan_and_other_seed_differs(conn, settings, tmp_path):
    imgs = _slides(conn, settings, tmp_path)
    plans = []
    for seed in (1, 1, 2):
        rid, st = run_chain(conn, settings, [{"id": "m", "node": "montage", "in": {"images": imgs},
                                              "params": {"style": "hype", "length_s": 4, "seed": seed, "fps": 24,
                                                         "size": [180, 320], "sfx": False}}], fake_deps())
        assert st == "done"
        plans.append(out(conn, rid, "m", "plan"))
    assert plans[0] == plans[1] and plans[0] != plans[2]


def test_montage_with_audio_input(conn, settings, tmp_path):
    imgs = _slides(conn, settings, tmp_path, 2)
    rid, st = run_chain(conn, settings, [{"id": "w", "node": "sfx", "params": {"name": M.sounds.names()[0]}},
                                         {"id": "m", "node": "montage", "in": {"images": imgs, "audio": "w.file"},
                                          "params": {"length_s": 3, "fps": 24, "size": [180, 320], "sfx": False}}],
                        fake_deps())
    assert st == "done"
    assert media.has_audio(media.path_for(settings.media_dir, out(conn, rid, "m", "video")))


def test_montage_needs_images_or_plan_and_rejects_bad_params(conn, settings):
    rid, st = run_chain(conn, settings, [{"id": "m", "node": "montage", "in": {}}], fake_deps())
    assert st != "done"
    for bad in ({"style": "disco"}, {"size": [101, 200]}, {"nope": 1}, {"length_s": 0}):
        with pytest.raises(ValidationError):
            M.MontageParams(**bad)


def test_sfx_wav_decodes(conn, settings):
    name = M.sounds.names()[0]
    rid, status = run_chain(conn, settings, [{"id": "s", "node": "sfx", "params": {"name": name, "seed": 2}}],
                            fake_deps())
    assert status == "done"
    sha = out(conn, rid, "s", "file")
    with wave.open(str(media.path_for(settings.media_dir, sha)), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getnframes() > 0
    rid2, _ = run_chain(conn, settings, [{"id": "s", "node": "sfx", "params": {"name": name, "seed": 2}}],
                        fake_deps())
    assert out(conn, rid2, "s", "file") == sha   # same seed, same bytes


def test_sfx_rejects_unknown_name():
    with pytest.raises(ValidationError):
        M.SfxParams(name="kazoo_of_doom")


@pytest.mark.parametrize("style", ["hype", "glitch", "slop"])
def test_montage_every_style_renders_with_sfx(conn, settings, tmp_path, style):
    imgs = _slides(conn, settings, tmp_path)
    rid, st = run_chain(conn, settings, [{"id": "m", "node": "montage", "in": {"images": imgs},
                                          "params": {"style": style, "length_s": 4, "seed": 7, "fps": 24,
                                                     "size": [180, 320], "sfx": True}}], fake_deps())
    assert st == "done"
    pr = probe(settings, out(conn, rid, "m", "video"))
    assert (pr.width, pr.height) == (180, 320) and abs(pr.duration_ms / 1000 - 4.0) < 0.2
