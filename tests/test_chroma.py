"""design: chroma key spec (docs/specs/2026-11-01-chroma-key-design.md). Synthetic clips only: no model, no network."""
import json
import subprocess

import cv2
import numpy as np
import pytest

from wonhee.edit import chroma


def wall(h=360, w=202, bgr=(235, 235, 235), noise=0.0, seed=0):
    im = np.empty((h, w, 3), np.uint8)
    im[:] = bgr
    if noise:
        n = np.random.default_rng(seed).normal(0, noise, im.shape)
        im = np.clip(im + n, 0, 255).astype(np.uint8)
    return im


def with_block(im, x0, x1, y0, y1, bgr):
    out = im.copy()
    out[y0:y1, x0:x1] = bgr
    return out


def person_mask(shape, x0, x1, y0, y1):
    m = np.zeros(shape[:2], np.float32)
    m[y0:y1, x0:x1] = 1
    return m


@pytest.mark.parametrize("bgr,kind", [((60, 200, 60), "green"), ((200, 90, 40), "blue"), ((236, 236, 236), "white"),
                                      ((60, 90, 200), "other")])
def test_detect_kinds(bgr, kind):
    bd = chroma.detect_backdrop([with_block(wall(bgr=bgr), 70, 130, 120, 360, (30, 30, 150))])
    assert bd.kind == kind
    assert bd.ring_coverage > 0.9


def test_detect_never_samples_the_bottom():
    im = wall(bgr=(236, 236, 236))
    im[int(360 * 0.7):, :] = (20, 20, 20)  # a dark floor along the whole bottom
    assert chroma.detect_backdrop([im]).kind == "white"


def test_hint_forces_a_colour():
    bd = chroma.detect_backdrop([wall(bgr=(236, 236, 236))], hint="#00ff00")
    assert bd.kind == "green" and tuple(bd.bgr) == (0, 255, 0)


def test_uneven_wall_is_not_ok():
    im = wall(bgr=(236, 236, 236))
    im[:, :100] = (150, 160, 170)
    bd = chroma.detect_backdrop([im])
    assert not bd.ok and bd.spread > 14


def test_green_key_removes_wall_keeps_subject():
    im = with_block(wall(bgr=(60, 200, 60)), 70, 130, 120, 360, (30, 30, 150))
    bd = chroma.detect_backdrop([im])
    a = chroma.matte(im, bd, 0.5, 0.5)
    assert a[40, 20] < 0.05          # wall
    assert a[250, 100] > 0.95        # subject


def test_white_wall_white_shirt_survives_with_connected_removal():
    # a white-shirted subject (not touching the edges the ring samples) on a white wall
    im = with_block(wall(bgr=(236, 236, 236), noise=2), 70, 130, 120, 330, (240, 240, 240))
    im[150:200, 85:115] = (60, 60, 160)  # face-ish
    bd = chroma.detect_backdrop([im])
    a = chroma.matte(im, bd, 0.5, 0.5)
    assert a[170, 100] > 0.95                      # face kept
    # the wall is gone, but the shirt colour == wall colour: without the guard the shirt IS keyed (documented)
    assert a[40, 20] < 0.05


def test_person_guard_protects_the_white_shirt():
    im = with_block(wall(bgr=(236, 236, 236), noise=2), 70, 130, 120, 360, (240, 240, 240))
    bd = chroma.detect_backdrop([im])
    g = person_mask(im.shape, 70, 130, 120, 360)
    a = chroma.matte(im, bd, 0.5, 0.5, guard=g)
    assert a[300, 100] > 0.95                      # shirt kept by the guard core
    assert a[40, 20] < 0.05                        # wall still gone


def test_guard_trims_specks_outside_the_person():
    im = with_block(wall(bgr=(236, 236, 236)), 70, 130, 120, 360, (40, 40, 40))
    im[20:30, 20:30] = (40, 40, 40)               # a dark speck on the wall, far from the person
    bd = chroma.detect_backdrop([im])
    g = person_mask(im.shape, 70, 130, 120, 360)
    assert chroma.matte(im, bd, 0.5, 0.5)[25, 25] > 0.9
    assert chroma.matte(im, bd, 0.5, 0.5, guard=g)[25, 25] < 0.05


def test_enclosed_background_is_removed():
    im = wall(bgr=(60, 200, 60))
    im[100:360, 60:140] = (30, 30, 150)           # subject
    im[160:260, 90:110] = (60, 200, 60)           # a gap inside it (between arm and body)
    bd = chroma.detect_backdrop([im])
    assert chroma.matte(im, bd, 0.5, 0.5)[200, 100] < 0.05


def test_small_enclosed_patch_is_kept():
    im = wall(bgr=(60, 200, 60))
    im[100:360, 60:140] = (30, 30, 150)
    im[200:204, 100:104] = (60, 200, 60)          # 16 px: a button, not a gap
    bd = chroma.detect_backdrop([im])
    assert chroma.matte(im, bd, 0.5, 0.5)[202, 102] > 0.9


def test_edge_spill_is_cleaned():
    row = np.zeros((1, 3, 3), np.uint8)
    row[:] = (100, 180, 100)                      # a green-tinted edge pixel
    out = chroma.despill(row, np.full((1, 3), 0.5, np.float32), "green", 1.0)
    assert out[0, 0, 1] <= max(out[0, 0, 0], out[0, 0, 2])


def _clip(path, bgr, white_block=False, frames=12, audio=True):
    w, h = 202, 360
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "12",
           "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(frames):
        x = 60 + i * 3
        p.stdin.write(with_block(wall(h, w, bgr), x, x + 60, 120, 360, (30, 30, 150)).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


def test_key_video_end_to_end(tmp_path):
    src = _clip(tmp_path / "in.mp4", (60, 200, 60))
    out = tmp_path / "out.mp4"
    rep = chroma.key_video(src, out, fill="#0000ff", guard="none")
    assert rep["backdrop"]["kind"] == "green"
    assert rep["frames"] == 12
    assert rep["flags"] == []
    cap = cv2.VideoCapture(str(out))
    ok, fr = cap.read()
    assert ok and fr.shape[:2] == (360, 202)
    assert fr[40, 20, 0] > 200 and fr[40, 20, 1] < 60       # wall replaced by blue fill (BGR)
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
                                      capture_output=True, text=True).stdout)
    assert {s["codec_type"] for s in probe["streams"]} == {"video", "audio"}


def test_key_video_uses_a_background_image(tmp_path):
    src = _clip(tmp_path / "in.mp4", (60, 200, 60), audio=False)
    bg = tmp_path / "bg.png"
    cv2.imwrite(str(bg), np.full((100, 100, 3), (0, 128, 255), np.uint8))
    out = tmp_path / "out.mp4"
    chroma.key_video(src, out, background=bg, guard="none")
    ok, fr = cv2.VideoCapture(str(out)).read()
    assert abs(int(fr[40, 20, 2]) - 255) < 25                # orange background (red channel high)


def test_flags_and_nulls(tmp_path):
    src = _clip(tmp_path / "in.mp4", (236, 236, 236))
    rep = chroma.key_video(src, tmp_path / "o.mp4", fill="matte", guard="none")
    assert rep["backdrop"]["kind"] == "white"
    assert rep["guard_used"] is False
    assert isinstance(rep["mean_subject_fraction"], float)
    # an unreadable clip reports unknowns as None, never 0
    empty = chroma.empty_report()
    assert empty["backdrop"]["spread"] is None and empty["mean_subject_fraction"] is None


def test_uneven_wall_flag(tmp_path):
    w, h = 202, 360
    p = tmp_path / "u.mp4"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "12",
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    im = wall(h, w, (236, 236, 236))
    im[:, :100] = (120, 140, 170)
    for _ in range(6):
        pr.stdin.write(im.tobytes())
    pr.stdin.close(); pr.wait()
    rep = chroma.key_video(p, tmp_path / "o.mp4", fill="matte", guard="none")
    assert "BACKDROP_UNEVEN" in rep["flags"]


def test_node_end_to_end_with_stored_report(conn, settings, tmp_path):
    """design: chroma key spec › Output and honesty: the node stores a video and a JSON report."""
    from wonhee.db import repo
    from wonhee.nodes import registry
    from wonhee.nodes.chroma import ChromaParams, chroma_fn
    assert "chroma_key" in registry.all_nodes()
    src = _clip(tmp_path / "in.mp4", (60, 200, 60))
    sha = repo.save_file(conn, settings.media_dir, src, "video", "video/mp4")

    class Ctx:
        pass
    ctx = Ctx(); ctx.conn, ctx.settings, ctx.flagged = conn, settings, []
    ctx.flag = lambda code, msg="": ctx.flagged.append(code)
    out = chroma_fn(ctx, {"video": sha}, ChromaParams(guard="none", fill="#202020"))
    from wonhee import media
    rep = json.loads(media.path_for(settings.media_dir, out["report"]).read_text())
    assert rep["backdrop"]["kind"] == "green" and out["video"]


def test_params_reject_unknown_values():
    from pydantic import ValidationError
    from wonhee.nodes.chroma import ChromaParams
    for bad in ({"backdrop": "red"}, {"guard": "yes"}, {"fill": "pink"}, {"nope": 1}):
        with pytest.raises(ValidationError):
            ChromaParams(**bad)


def test_seat_set_is_deterministic_and_sized():
    """design: chroma key spec › Cabin fill (window-seat set for the seated 3/4 shot)."""
    from wonhee.edit import cabin
    a, b = cabin.render_seat(270, 480), cabin.render_seat(270, 480)
    assert a.shape == (480, 270, 3) and np.array_equal(a, b)
