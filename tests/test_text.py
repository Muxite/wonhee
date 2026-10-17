import subprocess

import pytest
from PIL import Image

from tests.fakes import fake_deps
from tests.test_nodes_subs import out, run_chain, store_video
from wonhee import media
from wonhee.db import repo
from wonhee.edit import text as T
from wonhee.edit.text import TextSpec, render_layer, runs

needs_emoji = pytest.mark.skipif(not T.EMOJI.exists(), reason="Noto Color Emoji not installed")
needs_cjk = pytest.mark.skipif(not T.CJK.exists(), reason="Noto Sans CJK not installed")


def test_runs_split_emoji_cjk_and_keep_joiners():
    assert runs("hi 🎉 there") == [("base", "hi "), ("emoji", "🎉"), ("base", " there")]
    assert runs("❤️") == [("emoji", "❤️")]  # variation selector stays with its emoji
    assert runs("👨‍💻ok") == [("emoji", "👨‍💻"), ("base", "ok")]  # ZWJ sequence stays whole
    assert runs("POV 男生") == [("base", "POV "), ("cjk", "男生")]


def _colourful(layer: Image.Image) -> int:
    """Pixels that are clearly coloured (not white/black/grey outline text)."""
    rgb = layer.convert("RGBA")
    return sum(1 for r, g, b, a in rgb.get_flattened_data() if a > 200 and max(r, g, b) - min(r, g, b) > 80)


@needs_emoji
def test_emoji_render_in_colour():
    plain = render_layer(TextSpec("shipping"), (540, 960))
    emoji = render_layer(TextSpec("shipping 🎉❤️"), (540, 960))
    assert _colourful(plain) == 0 and _colourful(emoji) > 200


@needs_cjk
def test_cjk_renders_glyphs_not_tofu():
    layer = render_layer(TextSpec("男生約會 우주떡집"), (540, 960))
    tofu = render_layer(TextSpec("\U000E0001\U000E0001"), (540, 960))  # unassigned tags draw nothing useful
    assert layer.getbbox() is not None and (tofu.getbbox() is None or layer.getbbox() != tofu.getbbox())


def test_every_preset_and_position_places_text_in_frame():
    for preset in ("outline", "band", "script"):
        for pos in ("top", "upper", "center", "lower", "bottom"):
            box = render_layer(TextSpec("still shipping at 3:17 AM", preset, pos), (540, 960)).getbbox()
            assert box is not None and box[1] >= 0 and box[3] <= 960, (preset, pos)
    top = render_layer(TextSpec("x", "outline", "top"), (540, 960)).getbbox()
    bottom = render_layer(TextSpec("x", "outline", "bottom"), (540, 960)).getbbox()
    assert top[1] < 960 * 0.2 and bottom[3] <= 960 * 0.81


def test_long_text_wraps_inside_width():
    layer = render_layer(TextSpec("me and bro shipping our 4th AI wrapper after saying we would never build one"),
                         (540, 960))
    l, t, r, b = layer.getbbox()
    assert r - l <= 540 * 0.86 and b - t > 960 * 0.05  # several lines


def test_band_fills_full_width():
    l, t, r, b = render_layer(TextSpec("now in stealth", "band", "top"), (540, 960)).getbbox()
    assert l == 0 and r == 540


def test_text_image_node(conn, settings, tiny_jpg):
    sha = repo.save_file(conn, settings.media_dir, tiny_jpg, "image", "image/jpeg")
    rid, status = run_chain(conn, settings, [{"id": "t", "node": "text_image", "in": {"image": sha},
                                              "params": {"text": "Q4 results are in", "look": "digicam"}}],
                            fake_deps())
    assert status == "done"
    new = out(conn, rid, "t", "image")
    assert new != sha and media.probe(media.path_for(settings.media_dir, new)).width == 160


def test_text_video_node_window_keeps_duration_and_audio(conn, settings, tiny_video):
    sha = store_video(conn, settings, tiny_video)
    rid, status = run_chain(conn, settings, [{"id": "t", "node": "text_video", "in": {"video": sha},
                                              "params": {"text": "still shipping 🚀", "preset": "band",
                                                         "start_s": 1, "end_s": 2, "encoder": "x264",
                                                         "look": "moody"}}], fake_deps())
    assert status == "done"
    src, dst = media.path_for(settings.media_dir, sha), media.path_for(settings.media_dir, out(conn, rid, "t", "video"))
    assert abs(media.probe(dst).duration_ms - media.probe(src).duration_ms) < 200
    assert media.has_audio(dst) == media.has_audio(src)
