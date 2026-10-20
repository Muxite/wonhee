import subprocess

import pytest
from PIL import Image, ImageColor

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


def test_tall_frames_keep_text_off_the_buttons_and_caption():
    """design: DESIGN Stage 4 safe zones (phone screenshot 2026-10-20): above 76% of the height, and inside the middle
    72% of the width wherever a block reaches the button strip (35-88%); a top title may use the full width."""
    W, H = 540, 960
    long = "still shipping at 3:17 AM because the model said it was fine and the tests said otherwise"
    for pos in ("upper", "center", "lower", "bottom"):
        x0, y0, x1, y1 = render_layer(TextSpec(long, "outline", pos), (W, H)).getbbox()
        assert y1 <= H * 0.76 + 1
        if y1 > H * 0.35 and y0 < H * 0.88:
            assert x1 <= W * 0.86 + 1 and x0 >= W * 0.14 - 1, (pos, x0, x1)
    tx0, ty0, tx1, ty1 = render_layer(TextSpec("BIG TITLE AT THE TOP OF THE REEL", "outline", "top"), (W, H)).getbbox()
    assert ty1 < H * 0.35 and tx1 > W * 0.86  # the top band is free of buttons, so the title keeps its width
    sq = render_layer(TextSpec(long, "outline", "center"), (W, W)).getbbox()
    assert sq[2] > W * 0.86  # square/4:5 posts have no button strip over the image


# ---- spec 2026-10-20-reel-styles-design.md §4: roles, tracking, line height, boxes --------------------------------

def _ink_width(layer: Image.Image) -> int:
    box = layer.getchannel("A").getbbox()
    return box[2] - box[0] if box else 0


def test_every_role_resolves_to_a_font_that_draws():
    for role in T.ROLES:
        f = T.font_for(role, 40)
        assert f.getlength("Hamburgefonstiv") > 0, role


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        T.font_for("comic", 40)
    with pytest.raises(ValueError):
        TextSpec("x", role="comic")


def test_variable_roles_pick_their_instance():
    # Figtree SemiBold is wider than its default Light; Fraunces is set by axes (SOFT 100, weight 800)
    assert T.font_for("caption", 60).getlength("hello world") > T.ImageFont.truetype(str(T.FONTS / "Figtree-VF.ttf"), 60).getlength("hello world")
    assert T.font_for("soft_serif", 60).getlength("hello") != T.ImageFont.truetype(str(T.FONTS / "Fraunces-VF.ttf"), 60).getlength("hello")


def test_new_fields_are_validated():
    for bad in (dict(tracking_em=0.3), dict(tracking_em=-0.2), dict(line_height=0.5), dict(line_height=2.0), dict(box="blob")):
        with pytest.raises(ValueError):
            TextSpec("x", **bad)
    TextSpec("x", tracking_em=-0.1, line_height=0.8)
    TextSpec("x", tracking_em=0.2, line_height=1.6, box="paper")


def test_a_role_changes_the_glyphs_and_defaults_do_not():
    plain = render_layer(TextSpec("headline test"), (540, 960))
    assert render_layer(TextSpec("headline test", role=None, tracking_em=0.0, line_height=None, box="none"), (540, 960)).tobytes() == plain.tobytes()
    assert render_layer(TextSpec("headline test", role="headline"), (540, 960)).tobytes() != plain.tobytes()


def test_tracking_widens_monotonically():
    widths = [_ink_width(render_layer(TextSpec("tracking", role="caption", tracking_em=t), (1080, 1920)))
              for t in (-0.1, -0.05, 0.0, 0.1, 0.2)]
    assert widths == sorted(widths) and len(set(widths)) == 5


def test_tracking_works_with_every_role():
    for role in T.ROLES:
        assert _ink_width(render_layer(TextSpec("Ab", role=role, tracking_em=0.1), (540, 960))) > 0, role


@needs_emoji
@needs_cjk
def test_emoji_and_cjk_fallback_work_with_any_role_and_tracking():
    for role in ("headline", "pixel", "soft_serif"):
        lay = render_layer(TextSpec("hi 🎉 男生 우주", role=role, tracking_em=0.05), (540, 960))
        assert _colourful(lay) > 200, role  # the emoji kept its colour
        assert _ink_width(lay) > _ink_width(render_layer(TextSpec("hi", role=role, tracking_em=0.05), (540, 960)))


def test_line_height_sets_the_pitch_of_wrapped_lines():
    def block_h(lh):
        lay = render_layer(TextSpec("one\ntwo\nthree", position="center", role="editorial", line_height=lh), (1080, 1920))
        box = lay.getchannel("A").getbbox()
        return box[3] - box[1]
    assert block_h(0.9) < block_h(1.1) < block_h(1.5)


@pytest.mark.parametrize("kind,colour", [("box", "#ff0000"), ("pill", "#00ff00"), ("paper", "#0000ff")])
def test_box_kinds_draw_behind_the_text(kind, colour):
    spec = TextSpec("Boxed text", box=kind, box_colour=colour, colours=("white", "white"), position="center")
    lay = render_layer(spec, (1080, 1920))
    bare = render_layer(replace_box_none(spec), (1080, 1920))
    assert _ink_width(lay) > _ink_width(bare)
    c = ImageColor.getrgb(colour)
    painted = [p for p in lay.get_flattened_data() if p[3] == 255 and p[:3] == c]
    assert len(painted) > 1000  # the backing exists
    # behind, not in front: the white glyphs survive on top and sit inside the backing's extent
    white = Image.new("L", lay.size, 0)
    white.putdata([255 if p[3] == 255 and p[:3] == (255, 255, 255) else 0 for p in lay.get_flattened_data()])
    back = Image.new("L", lay.size, 0)
    back.putdata([255 if p[3] == 255 and p[:3] == c else 0 for p in lay.get_flattened_data()])
    wb, bb = white.getbbox(), back.getbbox()
    assert wb and sum(1 for v in white.get_flattened_data() if v) > 200
    assert bb[0] < wb[0] and bb[2] > wb[2] and bb[1] < wb[1] and bb[3] > wb[3]


def replace_box_none(spec):
    from dataclasses import replace
    return replace(spec, box="none")


def test_boxes_hug_each_line_and_default_colour_is_translucent():
    lay = render_layer(TextSpec("a\nlonger second line", box="box", position="center"), (1080, 1920))
    a = lay.getchannel("A")
    x0, y0, x1, y1 = a.getbbox()
    assert 0 < x0 and x1 < 1080  # the box hugs the text, it is not a full-width band
    assert a.crop((x0, y0, x0 + 4, y1)).getextrema()[1] == 0x99  # translucent black


def test_render_block_boxed_text_fits_its_box():
    from wonhee.edit.text import render_block
    im = render_block(TextSpec("a fairly long caption line that must shrink", box="pill", role="caption"), (400, 120), 1000)
    assert im.size == (400, 120) and im.getchannel("A").getbbox() is not None


def test_stroke_em_sets_the_outline_and_box_defaults_to_none():
    def ink(**kw):
        return sum(1 for p in render_layer(TextSpec("Stroke", **kw), (540, 960)).get_flattened_data() if p[3])
    assert ink(stroke_em=0.0) < ink() < ink(stroke_em=0.15)
    assert ink(role="ui", stroke_em=0.0) == ink(role="ui", stroke_em=0.0, box="none")
    with pytest.raises(ValueError):
        TextSpec("x", stroke_em=0.2)


def test_new_weight_roles_resolve_and_differ():
    w = lambda r: T.font_for(r, 60).getlength("Weights")
    assert w("ui") > w("ui_medium") and w("caption_bold") > w("caption")
    assert T.font_for("numeral", 60).getlength("123") > 0
