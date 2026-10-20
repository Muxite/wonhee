"""Layouts, from docs/specs/2026-10-20-meme-and-popart-formats-design.md (Generalized design + Acceptance)."""
import io
import json
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain, store_video
from wonhee import media
from wonhee.db import repo
from wonhee.edit import compose as C
from wonhee.edit import text as T
from wonhee.nodes.images import Images, Slide, load_images, save_images
from wonhee.nodes.subtitles import Line, Subtitles, save_subs


def L(**kw):
    return C.Layout.model_validate(kw)


def photo(w=400, h=500, subject=(0.3, 0.3, 0.7, 0.7)):
    im = Image.new("RGB", (w, h), (40, 90, 160))
    ImageDraw.Draw(im).rectangle((w * subject[0], h * subject[1], w * subject[2], h * subject[3]), fill=(250, 10, 10))
    return im


def mask_of(subject=(0.3, 0.3, 0.7, 0.7)):
    def fn(im):
        w, h = im.size
        m = np.zeros((h, w), np.float32)
        m[int(h * subject[1]):int(h * subject[3]), int(w * subject[0]):int(w * subject[2])] = 1
        return m
    return fn


def png_bytes(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


# ---- the sticker text preset ------------------------------------------------------------------------------------

@pytest.mark.parametrize("slant", [-12, -6, 0, 6, 12, 30])
def test_sticker_text_stays_in_its_box_at_every_slant(slant):
    spec = T.TextSpec("SO SORRY, UNNIE!", "sticker", slant=slant, colours=("#ffe600", "#5b21d6", "#ff3fa4"))
    box = (360, 140)
    im = T.render_block(spec, box, 1350)
    assert im.size == box and im.getbbox() is not None
    l, t, r, b = im.getbbox()
    assert l >= 0 and t >= 0 and r <= box[0] and b <= box[1]


def test_sticker_text_shows_two_strokes_and_the_fill():
    spec = T.TextSpec("HI", "sticker", colours=("#ffe600", "#5b21d6", "#ff3fa4"), drop=0)
    im = T.render_block(spec, (400, 200), 1350)
    mid = np.asarray(im)[im.height // 2]
    seen = {tuple(px[:3]) for px in mid if px[3] == 255}
    assert (255, 230, 0) in seen and (0x5b, 0x21, 0xd6) in seen and (255, 63, 164) in seen
    # reading a row left to right: outer stroke, inner stroke, then fill
    cols = [tuple(px[:3]) for px in mid if px[3] == 255]
    assert cols.index((255, 63, 164)) < cols.index((0x5b, 0x21, 0xd6)) < cols.index((255, 230, 0))


# ---- shapes, outlines, cutout, patterns --------------------------------------------------------------------------

def test_heart_keeps_the_subject_box_inside_and_outlines_keep_their_order():
    lay = L(canvas={"w": 400, "h": 500, "bg": "#ffffff"}, layers=[
        {"type": "photo", "src": 0, "shape": "heart", "focus": "subject", "box": {"x": 0, "y": 0, "w": 1, "h": 1},
         "outlines": [{"width_frac": 0.03, "colour": "#ff0000"}, {"width_frac": 0.03, "colour": "#00ff00"},
                      {"width_frac": 0.03, "colour": "#0000ff"}]}])
    im = C.render(lay, [photo(subject=(0.65, 0.05, 0.97, 0.35))], mask_fn=mask_of((0.65, 0.05, 0.97, 0.35)))
    a = np.asarray(im.convert("RGB")).astype(int)
    # a subject in the photo's corner: the crop zooms out until its box fits the heart, so none of it is clipped
    widths = 3 * round(0.03 * 400)
    shape, _ = C.shape_mask("heart", 400 - 2 * widths, 500 - 2 * widths)
    inside = np.zeros((500, 400), bool)
    inside[widths:500 - widths, widths:400 - widths] = np.asarray(shape) > 128
    face = (a == np.array([250, 10, 10])).all(axis=2)  # the photo's subject colour (outline red is pure #ff0000)
    assert face.sum() > 500 and not (face & ~inside).any()
    # scanning in from the left edge at the row of the heart's widest part: outermost outline first
    row = [tuple(p) for p in a[int(500 * 0.35)]]
    firsts = [row.index(c) for c in [(0, 0, 255), (0, 255, 0), (255, 0, 0)]]
    assert firsts == sorted(firsts)


def test_pattern_is_seeded_and_a_cutout_keeps_its_alpha():
    lay = L(canvas={"w": 300, "h": 300}, layers=[
        {"type": "pattern", "kind": "halftone", "colours": ["#ffe600", "#ff3fa4"]},
        {"type": "pattern", "kind": "stars", "colours": ["#00000000", "#1b1030"]},
        {"type": "photo", "src": 0, "cutout": True}])
    a = C.render(lay, [photo(300, 300)], 5, mask_fn=mask_of())
    b = C.render(lay, [photo(300, 300)], 5, mask_fn=mask_of())
    c = C.render(lay, [photo(300, 300)], 6, mask_fn=mask_of())
    assert png_bytes(a) == png_bytes(b) and png_bytes(a) != png_bytes(c)
    # inside the mask the photo's red; outside, the pattern shows through (never the photo's blue)
    arr = np.asarray(a.convert("RGB"))
    assert tuple(arr[150, 150]) == (250, 10, 10)
    assert tuple(arr[5, 5]) != (40, 90, 160)
    alone = C.render(L(canvas={"w": 300, "h": 300, "bg": "#00000000"}, layers=[{"type": "photo", "src": 0, "cutout": True}]),
                     [photo(300, 300)], mask_fn=mask_of())
    assert alone.getchannel("A").getextrema() == (0, 255)  # transparent canvas outside the subject


@pytest.mark.parametrize("kind", ["flat", "halftone", "stars", "glyph", "stripes"])
def test_every_pattern_kind_fills_its_box(kind):
    im = C.render(L(canvas={"w": 200, "h": 200}, layers=[{"type": "pattern", "kind": kind, "colours": ["#ffffff", "#000000"]}]))
    assert im.getchannel("A").getextrema() == (255, 255)
    assert (kind == "flat") or len({tuple(p) for p in np.asarray(im.convert("RGB")).reshape(-1, 3)}) >= 2


@pytest.mark.parametrize("kind", C.STICKER_KINDS)
def test_every_sticker_kind_draws_inside_its_box(kind):
    box = {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.3}
    lay = L(canvas={"w": 600, "h": 600, "bg": "#ffffff"}, layers=[
        {"type": "sticker", "kind": kind, "content": "ship it", "items": ["one", "two"] if kind == "checklist" else None,
         "box": box, "tail": {"x": 0.3, "y": 1.0} if kind == "bubble" else None}])
    im = C.render(lay)
    a = np.asarray(im.convert("RGB")).astype(int)
    drawn = np.argwhere((a != 255).any(axis=2))
    assert len(drawn) > 200
    assert drawn[:, 1].min() >= 150 - 1 and drawn[:, 1].max() <= 450 and drawn[:, 0].min() >= 150 - 1 and drawn[:, 0].max() <= 330


def test_shapes_cover_the_box():
    for kind in ("rect", "rounded", "circle", "heart", "star", "burst", "bubble"):
        m, inner = C.shape_mask(kind, 200, 160)
        assert m.size == (200, 160) and np.asarray(m).max() == 255 and 0 < inner[0] < inner[2] <= 200


# ---- templates, validation, safe zones ---------------------------------------------------------------------------

def test_every_template_renders_and_lists_its_slots():
    names = {t["name"] for t in C.list_templates()}
    assert names == {"meme_frame", "sticker_frame", "brag_collage", "popart_cutout", "slop_card"}
    for t in C.list_templates():
        assert t["slots"] and all("default" in s and s.get("doc") for s in t["slots"].values())
        im = C.render(C.load_template(t["name"], {"cutout": False} if "cutout" in t["slots"] else {}), [photo()], 1,
                      mask_fn=mask_of())
        assert im.size[0] == 1080 and im.getchannel("A").getextrema()[0] == 255, t["name"]


def test_layout_and_template_reject_unknowns():
    with pytest.raises(ValueError):
        L(layers=[{"type": "text", "bogus": 1}])
    with pytest.raises(ValueError):
        L(palette="no-such-palette")
    with pytest.raises(ValueError):
        L(layers=[{"type": "sticker", "kind": "nope"}])
    with pytest.raises(ValueError, match="no slot"):
        C.load_template("sticker_frame", {"nope": 1})
    with pytest.raises(ValueError, match="unknown template"):
        C.load_template("nope")


def extent_x(box, rot):
    cx, cy, hw, hh = C._extent(box, rot)
    return cx - hw, cx + hw, cy - hh, cy + hh


def test_safe_zone_moves_text_on_tall_canvases_only():
    """Spec amendment: the same rule as edit/text.py (12%..76% of the height; the centred 72% where the button strip is)."""
    W, H = 1080, 1920
    box, moved = C.safe_box((100, 1600, 900, 1800), W, H)  # caption zone
    assert moved and box[3] <= H * 0.76 + 1
    box, moved = C.safe_box((500, 700, 1050, 800), W, H)  # button strip, right edge
    assert moved and W * 0.14 - 1 <= box[0] and box[2] <= W * 0.86 + 1
    wide, moved = C.safe_box((0, 700, 1080, 800), W, H)  # full width in the strip: narrowed to the centred 72%
    assert moved and wide[0] >= W * 0.14 - 1 and wide[2] <= W * 0.86 + 1 and abs((wide[0] + wide[2]) / 2 - W / 2) < 2
    assert C.safe_box((100, 100, 900, 160), W, H)[1] is True  # above 12%
    assert C.safe_box((100, 1600, 900, 1800), 1080, 1350) == ((100, 1600, 900, 1800), False)  # feed shape: untouched
    box, moved = C.safe_box((0, 300, 1080, 500), W, H)  # top title above the strip keeps full width
    assert not moved


@pytest.mark.parametrize("rot", [-15, 12, 30, 90])
def test_safe_zone_checks_the_rotated_extent(rot):
    W, H = 1080, 1920
    for box in [(300, 640, 800, 760), (0, 800, 1080, 900), (700, 1000, 1080, 1100), (100, 1500, 900, 1700)]:
        new, _ = C.safe_box(box, W, H, rot)
        x0, x1, y0, y1 = extent_x(new, rot)
        assert y0 >= H * 0.12 - 2 and y1 <= H * 0.76 + 2, (box, rot)
        if y1 > H * 0.35 and y0 < H * 0.88:
            assert x0 >= W * 0.14 - 2 and x1 <= W * 0.86 + 2, (box, rot)


def test_layout_text_in_the_caption_zone_is_moved_and_noted():
    lay = L(canvas={"w": 540, "h": 960, "bg": "#00000000"}, layers=[{"type": "text", "content": "hello", "box": {"x": 0.1, "y": 0.85, "w": 0.8, "h": 0.1}}])
    im, notes = C.render_ex(lay)
    assert len(notes.adjusted) == 1
    assert im.getchannel("A").getbbox()[3] <= 960 * 0.77


# ---- the nodes ---------------------------------------------------------------------------------------------------

def store_images(conn, settings, tmp_path, *ims):
    items = []
    for i, im in enumerate(ims):
        p = tmp_path / f"in{i}.png"
        im.save(p)
        items.append(Slide(sha=repo.save_file(conn, settings.media_dir, p, "image", "image/png")))
    return save_images(conn, settings.media_dir, Images(items=items))


def test_compose_node_template_cutout_deterministic_and_flags(conn, settings, tmp_path, monkeypatch):
    from wonhee.edit import mask as mk
    monkeypatch.setattr(mk, "subject", lambda im, models_dir, kind="person", feather=0.004: mask_of()(Image.fromarray(im[..., ::-1])))
    ref = store_images(conn, settings, tmp_path, photo())
    nodes = [{"id": "c", "node": "compose", "in": {"images": ref},
              "params": {"template": "popart_cutout", "slots": {"palette": "mint"}, "seed": 3}}]
    rid, status = run_chain(conn, settings, nodes, fake_deps())
    assert status == "done"
    sha = out(conn, rid, "c", "image")
    rid2, _ = run_chain(conn, settings, [{**nodes[0], "params": {**nodes[0]["params"], "slots": {"palette": "mint"}}}], fake_deps())
    assert out(conn, rid2, "c", "image") == sha  # same layout, inputs, seed: identical bytes
    im = Image.open(media.path_for(settings.media_dir, sha))
    assert im.size == (1080, 1350)
    tall = [{"id": "t", "node": "compose", "params": {"layout": {
        "canvas": {"w": 540, "h": 960}, "layers": [{"type": "text", "content": "x", "box": {"x": 0, "y": 0.9, "w": 1, "h": 0.08}}]}}}]
    rid3, status = run_chain(conn, settings, tall, fake_deps())
    assert status == "done" and "LAYOUT_ADJUSTED" in flags(conn, rid3, "t")


def test_compose_params_need_exactly_one_of_layout_or_template():
    from wonhee.nodes.compose import ComposeParams
    with pytest.raises(ValueError):
        ComposeParams()
    with pytest.raises(ValueError):
        ComposeParams(template="meme_frame", layout={})
    from wonhee.nodes import registry
    schema = json.dumps(registry.get("compose").params.model_json_schema())
    for name in ("meme_frame", "sticker_frame", "brag_collage", "popart_cutout", "slop_card", "headline", "title"):
        assert name in schema


def test_grid_plan_nine_tiles_row_palettes_follow_the_plan(conn, settings, tmp_path):
    plan = C.plan_grid(9, ["popart-yellow", "popart-pink", "lilac"])
    assert [t.palette for t in plan] == ["popart-yellow"] * 3 + ["popart-pink"] * 3 + ["lilac"] * 3
    assert all(plan[i].pattern != plan[i + 1].pattern for i in (0, 1, 3, 4, 6, 7))  # left neighbours
    assert all(plan[i].pattern != plan[i + 3].pattern for i in range(6))  # upper neighbours
    with pytest.raises(ValueError):
        C.plan_grid(4, ["lilac"])
    ref = store_images(conn, settings, tmp_path, *[photo(subject=(0.2, 0.2, 0.5, 0.5)) for _ in range(9)])
    rid, status = run_chain(conn, settings, [{"id": "g", "node": "grid_plan", "in": {"images": ref},
                                              "params": {"slots": {"pattern": "flat"}}}], fake_deps())
    assert status == "done" and "GRID_PLANNED" in flags(conn, rid, "g")
    tiles = load_images(settings.media_dir, out(conn, rid, "g", "images")).items
    assert len(tiles) == 9
    for t, pl in zip(tiles, plan):
        px = Image.open(media.path_for(settings.media_dir, t.sha)).convert("RGB").getpixel((2, 2))
        want = C.colour(C.PALETTES[pl.palette][0], [])[:3]
        assert px == want


def test_grid_plan_split_cuts_one_photo_across_tiles(conn, settings, tmp_path):
    ref = store_images(conn, settings, tmp_path, photo(900, 900))
    rid, status = run_chain(conn, settings, [{"id": "g", "node": "grid_plan", "in": {"images": ref},
                                              "params": {"split": True, "tiles": 6, "template": "brag_collage",
                                                         "slots": {"palette": "mint"}}}], fake_deps())
    assert status == "done"
    assert len(load_images(settings.media_dir, out(conn, rid, "g", "images")).items) == 6


@pytest.mark.parametrize("template", ["sticker_frame", "popart_cutout", "meme_frame"])
def test_grid_plan_refuses_split_unless_the_photo_is_a_full_bleed_rect(conn, settings, tmp_path, template):
    ref = store_images(conn, settings, tmp_path, photo(900, 900))
    rid, status = run_chain(conn, settings, [{"id": "g", "node": "grid_plan", "in": {"images": ref},
                                              "params": {"split": True, "tiles": 3, "template": template}}], fake_deps())
    assert status != "done"


# ---- cache salt, copy guard, flags -------------------------------------------------------------------------------

def test_editing_a_template_is_a_cache_miss(conn, settings, tmp_path, monkeypatch):
    """spec amendment: the templates' digest is in the salt of compose, frame_video and grid_plan."""
    import shutil
    tdir = tmp_path / "templates"
    shutil.copytree(C.TEMPLATES, tdir)
    monkeypatch.setattr(C, "TEMPLATES", tdir)
    ref = store_images(conn, settings, tmp_path, photo())
    node = [{"id": "c", "node": "compose", "in": {"images": ref}, "params": {"template": "slop_card"}}]
    run_chain(conn, settings, node, fake_deps())
    rid, _ = run_chain(conn, settings, node, fake_deps())
    assert "CACHED" in flags(conn, rid, "c")
    f = tdir / "slop_card.json"
    f.write_text(f.read_text().replace('"default": "it just works"', '"default": "it works"'))
    rid, status = run_chain(conn, settings, node, fake_deps())
    assert status == "done" and "CACHED" not in flags(conn, rid, "c")
    from wonhee.nodes import registry
    for name in ("compose", "frame_video", "grid_plan"):
        assert registry.get(name).version >= 2 and registry.get(name).salt(settings)["templates"] == C.templates_digest()


def test_shipped_defaults_do_not_reuse_reference_wording():
    from pathlib import Path
    banned = ["sorry", "unnie", "on prem", "on-prem", "onprem", "easygoing", "legendary"]
    texts = [p.read_text().lower() for p in C.TEMPLATES.glob("*.json")]
    texts.append((Path(__file__).resolve().parents[1] / "docs" / "chains" / "sticker_frame.json").read_text().lower())
    for t in texts:
        assert not [b for b in banned if b in t], banned


def test_compose_drops_copied_lines_and_flags_idea_copy(conn, settings, tmp_path):
    refs = repo.save_bytes(conn, settings.media_dir, b'  overlay: "so sorry unnie today" (c1)\n', "text/plain")
    layout = {"canvas": {"w": 400, "h": 400}, "layers": [
        {"type": "sticker", "kind": "label", "content": "SO SORRY UNNIE", "box": {"x": .1, "y": .1, "w": .5, "h": .2}},
        {"type": "text", "content": "my own line", "box": {"x": .1, "y": .5, "w": .8, "h": .2}}]}
    rid, status = run_chain(conn, settings, [{"id": "c", "node": "compose", "in": {"refs": refs}, "params": {"layout": layout}}],
                            fake_deps())
    assert status == "done" and flags(conn, rid, "c").count("IDEA_COPY") == 1
    plain = Image.open(media.path_for(settings.media_dir, out(conn, rid, "c", "image"))).convert("RGB")
    rid2, _ = run_chain(conn, settings, [{"id": "c", "node": "compose", "params": {"layout": {
        **layout, "layers": layout["layers"][1:]}}}], fake_deps())
    assert Image.open(media.path_for(settings.media_dir, out(conn, rid2, "c", "image"))).convert("RGB").tobytes() == plain.tobytes()


def test_focus_subject_without_a_subject_flags_and_falls_back_to_centre(conn, settings, tmp_path, monkeypatch):
    from wonhee.edit import mask as mk
    monkeypatch.setattr(mk, "subject", lambda im, models_dir, kind="person", feather=0.004: np.zeros(im.shape[:2], np.float32))
    ref = store_images(conn, settings, tmp_path, photo())
    layout = {"canvas": {"w": 300, "h": 300}, "layers": [{"type": "photo", "src": 0, "shape": "heart", "focus": "subject"}]}
    rid, status = run_chain(conn, settings, [{"id": "c", "node": "compose", "in": {"images": ref}, "params": {"layout": layout}}],
                            fake_deps())
    assert status == "done" and "FOCUS_NO_SUBJECT" in flags(conn, rid, "c")


def test_text_layer_colours_do_not_recolour_emoji():
    if not T.EMOJI.exists():
        pytest.skip("Noto Color Emoji not installed")
    lay = L(canvas={"w": 600, "h": 300, "bg": "#00000000"}, layers=[
        {"type": "text", "content": "go ☁️", "size_frac": 0.3, "colour": "#ff0000", "stroke": ["#000000"], "preset": "outline",
         "box": {"x": 0, "y": 0, "w": 1, "h": 1}}])
    a = np.asarray(C.render(lay)).astype(int)
    opaque = a[..., 3] > 250
    assert (opaque & (a[..., 0] > 240) & (a[..., 1] < 20) & (a[..., 2] < 20)).sum() > 300  # the fill is red
    assert (opaque & (a[..., :3] > 235).all(axis=2)).sum() > 100  # the cloud's whites survive


@pytest.mark.parametrize("kind", ["bubble", "burst", "badge", "label", "note"])
@pytest.mark.parametrize("rot", [-15, 15])
def test_slanted_rotated_sticker_stays_in_its_rotated_box(kind, rot):
    W = H = 800
    box = (0.25, 0.3, 0.5, 0.25)
    lay = L(canvas={"w": W, "h": H, "bg": "#ffffff"}, layers=[
        {"type": "sticker", "kind": kind, "content": "REALLY LONG STICKER TEXT HERE", "slant": 12, "rotate": rot,
         "box": dict(zip("xywh", box))}])
    a = np.asarray(C.render(lay).convert("RGB")).astype(int)
    ys, xs = np.where((a != 255).any(axis=2))
    px = (round(box[0] * W), round(box[1] * H), round((box[0] + box[2]) * W), round((box[1] + box[3]) * H))
    x0, x1, y0, y1 = extent_x(px, rot)
    assert len(xs) > 200 and xs.min() >= x0 - 3 and xs.max() <= x1 + 3 and ys.min() >= y0 - 3 and ys.max() <= y1 + 3


# ---- frame_video -------------------------------------------------------------------------------------------------

def test_caption_groups_are_at_most_three_words_inside_their_lines():
    groups = C.caption_groups([(1000, 3000, "one two three four five"), (3500, 4000, "six")], 3)
    assert [g[2] for g in groups] == ["one two three", "four five", "six"]
    assert all(len(g[2].split()) <= 3 for g in groups)
    assert groups[0][0] == 1.0 and groups[1][0] == pytest.approx(1 + 3 * 2 / 5)
    for (a0, a1, _), (b0, b1, _) in zip(groups, groups[1:]):
        assert a1 <= b0 + 1e-9  # never two groups on screen at once
    assert C.caption_groups([(0, 1000, "a b c d")], 1)[3][2] == "d"
    assert C.caption_groups([(0, 0, "bad")]) == []


def count_ffmpeg(monkeypatch):
    calls = []
    real = subprocess.run

    def spy(argv, *a, **k):
        if argv and argv[0] == "ffmpeg":
            calls.append(argv)
        return real(argv, *a, **k)
    monkeypatch.setattr(subprocess, "run", spy)
    return calls


def test_frame_video_one_encode_keeps_aspect_title_in_top_band_and_captions(conn, settings, tiny_video, monkeypatch, tmp_path):
    sha = store_video(conn, settings, tiny_video)
    subs = save_subs(conn, settings.media_dir, Subtitles(lines=[
        Line(id=0, start_ms=100, end_ms=1500, text="when the pump hits", source="speech")]))
    calls = count_ffmpeg(monkeypatch)
    rid, status = run_chain(conn, settings, [{"id": "f", "node": "frame_video", "in": {"video": sha, "subtitles": subs},
                                              "params": {"slots": {"title": "THE TITLE GOES HERE"}, "encoder": "x264"}}],
                            fake_deps())
    assert status == "done"
    assert len(calls) == 1  # exactly one encode: overlay, scaled clip and captions share one graph
    assert "WORDS_ESTIMATED" in flags(conn, rid, "f")
    dst = media.path_for(settings.media_dir, out(conn, rid, "f", "video"))
    info = media.probe(dst)
    assert (info.width, info.height) == (1080, 1920) and abs(info.duration_ms - 3000) < 300 and media.has_audio(dst)
    frame = tmp_path / "f.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "0.2", "-i", str(dst), "-frames:v", "1", str(frame)], check=True)
    a = np.asarray(Image.open(frame).convert("RGB")).astype(int)
    red = (a[..., 0] > 180) & (a[..., 1] < 80) & (a[..., 2] < 80)  # the clip is red for its first second
    ys, xs = np.where(red)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert abs(w / h - 320 / 568) < 0.03  # the clip keeps its aspect
    assert 0.33 * 1920 <= ys.min() and ys.max() <= 0.65 * 1920  # inside the window
    white = (a.min(axis=2) > 235)
    top = np.where(white[: int(0.33 * 1920)].any(axis=1))[0]
    assert len(top) and top.min() >= 0.12 * 1920 - 2 and top.max() < 0.33 * 1920  # title lies in the top band
    cap = np.where((np.abs(a - np.array([0xc9, 0xa0, 0xe6])).sum(axis=2) < 30))
    assert len(cap[0]) > 500 and cap[0].min() >= 0.665 * 1920 and cap[0].max() <= 0.76 * 1920  # caption box in the safe zone
    late = tmp_path / "late.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "2.5", "-i", str(dst), "-frames:v", "1", str(late)], check=True)
    b = np.asarray(Image.open(late).convert("RGB")).astype(int)
    assert not (np.abs(b - np.array([0xc9, 0xa0, 0xe6])).sum(axis=2) < 30).any()  # captions end with their line


def test_frame_video_without_subtitles_is_still_one_encode(conn, settings, tiny_video, monkeypatch):
    sha = store_video(conn, settings, tiny_video)
    calls = count_ffmpeg(monkeypatch)
    rid, status = run_chain(conn, settings, [{"id": "f", "node": "frame_video", "in": {"video": sha},
                                              "params": {"encoder": "x264", "captions": "highlight"}}], fake_deps())
    assert status == "done" and len(calls) == 1


def test_caption_overlays_never_share_a_frame(conn, settings, tiny_video, monkeypatch):
    """spec amendment: overlays switch with gte(t,a)*lt(t,b), so at every t at most one group (<= 3 words) is on."""
    import re
    sha = store_video(conn, settings, tiny_video)
    subs = save_subs(conn, settings.media_dir, Subtitles(lines=[
        Line(id=0, start_ms=0, end_ms=1200, text="a b c d e f", source="speech"),
        Line(id=1, start_ms=1200, end_ms=2400, text="g h i", source="speech")]))
    calls = count_ffmpeg(monkeypatch)
    run_chain(conn, settings, [{"id": "f", "node": "frame_video", "in": {"video": sha, "subtitles": subs},
                                "params": {"encoder": "x264"}}], fake_deps())
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    wins = [(float(a), float(b)) for a, b in re.findall(r"enable='gte\(t,([\d.]+)\)\*lt\(t,([\d.]+)\)'", graph)]
    assert len(wins) == 3 and "between(" not in graph
    for t in [i / 1000 for i in range(0, 2500)] + [a for a, _ in wins] + [b for _, b in wins]:
        assert sum(1 for a, b in wins if a <= t < b) <= 1, t
