import json

from PIL import Image, ImageStat

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain
from wonhee import service
from wonhee.edit.carousel import layout, parse_size, span_for
from wonhee.media import path_for
from wonhee.nodes.images import load_images


def _photos(tmp_path):
    d = tmp_path / "dump"
    d.mkdir()
    Image.new("RGB", (800, 1000), (200, 120, 60)).save(d / "img2.jpg")
    Image.new("RGB", (800, 1000), (60, 120, 200)).save(d / "img10.jpg")
    Image.new("RGB", (2400, 1000), (120, 200, 60)).save(d / "img1.jpg")  # a panorama
    (d / "notes.txt").write_text("not an image")
    return d


def test_geometry():
    assert parse_size("4:5") == (1080, 1350) and parse_size("720x900") == (720, 900)
    assert span_for(2400, 1000, (1080, 1350)) == 3 and span_for(800, 1000, (1080, 1350)) == 1
    cols = layout(Image.new("RGB", (2400, 1000)), (1080, 1350), "cover", span=True)
    assert len(cols) == 3 and all(c.size == (1080, 1350) for c in cols)
    assert len(layout(Image.new("RGB", (2400, 1000)), (1080, 1350), "contain", span=False)) == 1


def test_directory_becomes_ordered_manifest(conn, settings, tmp_path):
    d = _photos(tmp_path)
    rid, text = service.node_run(conn, settings, fake_deps(), "carousel", {"images": str(d)},
                                 {"size": "1:1", "span": False})
    assert "3 image(s)" in text
    run_out = out(conn, rid, "carousel", "images")
    slides = load_images(settings.media_dir, run_out).items
    first = Image.open(path_for(settings.media_dir, slides[0].sha))
    assert first.size == (1080, 1080) and ImageStat.Stat(first).mean[1] > 180  # img1 (green) first: natural sort


def test_panorama_spans_slides_and_text_lands_on_last(conn, settings, tmp_path):
    d = _photos(tmp_path)
    j = tmp_path / "set.json"
    j.write_text(json.dumps([{"path": "dump/img1.jpg", "text": "Q4 site visit"}, "dump/img2.jpg"]))
    rid, _ = service.node_run(conn, settings, fake_deps(), "carousel", {"images": str(j)}, {})
    slides = load_images(settings.media_dir, out(conn, rid, "carousel", "images")).items
    assert len(slides) == 4  # 3-wide panorama + 1
    imgs = [Image.open(path_for(settings.media_dir, s.sha)).convert("L") for s in slides[:3]]
    spread = [ImageStat.Stat(i).stddev[0] for i in imgs]
    assert spread[2] > spread[0] + 5  # the caption made the last panorama slide busier


def test_look_applies_to_every_slide(conn, settings, tmp_path):
    d = _photos(tmp_path)
    plain_rid, _ = service.node_run(conn, settings, fake_deps(), "carousel", {"images": str(d)}, {"span": False})
    moody_rid, _ = service.node_run(conn, settings, fake_deps(), "carousel", {"images": str(d)},
                                    {"span": False, "look": "moody"})
    a = load_images(settings.media_dir, out(conn, plain_rid, "carousel", "images")).items
    b = load_images(settings.media_dir, out(conn, moody_rid, "carousel", "images")).items
    lum = lambda s: ImageStat.Stat(Image.open(path_for(settings.media_dir, s.sha)).convert("L")).mean[0]  # noqa: E731
    assert all(lum(y) < lum(x) - 3 for x, y in zip(a, b))


def test_export_writes_numbered_files(conn, settings, tmp_path):
    d = _photos(tmp_path)
    rid, _ = service.node_run(conn, settings, fake_deps(), "carousel", {"images": str(d)}, {"span": False})
    paths = service.export(conn, settings, out(conn, rid, "carousel", "images"), tmp_path / "post")
    assert [p.name for p in paths] == ["01.jpg", "02.jpg", "03.jpg"] and all(p.stat().st_size > 0 for p in paths)


def test_cache_hit_is_dropped_when_a_member_image_is_gone(conn, settings, tmp_path):
    d = _photos(tmp_path)
    rid, _ = service.node_run(conn, settings, fake_deps(), "carousel", {"images": str(d)}, {"span": False})
    slides = load_images(settings.media_dir, out(conn, rid, "carousel", "images")).items
    path_for(settings.media_dir, slides[1].sha).unlink()
    rid2, _ = service.node_run(conn, settings, fake_deps(), "carousel", {"images": str(d)}, {"span": False})
    again = load_images(settings.media_dir, out(conn, rid2, "carousel", "images")).items
    assert path_for(settings.media_dir, again[1].sha).is_file()


def test_too_many_slides_is_flagged(conn, settings, tmp_path):
    d = tmp_path / "many"
    d.mkdir()
    for i in range(21):
        Image.new("RGB", (100, 125), (i * 10, 0, 0)).save(d / f"{i}.jpg")
    rid, _ = service.node_run(conn, settings, fake_deps(), "carousel", {"images": str(d)}, {"size": "200x250"})
    assert "TOO_MANY_SLIDES" in flags(conn, rid, "carousel")
