import subprocess

import pytest
from PIL import Image, ImageChops, ImageStat

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain, store_video
from wonhee import media, procenv
from wonhee.edit.ass import ass_time, pick_font, render_ass
from wonhee.nodes.subtitles import Line, Subtitles, save_subs


def _has_ass():
    r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True,
                       env=procenv.clean_env())
    return " ass " in r.stdout


def subs_of(text, lang=None, start=500, end=2500):
    return Subtitles(language=lang, lines=[Line(id=0, start_ms=start, end_ms=end, text=text, source="speech")])


def test_ass_time():
    assert ass_time(0) == "0:00:00.00" and ass_time(3_723_450) == "1:02:03.45"


def test_escaping_and_wrap():
    a = render_ass(subs_of(r"a {b} c\d"), 320, 568)
    assert "a \\{b\\} c\u29f5d" in a
    long = render_ass(subs_of("word " * 30), 320, 568, max_chars=10)
    text = [l for l in long.splitlines() if l.startswith("Dialogue")][0].split(",,", 1)[1]
    assert text.count("\\N") == 1


def test_style_values():
    a = render_ass(subs_of("hi"), 320, 568)
    assert "PlayResX: 320" in a and "PlayResY: 568" in a and "WrapStyle: 0" in a
    assert "Style: Default,DejaVu Sans,20,&H00FFFFFF" in a and ",1,3,0,2,20,20,125,1" in a


def test_cjk_font_choice():
    assert pick_font(subs_of("hello")) == "DejaVu Sans"
    assert pick_font(subs_of("hello", "zh")) == "Noto Sans CJK SC"
    assert pick_font(subs_of("你好")) == "Noto Sans CJK SC"
    assert pick_font(subs_of("你好"), "Foo") == "Foo"
    assert "Noto Sans CJK SC" in render_ass(subs_of("こんにちは", "ja"), 320, 568)


def frame(path, ms, dest):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{ms / 1000:.3f}", "-i", str(path),
                    "-frames:v", "1", str(dest)], check=True, env=procenv.clean_env())
    return Image.open(dest).convert("L")


def mad(a, b, box):
    return ImageStat.Stat(ImageChops.difference(a.crop(box), b.crop(box))).mean[0]


def test_burn_changes_lower_third_only(conn, settings, tiny_video, tmp_path):
    if not _has_ass():
        pytest.skip("ffmpeg lacks the ass filter")
    sha = store_video(conn, settings, tiny_video)
    ss = save_subs(conn, settings.media_dir, subs_of("Hello burned subtitle", "en", 500, 2500))
    rid, status = run_chain(conn, settings, [{"id": "b", "node": "burn_subtitles",
                                              "in": {"video": sha, "subtitles": ss}}], fake_deps())
    assert status == "done"
    new = out(conn, rid, "b", "video")
    assert new != sha
    src, dst = media.path_for(settings.media_dir, sha), media.path_for(settings.media_dir, new)
    a, b = frame(src, 1500, tmp_path / "a.png"), frame(dst, 1500, tmp_path / "b.png")
    w, h = a.size
    assert mad(a, b, (0, 2 * h // 3, w, h)) > 2
    assert mad(a, b, (0, 0, w, h // 3)) < 2
    assert abs(media.probe(dst).duration_ms - media.probe(src).duration_ms) < 200
    assert media.has_audio(dst)


def test_empty_subtitles_returns_same_sha(conn, settings, tiny_video):
    sha = store_video(conn, settings, tiny_video)
    ss = save_subs(conn, settings.media_dir, Subtitles())
    rid, status = run_chain(conn, settings, [{"id": "b", "node": "burn_subtitles",
                                              "in": {"video": sha, "subtitles": ss}}], fake_deps())
    assert status == "done" and out(conn, rid, "b", "video") == sha
    assert "NO_TEXT" in flags(conn, rid, "b")


def test_literal_backslash_n_is_not_a_break():
    a = render_ass(subs_of("a\\Nb \\h c"), 320, 568)
    text = [l for l in a.splitlines() if l.startswith("Dialogue")][0].split(",,", 1)[1]
    assert "\\N" not in text and "\\h" not in text and "a\u29f5Nb" in text
