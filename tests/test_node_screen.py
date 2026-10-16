import shutil
import subprocess

import pytest

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain, store_video
from wonhee import procenv
from wonhee.analyze.frames import extract_every, extract_frame
from wonhee.analyze.phash import phash_hex
from wonhee.nodes.subtitles import Line, Subtitles, load_subs, save_subs

SRC = ["smptebars=s=320x568:r=30:d=3", "rgbtestsrc=s=320x568:r=30:d=3",
       "color=c=white:s=320x568:r=30:d=3,drawgrid=w=40:h=40:t=6:c=black"]


@pytest.fixture(scope="module")
def three_seg(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("screen")
    out_ = d / "v.mp4"
    args = ["ffmpeg", "-loglevel", "error", "-y"]
    for s in SRC:
        args += ["-f", "lavfi", "-i", s]
    args += ["-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[v]", "-map", "[v]",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_)]
    subprocess.run(args, check=True, env=procenv.clean_env())
    hs = [int(phash_hex(extract_frame(out_, t, d / f"c{t}.jpg")), 16) for t in (1500, 4500, 7500)]
    assert len(hs) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            assert (hs[i] ^ hs[j]).bit_count() > 4
    return out_


def rd(*texts):
    return {"frames": [{"index": i + 1, "text": t} for i, t in enumerate(texts)]}


def run(conn, settings, sha, params=None, ctx_sha=None):
    inp = {"video": sha, **({"context": ctx_sha} if ctx_sha else {})}
    return run_chain(conn, settings, [{"id": "s", "node": "screen_text", "in": inp, "params": params or {}}],
                     fake_deps())


def test_extract_every_times(three_seg, tmp_path):
    fr = extract_every(three_seg, 1.0, tmp_path / "f")
    assert [ms for ms, _ in fr] == list(range(0, 9000, 1000))


def test_extract_every_downscales_never_upscales(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    from PIL import Image
    for size, want in (("1080x1920", (720, 1280)), ("320x568", (320, 568))):
        v = tmp_path / f"{size}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        f"smptebars=s={size}:r=10:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(v)],
                       check=True, env=procenv.clean_env())
        fr = extract_every(v, 1.0, tmp_path / f"o{size}", 1280)
        assert fr
        with Image.open(fr[0][1]) as im:
            assert im.size == want


def test_three_reads_three_lines(conn, settings, fake_llm, three_seg):
    sha = store_video(conn, settings, three_seg)
    fake_llm.queue(ScreenRead=[rd("A", "B", "C")])
    rid, status = run(conn, settings, sha, {"every_s": 1})
    assert status == "done"
    assert len(fake_llm.calls()) == 1 and len(fake_llm.calls()[0]["images"]) == 3
    assert [(l.start_ms, l.end_ms) for l in load_subs(settings.media_dir, out(conn, rid, "s", "subtitles")).lines] \
        == [(0, 3000), (3000, 6000), (6000, 9000)]
    subs = load_subs(settings.media_dir, out(conn, rid, "s", "subtitles"))
    assert [l.text for l in subs.lines] == ["A", "B", "C"]
    assert all(l.source == "screen" for l in subs.lines)


def test_equal_adjacent_text_merges(conn, settings, fake_llm, three_seg):
    sha = store_video(conn, settings, three_seg)
    fake_llm.queue(ScreenRead=[rd("Hello!", "hello", "Bye")])
    rid, _ = run(conn, settings, sha, {"every_s": 1})
    subs = load_subs(settings.media_dir, out(conn, rid, "s", "subtitles"))
    assert [l.text for l in subs.lines] == ["Hello!", "Bye"]
    assert subs.lines[0].end_ms >= 5000


def test_context_in_prompt_and_language(conn, settings, fake_llm, three_seg):
    sha = store_video(conn, settings, three_seg)
    cs = save_subs(conn, settings.media_dir,
                   Subtitles(language="zh", lines=[Line(id=0, start_ms=0, end_ms=900, text="阿妩来了", source="speech")]))
    fake_llm.queue(ScreenRead=[rd("阿妩", None, None)])
    rid, _ = run(conn, settings, sha, {"every_s": 1}, cs)
    p = fake_llm.calls()[0]["prompt"]
    assert "阿妩来了" in p and "image 2 = t " in p and "BEGIN speech in this video" in p
    assert load_subs(settings.media_dir, out(conn, rid, "s", "subtitles")).language == "zh"


def test_out_of_range_index_ignored(conn, settings, fake_llm, three_seg):
    sha = store_video(conn, settings, three_seg)
    fake_llm.queue(ScreenRead=[{"frames": [{"index": 9, "text": "x"}, {"index": 0, "text": "y"},
                                           {"index": 1, "text": "ok"}]}])
    rid, _ = run(conn, settings, sha, {"every_s": 1})
    assert [l.text for l in load_subs(settings.media_dir, out(conn, rid, "s", "subtitles")).lines] == ["ok"]


def test_max_frames_widens_every_s(conn, settings, fake_llm, three_seg):
    from wonhee.db import queries
    from wonhee.media import path_for
    import json
    sha = store_video(conn, settings, three_seg)
    fake_llm.queue(ScreenRead=[rd("A", "B", "C")])
    rid, status = run(conn, settings, sha, {"every_s": 0.5, "max_frames": 3})
    assert status == "done"
    arts = queries.step_artifacts(conn, queries.current_step(conn, rid, "s")["id"])
    rec = [json.loads(path_for(settings.media_dir, a["blob_sha256"]).read_text()) for a in arts
           if a["role"] == "api_response"]
    assert abs(rec[0]["every_s_used"] - 3.0) < 0.1
    assert rec[0]["frames_sampled"] == 3


def test_no_text_flag(conn, settings, fake_llm, three_seg):
    sha = store_video(conn, settings, three_seg)
    fake_llm.queue(ScreenRead=[rd(None, None, None)])
    rid, status = run(conn, settings, sha, {"every_s": 1})
    assert "NO_TEXT" in flags(conn, rid, "s")
    assert load_subs(settings.media_dir, out(conn, rid, "s", "subtitles")).lines == []


def test_scope_reaches_prompt():
    from wonhee.nodes.screen import _prompt
    assert "phone or device screens" in _prompt([0], None) and "phone or device screens" not in _prompt([0], None, "all")
