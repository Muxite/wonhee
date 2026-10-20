import json
import re
import subprocess

import pytest

from tests.fakes import FakeApify, FakeDownloader, FakeTranscriber, fake_deps, make_item
from tests.test_chain_run import chain_of, go, start
from wonhee import media, procenv
from wonhee.analyze.speech import Segment, Transcript
from wonhee.db import queries, repo
from wonhee.nodes.subs import render_srt
from wonhee.nodes.subtitles import Line, Subtitles, load_subs, save_subs
from wonhee.sources.apify import ACTOR_SCRAPER


class CountingTranscriber(FakeTranscriber):
    def __init__(self):
        self.n = 0

    def transcribe(self, path):
        self.n += 1
        return Transcript("en", [Segment(400, 1500, " hello there "), Segment(1600, 1700, "  ")])


def store_video(conn, settings, path):
    return repo.save_file(conn, settings.media_dir, path, "video", "video/mp4")


def run_chain(conn, settings, nodes, deps=None):
    chain = chain_of(nodes)
    rid = start(conn, settings, chain)
    status = run_chain_go(conn, settings, chain, rid, deps)
    return rid, status


def run_chain_go(conn, settings, chain, rid, deps):
    from wonhee.nodes.runner import stages_for_chain
    from wonhee.pipeline.orchestrator import run_stages
    return run_stages(conn, settings, deps, rid, stages_for_chain(chain))


def out(conn, rid, step, port):
    return queries.step_outputs(conn, queries.current_step(conn, rid, step)["id"])[port]["ref"]


def flags(conn, rid, step):
    return [f["code"] for f in queries.step_flags(conn, queries.current_step(conn, rid, step)["id"])]


def test_speech_subs_transcribes_when_no_stored_transcript(conn, settings, tiny_video):
    sha = store_video(conn, settings, tiny_video)
    tr = CountingTranscriber()
    rid, status = run_chain(conn, settings, [{"id": "s", "node": "speech_subs", "in": {"video": sha}}],
                            fake_deps(transcriber=tr))
    assert status == "done" and tr.n == 1
    subs = load_subs(settings.media_dir, out(conn, rid, "s", "subtitles"))
    assert [(ln.id, ln.text, ln.source) for ln in subs.lines] == [(0, "hello there", "speech")]
    assert subs.language == "en" and "NO_TEXT" not in flags(conn, rid, "s")


def test_speech_subs_reuses_stored_transcript(conn, settings, tiny_video):
    sha = store_video(conn, settings, tiny_video)
    t = {"language": "ko", "segments": [{"start_ms": 0, "end_ms": 900, "text": "  "},
                                         {"start_ms": 1000, "end_ms": 2000, "text": "안녕"},
                                         {"start_ms": 2000, "end_ms": 2500, "text": "again"}]}
    tsha = repo.save_media_bytes(conn, settings.media_dir, json.dumps(t).encode(), "transcript", "application/json")
    empty = save_subs(conn, settings.media_dir, Subtitles())
    rid0, _ = run_chain(conn, settings, [{"id": "t", "node": "to_srt", "in": {"subtitles": empty}}])
    repo.save_analysis(conn, sha=sha, analyzer_version=1, step_id=queries.current_step(conn, rid0, "t")["id"], first_word_ms=None, language="ko",
                       transcript_sha=tsha, shots=[])
    tr = CountingTranscriber()
    rid, status = run_chain(conn, settings, [{"id": "s", "node": "speech_subs", "in": {"video": sha}}],
                            fake_deps(transcriber=tr))
    assert status == "done" and tr.n == 0
    subs = load_subs(settings.media_dir, out(conn, rid, "s", "subtitles"))
    assert [(ln.id, ln.text) for ln in subs.lines] == [(0, "안녕"), (1, "again")] and subs.language == "ko"


def test_speech_subs_no_audio_gives_empty_and_no_text_flag(conn, settings, tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x568:d=1:r=30",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)], check=True, env=procenv.clean_env())
    assert not media.has_audio(silent)
    sha = store_video(conn, settings, silent)
    tr = CountingTranscriber()
    rid, status = run_chain(conn, settings, [{"id": "s", "node": "speech_subs", "in": {"video": sha}}],
                            fake_deps(transcriber=tr))
    assert status == "done" and tr.n == 0  # NO_TEXT is not weak
    assert load_subs(settings.media_dir, out(conn, rid, "s", "subtitles")).lines == []
    assert "NO_TEXT" in flags(conn, rid, "s")


def L(i, s, e, text, src="speech"):
    return Line(id=i, start_ms=s, end_ms=e, text=text, source=src)


def test_merge_drops_near_duplicates_keeps_distinct(conn, settings):
    a = save_subs(conn, settings.media_dir, Subtitles(language=None, lines=[L(0, 1000, 2000, "Hello, there!"), L(1, 5000, 6000, "bye")]))
    b = save_subs(conn, settings.media_dir, Subtitles(language="en", lines=[
        L(0, 1100, 2100, "hello there", "screen"),      # dup of a0
        L(1, 2100, 3000, "something else", "screen"),   # no overlap
        L(2, 5000, 6000, "totally different", "screen"),  # overlaps but text differs
        L(3, 0, 500, "intro", "screen")]))
    rid, status = run_chain(conn, settings, [{"id": "m", "node": "merge_subs", "in": {"a": a, "b": b}}])
    assert status == "done"
    subs = load_subs(settings.media_dir, out(conn, rid, "m", "subtitles"))
    assert [(ln.id, ln.text) for ln in subs.lines] == [(0, "intro"), (1, "Hello, there!"), (2, "something else"),
                                                       (3, "bye"), (4, "totally different")]
    assert subs.language == "en"


def test_merge_drops_coarse_screen_lines_already_spoken(conn, settings):
    # Real shape from reel DcSv-shxIDR: screen lines are timed by the 2 s sampling, so one spans several
    # speech lines (or only part of one). Text already said nearby is a duplicate; other text stays.
    a = save_subs(conn, settings.media_dir, Subtitles(language="zh", lines=[
        L(0, 1740, 2860, "你会祝福我吗"), L(1, 2860, 8600, "我当然会替你开心"),
        L(2, 9000, 10180, "她可以陪你出门"), L(3, 10180, 12280, "在你难过的时候抱住你"),
        L(4, 66500, 69300, "也会因为你提到别人"), L(5, 69300, 70800, "偷偷计算自己"),
        L(6, 70800, 72500, "还剩下多少时间")]))
    b = save_subs(conn, settings.media_dir, Subtitles(language="zh", lines=[
        L(0, 2000, 4000, "你会祝福我吗", "screen"),               # covers only 43 % of its span: still a dup
        L(1, 10000, 20000, "在你难过的时候抱住你", "screen"),      # one screen line over many speech lines
        L(2, 70000, 74000, "计算自己还剩下多少时间", "screen"),    # split across two speech lines
        L(3, 106000, 108600, "说话中...", "screen")]))            # UI text nobody says: keep
    rid, status = run_chain(conn, settings, [{"id": "m", "node": "merge_subs", "in": {"a": a, "b": b}}])
    assert status == "done"
    subs = load_subs(settings.media_dir, out(conn, rid, "m", "subtitles"))
    assert [ln.text for ln in subs.lines if ln.source == "screen"] == ["说话中..."]
    assert len(subs.lines) == 8


def _merge(conn, settings, a_lines, b_lines):
    a = save_subs(conn, settings.media_dir, Subtitles(language="en", lines=a_lines))
    b = save_subs(conn, settings.media_dir, Subtitles(language="en", lines=b_lines))
    rid, status = run_chain(conn, settings, [{"id": "m", "node": "merge_subs", "in": {"a": a, "b": b}}])
    assert status == "done"
    return load_subs(settings.media_dir, out(conn, rid, "m", "subtitles"))


def test_merge_keeps_short_titles_over_dense_english_speech(conn, settings):
    speech = ["so this is how i get my whole week planned out before the sun comes up",
              "you take a piece of the evening and write down every single thing you want to do",
              "then you wait until the morning and you just start with the hardest one first",
              "and honestly there is no secret to it you just do the work and tell a friend",
              "i used to think routine was boring but a good story needs a strong structure",
              "that is the whole idea and it really does not take that long to set up"]
    a_lines = [L(i, i * 3000, i * 3000 + 3000, t) for i, t in enumerate(speech)]
    titles = ["PART 2", "WAIT FOR IT", "my 5am routine", "tag a friend", "storytime", "no excuses"]
    b_lines = [L(i, i * 3000, i * 3000 + 3000, t, "screen") for i, t in enumerate(titles)]
    subs = _merge(conn, settings, a_lines, b_lines)
    assert [ln.text for ln in subs.lines if ln.source == "screen"] == titles


def test_merge_keeps_mandarin_title_nobody_said(conn, settings):
    subs = _merge(conn, settings, [L(0, 0, 3000, "你会祝福我吗"), L(1, 3000, 6000, "我当然会替你开心")],
                  [L(0, 1000, 3000, "我的天", "screen")])
    assert [ln.text for ln in subs.lines if ln.source == "screen"] == ["我的天"]


def test_merge_drops_long_end_card_replaying_earlier_speech(conn, settings):
    subs = _merge(conn, settings, [L(0, 40000, 44000, "follow me for more tips on how to save money")],
                  [L(0, 54000, 58000, "Follow me for more tips on how to save money!", "screen")])
    assert [ln.source for ln in subs.lines] == ["speech"]


def parse_srt(text):
    out = []
    for block in text.strip().split("\n\n"):
        n, tm, *body = block.split("\n")
        m = re.fullmatch(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})", tm)
        g = list(map(int, m.groups()))
        out.append((int(n), (g[0] * 3600 + g[1] * 60 + g[2]) * 1000 + g[3], (g[4] * 3600 + g[5] * 60 + g[6]) * 1000 + g[7], body))
    return out


def test_srt_timing_and_wrapping():
    subs = Subtitles(lines=[L(0, 3_723_004, 3_725_999, "one two three four five six seven"),
                            L(1, 0, 1, "短い"),
                            L(2, 10, 20, "这是一个很长的中文句子需要换行才行")])
    text = render_srt(subs, 10)
    assert "01:02:03,004 --> 01:02:05,999" in text
    rows = parse_srt(text)
    assert [r[0] for r in rows] == [1, 2, 3]
    assert [(r[1], r[2]) for r in rows] == [(3_723_004, 3_725_999), (0, 1), (10, 20)]
    assert rows[0][3] == ["one two", "three four", "five six", "seven"]
    assert rows[1][3] == ["短い"]
    assert all(len(x) <= 5 for x in rows[2][3]) and "".join(rows[2][3]) == "这是一个很长的中文句子需要换行才行"
    assert render_srt(Subtitles(), 42) == ""


def test_to_srt_node_empty_input_makes_empty_file(conn, settings):
    empty = save_subs(conn, settings.media_dir, Subtitles())
    rid, status = run_chain(conn, settings, [{"id": "t", "node": "to_srt", "in": {"subtitles": empty}}])
    assert status == "done"
    assert media.path_for(settings.media_dir, out(conn, rid, "t", "file")).read_text() == ""


def test_grab_node_in_chain(conn, settings, tiny_video):
    apify = FakeApify({ACTOR_SCRAPER: [[make_item("AAA", "me", "2026-07-05", 1000)]]})
    deps = fake_deps(apify=apify, downloader=FakeDownloader(tiny_video))
    nodes = [{"id": "g", "node": "grab", "params": {"url": "https://www.instagram.com/reel/AAA/?igsh=x"}}]
    rid, status = run_chain(conn, settings, nodes, deps)
    assert status == "done"
    ref, vid = out(conn, rid, "g", "content"), out(conn, rid, "g", "video")
    cid = int(ref.split(":")[1])
    assert conn.execute("SELECT media_sha256 FROM content_item WHERE id=?", (cid,)).fetchone()[0] == vid
    assert apify.calls[0][1]["directUrls"] == ["https://www.instagram.com/reel/AAA/"]
    assert conn.execute("SELECT count(*) FROM project_reference").fetchone()[0] == 1
    rid2, _ = run_chain(conn, settings, nodes, deps)  # not cacheable: grabs again
    assert len(apify.calls) == 2


def test_grab_node_rejects_bad_url(conn, settings, tiny_video):
    nodes = [{"id": "g", "node": "grab", "params": {"url": "https://tiktok.com/x"}}]
    rid, status = run_chain(conn, settings, nodes, fake_deps())
    assert status == "failed"
    assert "NO_CANDIDATES" in flags(conn, rid, "g")


def test_grab_refetch_failure_stays_weak(conn, settings, fake_llm, tiny_video):
    from tests.test_grab import SHOT_TEXTS
    from wonhee import service
    urls = ["https://www.instagram.com/reel/AAA/", "https://www.instagram.com/reel/BBB/"]
    items = [make_item("AAA", "me", "2026-07-05", 1000), make_item("BBB", "me", "2026-07-06", 10)]

    def deps(fail=frozenset()):
        return fake_deps(apify=FakeApify({ACTOR_SCRAPER: [items]}), downloader=FakeDownloader(tiny_video, fail=fail))

    fake_llm.queue(ShotTexts=[SHOT_TEXTS])
    service.grab(conn, settings, deps(), urls)
    # AAA already has media; its refetch fails, BBB's succeeds: still weak, with a DOWNLOAD_FAILED flag
    run_id, _ = service.grab(conn, settings, deps(fail={urls[0]}), urls, refetch=True)
    step = queries.current_step(conn, run_id, "grab")
    assert step["status"] == "weak"
    assert "DOWNLOAD_FAILED" in [f["code"] for f in queries.step_flags(conn, step["id"])]


def test_speech_subs_puts_words_on_lines(conn, settings, tiny_video):
    from wonhee.analyze.speech import SpokenWord

    class Words(FakeTranscriber):
        def transcribe(self, path):
            return Transcript("en", [Segment(400, 1500, " hello there ", [SpokenWord(400, 900, "hello"),
                                                                         SpokenWord(950, 1500, "there")])])

    sha = store_video(conn, settings, tiny_video)
    rid, status = run_chain(conn, settings, [{"id": "s", "node": "speech_subs", "in": {"video": sha}}],
                            fake_deps(transcriber=Words()))
    assert status == "done"
    ln = load_subs(settings.media_dir, out(conn, rid, "s", "subtitles")).lines[0]
    assert [(w.start_ms, w.end_ms, w.text) for w in ln.words] == [(400, 900, "hello"), (950, 1500, "there")]
    from wonhee.nodes.subtitles import line_words
    assert line_words(ln)[1] is False


def test_stored_transcript_without_words_keeps_lines_wordless(conn, settings, tiny_video):
    """No re-run of whisper for words: lines stay word-less and consumers estimate."""
    sha = store_video(conn, settings, tiny_video)
    t = {"language": "en", "segments": [{"start_ms": 0, "end_ms": 1000, "text": "a b"}]}
    tsha = repo.save_media_bytes(conn, settings.media_dir, json.dumps(t).encode(), "transcript", "application/json")
    empty = save_subs(conn, settings.media_dir, Subtitles())
    rid0, _ = run_chain(conn, settings, [{"id": "t", "node": "to_srt", "in": {"subtitles": empty}}])
    repo.save_analysis(conn, sha=sha, analyzer_version=1, step_id=queries.current_step(conn, rid0, "t")["id"],
                       first_word_ms=None, language="en", transcript_sha=tsha, shots=[])
    tr = CountingTranscriber()
    rid, _ = run_chain(conn, settings, [{"id": "s", "node": "speech_subs", "in": {"video": sha}}],
                       fake_deps(transcriber=tr))
    assert tr.n == 0
    assert load_subs(settings.media_dir, out(conn, rid, "s", "subtitles")).lines[0].words is None


def test_line_words_estimates_evenly_and_cjk_per_character():
    from wonhee.nodes.subtitles import Word, line_words
    ws, est = line_words(L(0, 1000, 2000, "one two  three four"))
    assert est and [w.text for w in ws] == ["one", "two", "three", "four"]
    assert ws[0].start_ms == 1000 and ws[-1].end_ms == 2000
    assert all(a.end_ms == b.start_ms for a, b in zip(ws, ws[1:]))
    ws, est = line_words(L(0, 0, 900, "안녕하세"))
    assert est and [w.text for w in ws] == list("안녕하세") and ws[1].start_ms == 225
    assert line_words(L(0, 0, 10, "  ")) == ([], False)
    real = Line(id=0, start_ms=0, end_ms=10, text="x", source="speech", words=[Word(start_ms=0, end_ms=10, text="x")])
    assert line_words(real)[1] is False


def test_words_are_omitted_from_json_when_none_and_old_json_loads():
    a = Subtitles(language="en", lines=[L(0, 0, 10, "x")])
    assert "words" not in a.model_dump_json()
    old = '{"language":"en","lines":[{"id":0,"start_ms":0,"end_ms":10,"text":"x","source":"speech"}]}'
    assert a.model_dump_json() == old
    assert Subtitles.model_validate_json(old).lines[0].words is None
    from wonhee.nodes.subtitles import Word
    b = Subtitles(lines=[Line(id=0, start_ms=0, end_ms=5, text="x", source="speech",
                              words=[Word(start_ms=0, end_ms=5, text="x")])])
    assert Subtitles.model_validate_json(b.model_dump_json()).lines[0].words[0].text == "x"
    assert "x" in render_srt(b, 42)


def test_merge_carries_words_on_a_lines(conn, settings):
    from wonhee.nodes.subtitles import Word
    w = [Word(start_ms=0, end_ms=500, text="hi")]
    a = save_subs(conn, settings.media_dir, Subtitles(lines=[Line(id=0, start_ms=0, end_ms=500, text="hi", source="speech", words=w)]))
    b = save_subs(conn, settings.media_dir, Subtitles(lines=[L(0, 9000, 9500, "unrelated card text", "screen")]))
    rid, status = run_chain(conn, settings, [{"id": "m", "node": "merge_subs", "in": {"a": a, "b": b}}])
    assert status == "done"
    lines = load_subs(settings.media_dir, out(conn, rid, "m", "subtitles")).lines
    assert lines[0].words == w and lines[1].words is None
