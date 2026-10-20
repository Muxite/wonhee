from tests.test_nodes_subs import L, flags, out, run_chain
from wonhee.db import repo
from wonhee.nodes.subtitles import Subtitles, load_subs, save_subs


def mk(conn, settings, n, lang="ko"):
    lines = [L(i, i * 1000, i * 1000 + 800, f"src{i}") for i in range(n)]
    return save_subs(conn, settings.media_dir, Subtitles(language=lang, lines=lines))


def reply(ids, glossary=(), **kw):
    return {"lines": [{"id": i, "text": f"tr{i}"} for i in ids],
            "glossary": [{"source": s, "target": t} for s, t in glossary], **kw}


def tr(conn, settings, sha, params=None, extra_in=None):
    node = {"id": "t", "node": "translate", "in": {"subs": sha, **(extra_in or {})}, "params": params or {}}
    return run_chain(conn, settings, [node])


def test_keeps_ids_timings_and_notes(conn, settings, fake_llm):
    sha = mk(conn, settings, 3)
    fake_llm.queue(TranslateReply=[reply([0, 1, 2], [("김", "Kim")], gist="A gist.", notes=["pun on X"],
                                         caption="Hello caption")])
    rid, status = tr(conn, settings, sha)
    assert status == "done"
    subs = load_subs(settings.media_dir, out(conn, rid, "t", "subs"))
    assert subs.language == "en"
    assert [(l.id, l.start_ms, l.end_ms, l.text, l.source) for l in subs.lines] == [
        (i, i * 1000, i * 1000 + 800, f"tr{i}", "speech") for i in range(3)]
    from wonhee.media import path_for
    notes = path_for(settings.media_dir, out(conn, rid, "t", "notes")).read_text()
    assert notes.startswith("A gist.") and "Kim" in notes and "pun on X" in notes
    assert "caption: Hello caption" not in notes  # no caption input given
    assert fake_llm.calls()[0]["prompt"].count("src0") >= 1


def test_batches_share_glossary(conn, settings, fake_llm):
    sha = mk(conn, settings, 170)
    fake_llm.queue(TranslateReply=[reply(range(0, 80), [("민수", "Minsu")]),
                                   reply(range(80, 160), [("민수", "Other")]),
                                   reply(range(160, 170))])
    rid, status = tr(conn, settings, sha)
    assert status == "done"
    calls = fake_llm.calls()
    assert len(calls) == 3
    assert "Minsu" in calls[1]["prompt"] and "Minsu" not in calls[0]["prompt"]
    assert "tr79" in calls[1]["prompt"] and "do not translate these again" in calls[1]["prompt"]
    assert "Minsu" in calls[2]["prompt"] and "Other" not in calls[2]["prompt"]
    assert "gist" in calls[0]["prompt"] and all("gist" not in c["prompt"] for c in calls[1:])
    assert len(load_subs(settings.media_dir, out(conn, rid, "t", "subs")).lines) == 170


def test_missing_ids_retry_then_gap(conn, settings, fake_llm):
    sha = mk(conn, settings, 4)
    fake_llm.queue(TranslateReply=[reply([0, 2], [("김", "Kim")]), reply([1])])
    rid, status = tr(conn, settings, sha)
    assert status == "weak"
    subs = load_subs(settings.media_dir, out(conn, rid, "t", "subs"))
    assert [l.text for l in subs.lines] == ["tr0", "tr1", "tr2", "src3"]
    assert flags(conn, rid, "t").count("TRANSLATION_GAP") == 1
    calls = fake_llm.calls()
    assert len(calls) == 2 and "Kim" in calls[1]["prompt"]
    assert "[1] 1.0s: src1" in calls[1]["prompt"] and "[0] 0.0s: src0" not in calls[1]["prompt"]
    assert "do not translate these again" in calls[1]["prompt"] and "[2] src2 => tr2" in calls[1]["prompt"]
    from wonhee.db import queries
    det = [f["detail"] for f in queries.step_flags(conn, queries.current_step(conn, rid, "t")["id"])
           if f["code"] == "TRANSLATION_GAP"]
    assert det == ["ids: 3"]
    from wonhee.db import queries
    assert queries.current_step(conn, rid, "t")["status"] == "weak"


def test_invented_ids_ignored(conn, settings, fake_llm):
    sha = mk(conn, settings, 2)
    fake_llm.queue(TranslateReply=[reply([0, 1, 99])])
    rid, status = tr(conn, settings, sha)
    subs = load_subs(settings.media_dir, out(conn, rid, "t", "subs"))
    assert [l.id for l in subs.lines] == [0, 1] and "TRANSLATION_GAP" not in flags(conn, rid, "t")
    assert len(fake_llm.calls()) == 1


def test_caption_in_batch_one_only(conn, settings, fake_llm):
    acc = repo.upsert_account(conn, "instagram", "u", "u")
    cid = repo.upsert_content(conn, platform="instagram", external_id="V1", url="https://www.instagram.com/reel/V1/",
                              account_id=acc, caption="UNIQUECAPTION")
    sha = mk(conn, settings, 100)
    fake_llm.queue(TranslateReply=[reply(range(0, 80), caption="Cap en"), reply(range(80, 100))])
    rid, status = tr(conn, settings, sha, extra_in={"caption": f"content:{cid}"})
    assert status == "done"
    calls = fake_llm.calls()
    assert "UNIQUECAPTION" in calls[0]["prompt"] and "UNIQUECAPTION" not in calls[1]["prompt"]
    from wonhee.media import path_for
    assert "caption: Cap en" in path_for(settings.media_dir, out(conn, rid, "t", "notes")).read_text()


def test_empty_input_no_calls(conn, settings, fake_llm):
    sha = save_subs(conn, settings.media_dir, Subtitles())
    rid, status = tr(conn, settings, sha)
    assert status == "done" and fake_llm.calls() == []
    assert load_subs(settings.media_dir, out(conn, rid, "t", "subs")).lines == []
    assert "NO_TEXT" in flags(conn, rid, "t")


def test_blank_translation_goes_through_retry(conn, settings, fake_llm):
    sha = mk(conn, settings, 2)
    fake_llm.queue(TranslateReply=[{"lines": [{"id": 0, "text": "tr0"}, {"id": 1, "text": "  "}]}, reply([1])])
    rid, status = tr(conn, settings, sha)
    assert status == "done" and len(fake_llm.calls()) == 2
    assert [l.text for l in load_subs(settings.media_dir, out(conn, rid, "t", "subs")).lines] == ["tr0", "tr1"]


def test_local_engine_uses_the_on_box_model_and_costs_nothing(conn, settings, fake_llm, monkeypatch):
    import json
    from wonhee import localllm
    sent = []

    def post(s, body):
        sent.append(body)
        return {"message": {"content": json.dumps(reply([0, 1]))}}
    monkeypatch.setattr(localllm, "_post", post)
    rid, status = tr(conn, settings, mk(conn, settings, 2), {"engine": "local"})
    assert status == "done" and len(sent) == 1 and fake_llm.calls() == []
    assert [l.text for l in load_subs(settings.media_dir, out(conn, rid, "t", "subs")).lines] == ["tr0", "tr1"]
    assert conn.execute("SELECT coalesce(sum(cost_usd),0) FROM step WHERE run_id=?", (rid,)).fetchone()[0] == 0


def test_translation_drops_words_keeps_timings(conn, settings, fake_llm):
    from wonhee.nodes.subtitles import Line, Word
    ln = Line(id=0, start_ms=100, end_ms=900, text="src0", source="speech",
              words=[Word(start_ms=100, end_ms=900, text="src0")])
    sha = save_subs(conn, settings.media_dir, Subtitles(language="ko", lines=[ln]))
    fake_llm.queue(TranslateReply=[reply([0])])
    rid, status = tr(conn, settings, sha)
    assert status == "done"
    out_ln = load_subs(settings.media_dir, out(conn, rid, "t", "subs")).lines[0]
    assert out_ln.words is None and (out_ln.start_ms, out_ln.end_ms) == (100, 900)
