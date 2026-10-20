from datetime import date

from wonhee.db import queries, repo
from wonhee.pipeline.context import Stage
from wonhee.pipeline.orchestrator import run_stages
from wonhee.scout.stages import discover, fetch, pick, resolve
from wonhee.sources.apify import ACTOR_HASHTAG, ACTOR_SCRAPER
from tests.fakes import FakeApify, FakeDownloader, fake_deps, make_item

TODAY = date(2026, 10, 15)
NOW = {"mode": "discover", "niche": "gym", "window": {"start": "2026-09-15", "end": "2026-10-15", "label": "now"}, "constraints": {}}
STAGES = [Stage("resolve", (), resolve.run), Stage("discover", ("resolve",), discover.run),
          Stage("pick", ("discover",), pick.run), Stage("fetch", ("pick",), fetch.run)]


def _setup(conn, settings, fake_llm, apify, downloader):
    fake_llm.queue(HashtagPlan=[{"tags": ["gym"]}, {"tags": ["gym"]}],
                   ClusterVerdict=[{"items": [{"key": "7", "name": "T7", "summary": "s", "is_trend": True},
                                              {"key": "8", "name": "T8", "summary": "s", "is_trend": True}]}] * 2,
                   PickChoice=[{"trend_id": 1, "reason": "most filmable", "secondary_ids": [2]},
                               {"trend_id": 4, "reason": "again", "secondary_ids": []}])
    i = repo.create_intent(conn, "x", "scout", 1, NOW); repo.confirm_intent(conn, i)
    p = repo.create_project(conn, "x", "scout", i)
    r = repo.start_run(conn, p, 5.0)
    return p, r, fake_deps(apify=apify, downloader=downloader)


def _items():
    return [make_item(c, f"u{c}", "2026-10-01", 100 * n, audio=a) for n, (c, a) in
            enumerate([("A", "7"), ("B", "7"), ("C", "7"), ("D", "8"), ("E", "8")], 1)]


def test_fetch_rescrapes_before_download(conn, settings, fake_llm, tiny_video):
    apify = FakeApify({ACTOR_HASHTAG: [_items()], ACTOR_SCRAPER: [_items()[:3]]})
    dl = FakeDownloader(tiny_video)
    p, r, deps = _setup(conn, settings, fake_llm, apify, dl)
    assert run_stages(conn, settings, deps, r, STAGES, today=TODAY) == "done"
    actors = [a for a, _ in apify.calls]
    assert actors.index(ACTOR_SCRAPER) > actors.index(ACTOR_HASHTAG)
    assert len(dl.calls) == 3
    assert conn.execute("SELECT count(*) FROM project_reference WHERE role='style_ref'").fetchone()[0] == 3


def test_fetch_partial_failure_is_weak(conn, settings, fake_llm, tiny_video):
    apify = FakeApify({ACTOR_HASHTAG: [_items()], ACTOR_SCRAPER: [_items()[:3]]})
    dl = FakeDownloader(tiny_video, fail={"https://www.instagram.com/reel/C/", "https://www.instagram.com/reel/B/"})
    p, r, deps = _setup(conn, settings, fake_llm, apify, dl)
    assert run_stages(conn, settings, deps, r, STAGES, today=TODAY) == "weak"
    codes = {f["code"] for f in queries.step_flags(conn, queries.current_step(conn, r, "fetch")["id"])}
    assert codes == {"DOWNLOAD_FAILED", "LOW_SAMPLE"}


def test_pick_reads_current_discover_only(conn, settings, fake_llm, tiny_video):
    apify = FakeApify({ACTOR_HASHTAG: [_items(), _items()], ACTOR_SCRAPER: [_items()[:3]]})
    p, r, deps = _setup(conn, settings, fake_llm, apify, FakeDownloader(tiny_video))
    run_stages(conn, settings, deps, r, STAGES[:3], today=TODAY)
    run_stages(conn, settings, deps, r, STAGES[:3], start_at="discover", today=TODAY)
    # second discover created trends 3,4; pick chose 4 (valid only in the new set)
    assert queries.primary_trend(conn, p)["trend_id"] == 4


def _flags(conn, r, stage):
    return queries.step_flags(conn, queries.current_step(conn, r, stage)["id"])


import json

from wonhee.media import path_for


def _pick_setup(conn, settings, fake_llm, tiny_video, pick_reply, items=None):
    apify = FakeApify({ACTOR_HASHTAG: [items or _items()], ACTOR_SCRAPER: [_items()[:3]]})
    p, r, deps = _setup(conn, settings, fake_llm, apify, FakeDownloader(tiny_video))
    d = json.loads(fake_llm.replies_path.read_text())
    d["PickChoice"] = [pick_reply]
    d["ClusterVerdict"] = d["ClusterVerdict"][:1]
    fake_llm.replies_path.write_text(json.dumps(d))
    return p, r, deps


def test_pick_single_candidate_makes_no_model_call(conn, settings, fake_llm, tiny_video):
    items = _items()[:3]
    p, r, deps = _pick_setup(conn, settings, fake_llm, tiny_video, {"trend_id": 1, "reason": "x", "secondary_ids": []}, items)
    d = json.loads(fake_llm.replies_path.read_text())
    d["ClusterVerdict"] = [{"items": [{"key": "7", "name": "T7", "summary": "s", "is_trend": True}]}]
    fake_llm.replies_path.write_text(json.dumps(d))
    run_stages(conn, settings, deps, r, STAGES[:3], today=TODAY)
    assert queries.primary_trend(conn, p)["name"] == "T7"
    assert not [c for c in fake_llm.calls() if c["response_format"]["json_schema"]["name"] == "PickChoice"]


def test_pick_invalid_trend_id_uses_most_popular(conn, settings, fake_llm, tiny_video):
    p, r, deps = _pick_setup(conn, settings, fake_llm, tiny_video, {"trend_id": 99, "reason": "x", "secondary_ids": [1]})
    run_stages(conn, settings, deps, r, STAGES[:3], today=TODAY)
    assert queries.primary_trend(conn, p)["name"] == "T8"  # median plays 450 vs 200
    assert conn.execute("SELECT count(*) FROM project_trend WHERE role='secondary'").fetchone()[0] == 0
    flags = {(f["code"], f["detail"]) for f in _flags(conn, r, "pick")}
    assert [c for c, _ in flags] == ["INVALID_PICK"]
    assert queries.current_step(conn, r, "pick")["status"] == "weak"


def test_pick_records_reason_on_step(conn, settings, fake_llm, tiny_video):
    """design: DESIGN trend scout row ("pick records why it chose"); contract 5 (provenance in tables, not notes)."""
    p, r, deps = _pick_setup(conn, settings, fake_llm, tiny_video,
                             {"trend_id": 2, "reason": "easiest to film", "secondary_ids": [1]})
    run_stages(conn, settings, deps, r, STAGES[:3], today=TODAY)
    arts = queries.step_artifacts(conn, queries.current_step(conn, r, "pick")["id"])
    decisions = [json.loads(path_for(settings.media_dir, a["blob_sha256"]).read_text())
                 for a in arts if a["role"] == "llm_reply"]
    rec = [d for d in decisions if d.get("decision") == "pick"]
    assert len(rec) == 1 and rec[0]["reason"] == "easiest to film"
    assert rec[0]["primary"] == 2 and rec[0]["secondary"] == [1] and rec[0]["by"] == "model"


def _fetch_run(conn, settings, fake_llm, deps_kw, tiny_video):
    apify = FakeApify({ACTOR_HASHTAG: [_items()], ACTOR_SCRAPER: deps_kw.get("scrape", [_items()[:3]])})
    dl = deps_kw.get("dl") or FakeDownloader(tiny_video)
    p, r, deps = _setup(conn, settings, fake_llm, apify, dl)
    return p, r, deps, dl


def test_fetch_all_downloads_failing_fails_step(conn, settings, fake_llm, tiny_video):
    urls = {f"https://www.instagram.com/reel/{c}/" for c in "ABC"}
    p, r, deps, _ = _fetch_run(conn, settings, fake_llm, {"dl": FakeDownloader(tiny_video, fail=urls)}, tiny_video)
    assert run_stages(conn, settings, deps, r, STAGES, today=TODAY) == "failed"
    flags = _flags(conn, r, "fetch")
    assert {f["code"] for f in flags} == {"DOWNLOAD_FAILED"}
    assert any("https://www.instagram.com/reel/A/" in f["detail"] for f in flags)


def test_fetch_skips_already_downloaded(conn, settings, fake_llm, tiny_video):
    p, r, deps, dl = _fetch_run(conn, settings, fake_llm, {}, tiny_video)
    assert run_stages(conn, settings, deps, r, STAGES, today=TODAY) == "done"
    dl2 = FakeDownloader(tiny_video)
    deps.downloader = dl2
    n_scrapes = len(deps.apify.calls)
    assert run_stages(conn, settings, deps, r, STAGES, start_at="fetch", today=TODAY) == "done"
    assert dl2.calls == [] and len(deps.apify.calls) == n_scrapes
    assert conn.execute("SELECT count(*) FROM project_reference WHERE role='style_ref'").fetchone()[0] == 3


def test_fetch_item_missing_from_rescrape_is_flagged(conn, settings, fake_llm, tiny_video):
    p, r, deps, dl = _fetch_run(conn, settings, fake_llm, {"scrape": [_items()[:2]]}, tiny_video)
    assert run_stages(conn, settings, deps, r, STAGES, today=TODAY) == "weak"
    details = [f["detail"] for f in _flags(conn, r, "fetch") if f["code"] == "DOWNLOAD_FAILED"]
    assert len(details) == 1 and "https://www.instagram.com/reel/C/" in details[0]


def test_fetch_save_failure_is_flagged_and_others_continue(conn, settings, fake_llm, tiny_video, monkeypatch):
    from wonhee.scout.stages import fetch as fetch_mod
    real = fetch_mod.repo.save_file
    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        return real(*a, **kw)

    monkeypatch.setattr(fetch_mod.repo, "save_file", flaky)
    p, r, deps, dl = _fetch_run(conn, settings, fake_llm, {}, tiny_video)
    assert run_stages(conn, settings, deps, r, STAGES, today=TODAY) == "weak"
    assert len(dl.calls) == 3
    details = [f["detail"] for f in _flags(conn, r, "fetch") if f["code"] == "DOWNLOAD_FAILED"]
    assert len(details) == 1 and "disk full" in details[0] and "instagram.com/reel/" in details[0]
