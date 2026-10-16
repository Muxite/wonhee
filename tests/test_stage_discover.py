from datetime import date

from wonhee.db import queries, repo
from wonhee.pipeline.context import Stage
from wonhee.pipeline.orchestrator import run_stages
from wonhee.scout.stages import discover, resolve
from wonhee.sources.apify import ACTOR_HASHTAG, ACTOR_REELS, ACTOR_SCRAPER
from wonhee.sources.websearch import SearchResult
from tests.fakes import FakeApify, FakeSearch, fake_deps, make_item

TODAY = date(2026, 10, 15)
STAGES = [Stage("resolve", (), resolve.run), Stage("discover", ("resolve",), discover.run)]


def _run(conn, settings, req, deps):
    i = repo.create_intent(conn, "x", "scout", 1, req)
    repo.confirm_intent(conn, i)
    p = repo.create_project(conn, "x", "scout", i)
    r = repo.start_run(conn, p, 5.0)
    return r, run_stages(conn, settings, deps, r, STAGES, today=TODAY)


NOW = {"mode": "discover", "niche": "gym humor",
       "window": {"start": "2026-09-15", "end": "2026-10-15", "label": "now"}, "constraints": {}}
THROWBACK = {"mode": "discover", "niche": "gym humor",
             "window": {"start": "2025-09-15", "end": "2025-11-14", "label": "1 year ago"}, "constraints": {}}


def test_now_path_clusters_by_audio(conn, settings, fake_llm):
    items = [make_item("A", "u1", "2026-10-01", 900, audio="7"), make_item("B", "u2", "2026-10-02", 100, audio="7"),
             make_item("C", "u3", "2026-10-03", 50, audio="8"), make_item("OLD", "u4", "2026-01-01", 9999, audio="8")]
    apify = FakeApify({ACTOR_HASHTAG: [items]})
    fake_llm.queue(HashtagPlan=[{"tags": ["gymhumor", "gymtok", "legday"]}],
                   ClusterVerdict=[{"items": [{"key": "7", "name": "Leg day stairs", "summary": "s", "is_trend": True}]}])
    run, status = _run(conn, settings, NOW, fake_deps(apify=apify))
    assert status == "done"
    step = queries.current_step(conn, run, "discover")["id"]
    (t,) = queries.trends_of_step(conn, step)
    assert t["name"] == "Leg day stairs" and t["example_count"] == 2
    assert conn.execute("SELECT count(*) FROM metric_snapshot").fetchone()[0] == 4  # R14: all seen items snapshotted


def test_throwback_web_path_filters_window(conn, settings, fake_llm):
    search = FakeSearch([SearchResult("https://blog/oct-2025", "Top reels Oct 2025", "the stair one", None)])
    apify = FakeApify({ACTOR_REELS: [[make_item("IN", "coach", "2025-10-10", 500), make_item("OUT", "coach", "2026-02-01", 800)]],
                       ACTOR_SCRAPER: [[]]})
    fake_llm.queue(WebCandidates=[{"items": [{"name": "Stair fail", "summary": "s", "creator_handles": ["coach"],
                                              "example_urls": [], "hashtags": [], "source_ids": [1, 999]}]}])
    run, status = _run(conn, settings, THROWBACK, fake_deps(apify=apify, search=search))
    assert status == "weak"  # one example → LOW_SAMPLE
    step = queries.current_step(conn, run, "discover")["id"]
    (t,) = queries.trends_of_step(conn, step)
    assert t["example_count"] == 1
    assert conn.execute("SELECT count(*) FROM trend_source").fetchone()[0] == 1  # 999 dropped
    assert any("September 2025" in q or "October 2025" in q for q in search.queries)


def test_no_candidates_fails_with_hint(conn, settings, fake_llm):
    fake_llm.queue(HashtagPlan=[{"tags": ["x"]}], ClusterVerdict=[{"items": []}])
    run, status = _run(conn, settings, NOW, fake_deps(apify=FakeApify({ACTOR_HASHTAG: [[]]})))
    assert status == "failed"
    flags = queries.step_flags(conn, queries.current_step(conn, run, "discover")["id"])
    assert flags[0]["code"] == "NO_CANDIDATES" and "wider window" in flags[0]["detail"]
    assert not any("ClusterVerdict" in c["response_format"]["json_schema"]["name"] for c in fake_llm.calls())


def test_missing_apify_key_is_clear(conn, settings, fake_llm):
    from wonhee.pipeline.deps import Missing
    fake_llm.queue(HashtagPlan=[{"tags": ["x"]}])
    run, status = _run(conn, settings, NOW, fake_deps(apify=Missing("APIFY_API_KEY")))
    flags = queries.step_flags(conn, queries.current_step(conn, run, "discover")["id"])
    assert status == "failed" and flags[0]["code"] == "MISSING_KEY"


REF = {"mode": "from_reference", "reference_urls": ["https://www.instagram.com/reel/R1/"],
       "window": {"start": "2026-09-15", "end": "2026-10-15", "label": "now"}, "constraints": {}}


def test_reference_path_collects_same_audio_reels(conn, settings, fake_llm):
    apify = FakeApify({
        ACTOR_SCRAPER: [[make_item("R1", "me", "2026-10-01", 100, audio="9", tags=("gymtok", "legday"))]],
        ACTOR_HASHTAG: [[make_item("S1", "u1", "2026-10-02", 50, audio="9"), make_item("S2", "u2", "2026-10-02", 60, audio="1")]]})
    fake_llm.queue(ClusterVerdict=[{"items": [{"key": "reference", "name": "Ref trend", "summary": "s", "is_trend": True}]}])
    run, status = _run(conn, settings, REF, fake_deps(apify=apify))
    assert status == "done"
    assert [a for a, _ in apify.calls] == [ACTOR_SCRAPER, ACTOR_HASHTAG]
    assert apify.calls[1][1]["hashtags"] == ["gymtok", "legday"]
    (t,) = queries.trends_of_step(conn, queries.current_step(conn, run, "discover")["id"])
    assert t["name"] == "Ref trend" and t["example_count"] == 2
    assert conn.execute("SELECT count(*) FROM project_reference").fetchone()[0] == 1
    (call,) = [c for c in fake_llm.calls() if c["response_format"]["json_schema"]["name"] == "ClusterVerdict"]
    assert "caption R1" in call["prompt"] and "n=2" in call["prompt"] and "plays=150" in call["prompt"]


def test_resolve_fails_when_reference_missing(conn, settings, fake_llm):
    run, status = _run(conn, settings, REF, fake_deps(apify=FakeApify({ACTOR_SCRAPER: [[]]})))
    assert status == "failed"
    (f,) = queries.step_flags(conn, queries.current_step(conn, run, "resolve")["id"])
    assert f["code"] == "NO_CANDIDATES" and "R1" in f["detail"]


def test_web_path_widens_window_without_new_apify_calls(conn, settings, fake_llm):
    search = FakeSearch([SearchResult("https://blog/x", "T", "s", None)])
    apify = FakeApify({ACTOR_REELS: [[make_item("NEAR", "coach", "2025-12-20", 500),
                                      make_item("FAR", "coach", "2027-01-01", 5)]]})
    fake_llm.queue(WebCandidates=[{"items": [{"name": "N", "summary": "s", "creator_handles": ["coach"],
                                              "example_urls": [], "hashtags": [], "source_ids": [1]}]}])
    run, status = _run(conn, settings, THROWBACK, fake_deps(apify=apify, search=search))
    assert status == "weak"
    step = queries.current_step(conn, run, "discover")["id"]
    assert {f["code"] for f in queries.step_flags(conn, step)} == {"WINDOW_WIDENED", "LOW_SAMPLE"}
    assert len(apify.calls) == 1


def test_web_path_limits_model_supplied_inputs(conn, settings, fake_llm):
    search = FakeSearch([SearchResult("https://blog/x", "T", "s", None)])
    apify = FakeApify({ACTOR_REELS: [[make_item("IN", "coach", "2025-10-10", 500)]]})
    handles = ["@x", " y ", "bad handle", ""] + [f"h{i}" for i in range(8)]
    fake_llm.queue(WebCandidates=[{"items": [{"name": "N", "summary": "s", "creator_handles": handles,
                                              "example_urls": ["https://evil.example/a"], "hashtags": [],
                                              "source_ids": [1]}]}])
    _run(conn, settings, THROWBACK, fake_deps(apify=apify, search=search))
    assert [a for a, _ in apify.calls] == [ACTOR_REELS]
    assert apify.calls[0][1]["username"] == ["x", "y", "h0"]


def test_now_path_without_usable_tags_fails_before_apify(conn, settings, fake_llm):
    fake_llm.queue(HashtagPlan=[{"tags": ["", "#", "two words"]}])
    apify = FakeApify()
    run, status = _run(conn, settings, NOW, fake_deps(apify=apify))
    assert status == "failed" and apify.calls == []
    flags = queries.step_flags(conn, queries.current_step(conn, run, "discover")["id"])
    assert flags[0]["code"] == "NO_CANDIDATES" and "more specific niche" in flags[0]["detail"]


def test_search_result_url_reaches_model_and_ig_site_query(conn, settings, fake_llm):
    ig = "https://www.instagram.com/reel/ABC123/"
    search = FakeSearch([SearchResult(ig, "Stair fail reel", "snip", None)])
    apify = FakeApify({ACTOR_SCRAPER: [[make_item("ABC123", "coach", "2025-10-10", 500)]]})
    fake_llm.queue(WebCandidates=[{"items": [{"name": "N", "summary": "s", "creator_handles": [],
                                              "example_urls": [ig], "hashtags": [], "source_ids": [1]}]}])
    _run(conn, settings, THROWBACK, fake_deps(apify=apify, search=search))
    (call,) = [c for c in fake_llm.calls() if c["response_format"]["json_schema"]["name"] == "WebCandidates"]
    assert f"Stair fail reel — {ig} — snip" in call["prompt"]
    assert any(q.startswith("site:instagram.com/reel gym humor") and "October 2025" in q for q in search.queries)


def test_profile_scrape_is_widened_and_limited(conn, settings, fake_llm):
    search = FakeSearch([SearchResult("https://blog/x", "T", "s", None)])
    apify = FakeApify({ACTOR_REELS: [[make_item("IN", "coach", "2025-10-10", 500)]]})
    fake_llm.queue(WebCandidates=[{"items": [{"name": "N", "summary": "s", "creator_handles": ["coach"],
                                              "example_urls": [], "hashtags": [], "source_ids": [1]}]}])
    _run(conn, settings, THROWBACK, fake_deps(apify=apify, search=search))
    inp = apify.calls[0][1]
    assert inp["onlyPostsNewerThan"] == "2025-07-17" and inp["resultsLimit"] == 60  # 2025-09-15 minus 60 days


def test_apify_gets_remaining_budget(conn, settings, fake_llm):
    fake_llm.queue(HashtagPlan=[{"tags": ["x"]}], ClusterVerdict=[{"items": []}])
    apify = FakeApify({ACTOR_HASHTAG: [[make_item("A", "u", "2026-10-01", 5, audio="7")]]})
    _run(conn, settings, NOW, fake_deps(apify=apify))
    assert apify.kwargs[0]["max_total_charge_usd"] == 5.0  # run cap 5.0, nothing spent yet


def test_cost_cap_inside_stage_stops_before_next_apify_call(conn, settings, fake_llm):
    search = FakeSearch([SearchResult("https://blog/x", "T", "s", None)])
    apify = FakeApify({ACTOR_REELS: [[make_item("IN", "coach", "2025-10-10", 500)]]}, cost=6.0)
    fake_llm.queue(WebCandidates=[{"items": [{"name": "N", "summary": "s", "creator_handles": ["coach"],
                                              "example_urls": ["https://www.instagram.com/reel/Z/"],
                                              "hashtags": [], "source_ids": [1]}]}])
    run, status = _run(conn, settings, THROWBACK, fake_deps(apify=apify, search=search))
    assert status == "failed" and len(apify.calls) == 1
    (f,) = queries.step_flags(conn, queries.current_step(conn, run, "discover")["id"])
    assert f["code"] == "COST_CAP" and "$6.00 of $5.00 during discover" in f["detail"]


def test_failed_apify_run_cost_is_recorded(conn, settings, fake_llm):
    from wonhee.sources.apify import ApifyError

    class Failing(FakeApify):
        def run(self, actor, run_input, **kw):
            raise ApifyError("finished FAILED", 0.3)
    fake_llm.queue(HashtagPlan=[{"tags": ["x"]}])
    run, status = _run(conn, settings, NOW, fake_deps(apify=Failing()))
    assert status == "failed"
    assert abs(queries.current_step(conn, run, "discover")["cost_usd"] - 0.3) < 1e-9


def test_widened_window_is_weak_even_with_enough_examples(conn, settings, fake_llm):
    search = FakeSearch([SearchResult("https://blog/x", "T", "s", None)])
    apify = FakeApify({ACTOR_REELS: [[make_item("N1", "coach", "2025-12-20", 500),
                                      make_item("N2", "coach", "2025-12-21", 400)]]})
    fake_llm.queue(WebCandidates=[{"items": [{"name": "N", "summary": "s", "creator_handles": ["coach"],
                                              "example_urls": [], "hashtags": [], "source_ids": [1]}]}])
    run, status = _run(conn, settings, THROWBACK, fake_deps(apify=apify, search=search))
    assert status == "weak"
    step = queries.current_step(conn, run, "discover")
    assert {f["code"] for f in queries.step_flags(conn, step["id"])} == {"WINDOW_WIDENED"}
    assert step["status"] == "weak"
