"""GitHub slop machine, on fakes (no network, no GPU, no browser).
design: docs/specs/2026-10-20-github-slop-design.md (Pipeline, Rules, Surfaces, Tests)."""
import json
from pathlib import Path

import pytest
from PIL import Image

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain
from wonhee import localllm, media, service
from wonhee.edit import slop as draw
from wonhee.edit.slop import SlopScript
from wonhee.nodes import slop as nodes
from wonhee.sources import github

SCRIPT = {"hook": "this repo is a lifestyle", "what": "A tiny parser that turns text into trees.",
          "why": "Because regexes were tried and everyone cried.", "roast": "Forty-seven TODOs, zero regrets.",
          "caption": "tree-hugging parsers", "hashtags": ["#parser", "opensource"]}


def item(name, **kw):
    base = {"full_name": f"acme/{name}", "name": name, "owner": {"login": "acme"}, "html_url": f"https://github.com/acme/{name}",
            "description": "a tiny parser", "stargazers_count": 321, "forks_count": 12, "fork": False, "archived": False,
            "private": False, "size": 1200, "license": {"spdx_id": "MIT"}, "topics": ["parser"],
            "created_at": "2021-03-04T00:00:00Z", "pushed_at": "2026-09-30T00:00:00Z"}
    return base | kw


class FakeGH:
    def __init__(self, items):
        self.items, self.queries = items, []

    def search(self, query, page=1, per_page=30):
        self.queries.append(query)
        return list(self.items)

    def repo(self, full_name):
        return next(i for i in self.items if i["full_name"] == full_name)

    def languages(self, full_name):
        return {"Python": 8000, "Shell": 2000}

    def readme(self, full_name):
        return "# parser\nTurns text into trees. " * 400


@pytest.fixture
def gh(monkeypatch):
    def install(items):
        fake = FakeGH(items)
        monkeypatch.setattr(nodes, "make_client", lambda settings: fake)
        monkeypatch.setattr(nodes, "clone_repo", lambda full, dest: _fake_clone(dest))
        return fake
    return install


def _fake_clone(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def dossier():
    return {"full_name": "acme/parser", "owner": "acme", "name": "parser", "url": "https://github.com/acme/parser",
            "description": "a tiny parser", "stars": 321, "forks": 12, "languages": {"Python": 8000, "Shell": 2000},
            "license": "MIT", "topics": [], "created_at": "2021-03-04T00:00:00Z", "pushed_at": "2026-09-30T00:00:00Z",
            "readme_excerpt": "x", "clone_path": "/nonexistent"}


def test_gh_pick_skips_seen_forks_archived_and_big_repos_and_writes_the_dossier(conn, settings, gh):
    conn.execute("INSERT INTO slop_seen(repo, first_seen) VALUES ('acme/seen','now')")
    gh([item("seen"), item("forked", fork=True), item("old", archived=True), item("huge", size=60_000),
        item("nodesc", description=None), item("good")])
    rid, status = run_chain(conn, settings, [{"id": "p", "node": "gh_pick"}], fake_deps())
    assert status == "done"
    d = json.loads(media.path_for(settings.media_dir, out(conn, rid, "p", "dossier")).read_text())
    assert d["full_name"] == "acme/good" and d["stars"] == 321 and d["license"] == "MIT"
    assert d["languages"] == {"Python": 8000, "Shell": 2000} and len(d["readme_excerpt"]) <= 3000
    assert {"owner", "url", "forks", "topics", "created_at", "pushed_at", "clone_path"} <= set(d)
    assert conn.execute("SELECT 1 FROM slop_seen WHERE repo='acme/good'").fetchone()


def test_gh_pick_with_nothing_fresh_fails_with_a_flag(conn, settings, gh):
    gh([item("forked", fork=True)])
    rid, status = run_chain(conn, settings, [{"id": "p", "node": "gh_pick"}], fake_deps())
    assert status == "failed" and "SLOP_NO_REPO" in flags(conn, rid, "p")


def test_gh_pick_repo_param_picks_that_repo(conn, settings, gh):
    fake = gh([item("a"), item("b")])
    rid, _ = run_chain(conn, settings, [{"id": "p", "node": "gh_pick", "params": {"repo": "acme/b"}}], fake_deps())
    d = json.loads(media.path_for(settings.media_dir, out(conn, rid, "p", "dossier")).read_text())
    assert d["full_name"] == "acme/b" and fake.queries == []


def test_clone_is_shallow_with_a_clean_env_and_a_timeout(monkeypatch, tmp_path):
    seen = {}

    class P:
        returncode, stderr = 0, ""

    def run(argv, **kw):
        seen.update(argv=argv, **kw)
        return P()
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(github, "run_timed", run)
    github.clone("acme/parser", tmp_path / "x")
    assert "--depth" in seen["argv"] and seen["timeout"] > 0
    assert "GITHUB_TOKEN" not in seen["env"]


def test_slop_script_refuses_long_lines_and_retries(conn, settings, monkeypatch):
    long = SCRIPT | {"hook": "one two three four five six seven eight nine ten"}
    replies = iter([long, SCRIPT])
    monkeypatch.setattr(localllm, "_post", lambda s, body: {"message": {"content": json.dumps(next(replies))}})
    from wonhee.nodes.subtitles import save_text
    ref = save_text(conn, settings.media_dir, json.dumps(dossier()))
    rid, status = run_chain(conn, settings, [{"id": "s", "node": "slop_script", "in": {"dossier": ref}}], fake_deps())
    assert status == "done"
    script = json.loads(media.path_for(settings.media_dir, out(conn, rid, "s", "script")).read_text())
    assert script["hook"] == SCRIPT["hook"] and script["hashtags"] == ["parser", "opensource"]


def test_script_limits():
    for field, n in (("hook", 9), ("what", 24), ("why", 24), ("roast", 16)):
        with pytest.raises(ValueError):
            SlopScript.model_validate(SCRIPT | {field: " ".join(["w"] * (n + 1))})
    with pytest.raises(ValueError):
        SlopScript.model_validate(SCRIPT | {"caption": "x" * 301})
    with pytest.raises(ValueError):
        SlopScript.model_validate(SCRIPT | {"hashtags": list("abcdefg")})
    with pytest.raises(ValueError):
        SlopScript.model_validate(SCRIPT | {"extra": 1})


def test_card_facts_come_from_the_dossier_not_the_script():
    rows = dict(draw.stat_lines(dossier()))
    assert rows["stars"] == "321" and rows["licence"] == "MIT"
    lying = SlopScript.model_validate(SCRIPT | {"hook": "999999 stars of GPL"})
    assert dict(draw.stat_lines(dossier()))["stars"] == "321"
    assert lying.hook.startswith("999999")  # the model's words stay on slides 1/2/4; slide 3 never reads the script


def test_render_slides_are_4_by_5_and_credit_the_repo_on_every_slide():
    slides = draw.render_slides(dossier(), SlopScript.model_validate(SCRIPT))
    assert len(slides) == 4 and all(s.size == (1080, 1350) for s in slides)
    for s in slides:
        footer = s.crop((0, 1262, 540, 1350)).convert("L")
        assert footer.getbbox() is not None and footer.getextrema()[1] > 200  # white credit text on a dark band


def _cards_chain(conn, settings, tmp_path):
    from wonhee.nodes.subtitles import save_text
    d = save_text(conn, settings.media_dir, json.dumps(dossier() | {"clone_path": str(tmp_path)}))
    s = save_text(conn, settings.media_dir, json.dumps(SCRIPT | {"hashtags": ["parser"]}))
    return run_chain(conn, settings, [{"id": "c", "node": "slop_cards", "in": {"dossier": d, "script": s}}], fake_deps())


def test_slop_cards_make_four_pngs(conn, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(draw, "graph_still", lambda clone, title, size=None: Image.new("RGB", (1080, 700), "navy"))
    rid, status = _cards_chain(conn, settings, tmp_path)
    assert status == "done" and "SLOP_NO_GRAPH" not in flags(conn, rid, "c")
    from wonhee.nodes.images import load_images
    items = load_images(settings.media_dir, out(conn, rid, "c", "images")).items
    assert len(items) == 4
    for it in items:
        with Image.open(media.path_for(settings.media_dir, it.sha)) as im:
            assert im.size == (1080, 1350) and im.format == "PNG"


def test_slop_cards_without_a_browser_flag_and_still_make_four(conn, settings, tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)

    def no_browser(*a, **k):
        raise draw.ExportUnavailable("no chromium")
    monkeypatch.setattr(draw, "export_png", no_browser)
    rid, status = _cards_chain(conn, settings, tmp_path)
    assert status == "done" and "SLOP_NO_GRAPH" in flags(conn, rid, "c")
    from wonhee.nodes.images import load_images
    assert len(load_images(settings.media_dir, out(conn, rid, "c", "images")).items) == 4


def test_slop_next_writes_sets_and_a_newest_first_feed(conn, settings, gh, monkeypatch):
    gh([item("one"), item("two")])
    monkeypatch.setattr(localllm, "_post", lambda s, body: {"message": {"content": json.dumps(SCRIPT)}})
    monkeypatch.setattr(draw, "graph_still", lambda *a, **k: None)
    ok, text = service.slop_next(conn, settings, fake_deps(), n=2)
    assert ok
    root = settings.home / "slop"
    feed = json.loads((root / "feed.json").read_text())
    assert len(feed) == 2 and feed[0]["created"] >= feed[1]["created"]
    first = root / feed[0]["slides"][0]
    assert first.is_file() and sorted(p.name for p in first.parent.iterdir()) == [
        "01.png", "02.png", "03.png", "04.png", "caption.txt", "script.json"]
    cap = (first.parent / "caption.txt").read_text()
    assert "#parser" in cap and "https://github.com/acme/" in cap
    page = (root / "index.html").read_text()
    assert "feed.json" in page and "30000" in page


def test_slop_render_takes_an_agent_script_and_rejects_a_bad_one(conn, settings, gh, monkeypatch):
    gh([item("agent")])
    monkeypatch.setattr(draw, "graph_still", lambda *a, **k: None)
    ok, text = service.slop_pick(conn, settings, fake_deps(), "acme/agent")
    assert ok and "acme__agent.json" in text
    with pytest.raises(ValueError, match="SlopScript"):
        service.slop_render(conn, settings, fake_deps(), "acme/agent", json.dumps(SCRIPT | {"hook": "a " * 20}))
    ok, text = service.slop_render(conn, settings, fake_deps(), "acme/agent", json.dumps(SCRIPT))
    assert ok and "tree-hugging" in text


def test_serve_binds_loopback_only(settings):
    server = service.slop_serve(settings, 0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


# ---- amended design (2026-10-20 alignment check) ----------------------------------------------------------------

def _pick_and_render(conn, settings, gh, monkeypatch, script, name="agent"):
    gh([item(name)])
    monkeypatch.setattr(draw, "graph_still", lambda *a, **k: None)
    service.slop_pick(conn, settings, fake_deps(), f"acme/{name}")
    return service.slop_render(conn, settings, fake_deps(), f"acme/{name}", json.dumps(script))


def test_digit_not_in_the_dossier_is_refused_on_the_agent_path(conn, settings, gh, monkeypatch):
    """design: github-slop §Rules numbers are checked in code."""
    with pytest.raises(ValueError, match="10000"):
        _pick_and_render(conn, settings, gh, monkeypatch, SCRIPT | {"caption": "used by 10000 teams"})
    ok, _ = service.slop_render(conn, settings, fake_deps(), "acme/agent", json.dumps(SCRIPT | {"caption": "321 stars"}))
    assert ok  # 321 is the dossier's star count


def test_digit_guard_makes_the_local_model_retry(conn, settings, monkeypatch):
    """design: github-slop §Rules; validation errors go back to the model for one retry."""
    lie = SCRIPT | {"what": "Used by 5000 companies worldwide."}
    replies = iter([lie, SCRIPT])
    monkeypatch.setattr(localllm, "_post", lambda s, body: {"message": {"content": json.dumps(next(replies))}})
    from wonhee.nodes.subtitles import save_text
    ref = save_text(conn, settings.media_dir, json.dumps(dossier()))
    rid, status = run_chain(conn, settings, [{"id": "s", "node": "slop_script", "in": {"dossier": ref}}], fake_deps())
    assert status == "done"
    assert "5000" not in media.path_for(settings.media_dir, out(conn, rid, "s", "script")).read_text()


def test_small_language_shares_print_as_under_one_percent():
    """design: github-slop §slop_cards: shares under 1% print as <1%."""
    assert draw.pct(0.004) == "<1%" and draw.pct(0.5) == "50%" and draw.pct(0.0149) == "1%"


def test_slide_three_is_pixel_identical_whatever_the_script_says():
    """design: github-slop §Tests: slide-3 facts and licence come from the dossier only."""
    honest = draw.render_slides(dossier(), SlopScript.model_validate(SCRIPT))
    lying = draw.render_slides(dossier(), SlopScript.model_validate(
        SCRIPT | {"hook": "GPL 99999 stars", "what": "GPL licence, 99999 stars", "why": "MIT is a lie"}))
    assert honest[2].tobytes() == lying[2].tobytes()
    assert dict(draw.stat_lines(dossier()))["licence"] == "MIT"
    assert draw.stat_lines(dossier() | {"license": None})[-1] == ("licence", "none stated")


def test_graph_with_no_visible_edges_is_unreadable():
    """design: github-slop §slop_cards: an unreadable graph counts as no graph."""
    assert draw._dense_crop(Image.new("RGB", (1080, 700), (13, 13, 16))) is None
    ok = Image.new("RGB", (1080, 700), (13, 13, 16))
    ok.paste(Image.new("RGB", (400, 4), draw.EDGE_RGB), (100, 100))
    ok.paste(Image.new("RGB", (400, 4), draw.EDGE_RGB), (100, 300))
    assert draw._dense_crop(ok) is not None


def test_feed_is_in_pick_order_even_with_equal_timestamps(conn, settings, gh, monkeypatch):
    """design: github-slop §Tests: newest first by order of picking, not only by timestamp."""
    monkeypatch.setattr("wonhee.clock.now_iso", lambda: "2026-10-20T00:00:00+00:00")
    gh([item("first")])
    monkeypatch.setattr(draw, "graph_still", lambda *a, **k: None)
    service.slop_pick(conn, settings, fake_deps(), "acme/first")
    gh([item("first"), item("second")])
    service.slop_pick(conn, settings, fake_deps(), "acme/second")
    service.slop_render(conn, settings, fake_deps(), "acme/first", json.dumps(SCRIPT))
    service.slop_render(conn, settings, fake_deps(), "acme/second", json.dumps(SCRIPT))
    feed = json.loads((settings.home / "slop" / "feed.json").read_text())
    assert [e["repo"] for e in feed] == ["acme/second", "acme/first"]


def test_rate_limit_and_clone_failure_flag(conn, settings, gh, monkeypatch):
    """design: github-slop §Store flags SLOP_RATE_LIMIT, SLOP_CLONE_FAILED."""
    fake = gh([item("a")])

    def limited(*a, **k):
        raise github.RateLimited("slow down")
    monkeypatch.setattr(fake, "search", limited)
    rid, status = run_chain(conn, settings, [{"id": "p", "node": "gh_pick"}], fake_deps())
    assert status == "failed" and "SLOP_RATE_LIMIT" in flags(conn, rid, "p")

    gh([item("b")])

    def bad_clone(full, dest):
        raise github.CloneFailed("timeout")
    monkeypatch.setattr(nodes, "clone_repo", bad_clone)
    rid, status = run_chain(conn, settings, [{"id": "q", "node": "gh_pick"}], fake_deps())
    assert status == "failed" and "SLOP_CLONE_FAILED" in flags(conn, rid, "q")
    assert conn.execute("SELECT 1 FROM slop_seen WHERE repo='acme/b'").fetchone()  # seen before the clone


def test_hash_signs_are_stripped_in_the_render_path(conn, settings, gh, monkeypatch):
    """design: github-slop §SlopScript: a leading # is stripped, not refused."""
    ok, _ = _pick_and_render(conn, settings, gh, monkeypatch, SCRIPT | {"hashtags": ["#one", "##two"]}, "hash")
    assert ok
    cap = next((settings.home / "slop").glob("*/acme__hash/caption.txt")).read_text()
    assert "#one #two" in cap and "##" not in cap


def test_serve_allows_only_the_page_feed_and_dated_sets():
    """design: github-slop §Surfaces serve."""
    ok = ["/", "/index.html", "/feed.json", "/2026-10-20/a__b/01.png", "/feed.json?x=1"]
    bad = ["/repos", "//repos", "///repos/x/.git/config", "/%2Frepos", "/%2frepos/x", "/repos%2Fx", "/dossiers/a.json",
           "/2026-10-20/../repos/x", "/2026-10-20/%2e%2e/repos", "/%252e", "/other", "/\\repos"]
    assert all(service.slop_servable(p) for p in ok)
    assert not any(service.slop_servable(p) for p in bad)


def test_next_deletes_clones_older_than_seven_days(conn, settings):
    """design: github-slop §Store age cap."""
    import os
    import time
    repos = settings.home / "slop" / "repos"
    old, new = repos / "o__old", repos / "o__new"
    old.mkdir(parents=True)
    new.mkdir()
    t = time.time() - 8 * 86400
    os.utime(old, (t, t))
    assert service._slop_prune_clones(settings) == 1
    assert not old.exists() and new.exists()


def test_doctor_reports_github_token_as_info_not_a_warning(settings, monkeypatch):
    """design: github-slop §gh_pick: GITHUB_TOKEN is an optional key."""
    from wonhee import doctor
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    c = {c.name: c for c in doctor.checks(settings)}["key GITHUB_TOKEN"]
    assert c.ok and c.severity == "info"
