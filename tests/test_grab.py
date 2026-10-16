import pytest

from tests.fakes import FakeApify, FakeDownloader, fake_deps, make_item
from wonhee import service
from wonhee.db import queries
from wonhee.sources.apify import ACTOR_SCRAPER

URL = "https://www.instagram.com/reel/AAA/?igsh=tracking"
SHOT_TEXTS = {"shots": [{"ord": 1, "on_screen_text": "POV", "visual": "red"},
                        {"ord": 2, "on_screen_text": None, "visual": "blue"},
                        {"ord": 3, "on_screen_text": None, "visual": "green"}]}


def _deps(tiny_video, items=None, fail=frozenset()):
    apify = FakeApify({ACTOR_SCRAPER: [items if items is not None else [make_item("AAA", "me", "2026-07-05", 1000)]]})
    return apify, fake_deps(apify=apify, downloader=FakeDownloader(tiny_video, fail=fail))


def test_grab_mine_stores_everything(conn, settings, fake_llm, tiny_video):
    fake_llm.queue(ShotTexts=[SHOT_TEXTS])
    apify, deps = _deps(tiny_video)
    run_id, text = service.grab(conn, settings, deps, [URL], "ours")
    assert queries.run_row(conn, run_id)["status"] == "done"
    assert apify.calls[0][1]["directUrls"] == ["https://www.instagram.com/reel/AAA/"]  # tracking query dropped
    assert conn.execute("SELECT relation FROM account WHERE handle='me'").fetchone()[0] == "ours"
    assert conn.execute("SELECT kind FROM project").fetchone()[0] == "library"
    assert conn.execute("SELECT count(*) FROM project_output").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM metric_snapshot").fetchone()[0] == 1
    assert conn.execute("SELECT shot_count FROM v_media_style").fetchone()[0] == 3
    assert "@me (ours)" in text and "plays 1,000" in text and 'text "POV"' in text and "when the pump hits" in text


def test_grab_external_never_downgrades_marked_account(conn, settings, fake_llm, tiny_video):
    fake_llm.queue(ShotTexts=[SHOT_TEXTS])
    service.grab(conn, settings, _deps(tiny_video)[1], [URL], "friend")
    service.grab(conn, settings, _deps(tiny_video)[1], [URL], "external")  # same reel again: cached analysis
    assert conn.execute("SELECT relation FROM account WHERE handle='me'").fetchone()[0] == "friend"
    assert len(fake_llm.calls()) == 1
    assert conn.execute("SELECT count(*) FROM content_item").fetchone()[0] == 1


def test_grab_rejects_non_instagram(conn, settings, tiny_video):
    with pytest.raises(ValueError, match="not an Instagram"):
        service.grab(conn, settings, _deps(tiny_video)[1], ["https://tiktok.com/x"], "ours")
    with pytest.raises(ValueError, match="relation"):
        service.grab(conn, settings, _deps(tiny_video)[1], [URL], "boss")


def test_grab_not_found_fails_clearly(conn, settings, tiny_video):
    run_id, text = service.grab(conn, settings, _deps(tiny_video, items=[{"error": "not_found"}])[1], [URL])
    assert queries.run_row(conn, run_id)["status"] == "failed"
    assert "reel(s) not found" in service.show_run(conn, run_id, settings.media_dir) or "not found" in text


def test_grab_download_failure_is_failed_and_resumable(conn, settings, fake_llm, tiny_video):
    run_id, _ = service.grab(conn, settings, _deps(tiny_video, fail={"https://www.instagram.com/reel/AAA/"})[1], [URL])
    assert queries.run_row(conn, run_id)["status"] == "failed"
    fake_llm.queue(ShotTexts=[SHOT_TEXTS])
    out = service.resume(conn, settings, _deps(tiny_video)[1], run_id, "grab", {})
    assert queries.run_row(conn, run_id)["status"] == "done" and "shot 1" in out


def test_refetch_replaces_stored_video(conn, settings, fake_llm, tiny_video, tmp_path):
    import shutil, subprocess
    fake_llm.queue(ShotTexts=[SHOT_TEXTS, SHOT_TEXTS])
    service.grab(conn, settings, _deps(tiny_video)[1], [URL], "external")
    first = conn.execute("SELECT media_sha256 FROM content_item").fetchone()[0]
    other = tmp_path / "other.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tiny_video), "-vf", "scale=160:284", "-c:a", "copy",
                    str(other)], check=True)
    service.grab(conn, settings, _deps(other)[1], [URL], "external")  # no refetch: keeps the stored file
    assert conn.execute("SELECT media_sha256 FROM content_item").fetchone()[0] == first
    service.grab(conn, settings, _deps(other)[1], [URL], "external", refetch=True)
    second = conn.execute("SELECT media_sha256 FROM content_item").fetchone()[0]
    assert second != first
    assert conn.execute("SELECT count(*) FROM media_analysis WHERE media_sha256=?", (second,)).fetchone()[0] == 1
