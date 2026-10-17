import json
from pathlib import Path

import httpx
import pytest

from wonhee.db import repo
from wonhee.sources import apify, fetch, ingest, websearch

FIX = json.loads((Path(__file__).parent / "fixtures" / "apify_reels.json").read_text())


def test_parse_hidden_likes_is_null():
    r = apify.parse_item(FIX[0])
    assert r.external_id == "AAA" and r.likes is None and r.plays == 12000 and r.audio_id == "900"
    assert r.posted_at == "2025-10-10T12:00:00+00:00"


def test_parse_play_count_fallbacks_and_errors():
    assert apify.parse_item(FIX[1]).plays == 5000
    assert apify.parse_item(FIX[2]).plays == 12500
    assert apify.parse_item(FIX[3]) is None


def _step(conn):
    i = repo.create_intent(conn, "t", "scout", 1, {})
    p = repo.create_project(conn, "p", "scout", i)
    return repo.start_step(conn, repo.start_run(conn, p, 1.0), "discover", {}, None)


def test_ingest_duplicate_in_batch(conn):
    s = _step(conn)
    recs = [r for r in map(apify.parse_item, FIX) if r]
    ids = ingest.ingest_records(conn, s, recs, "2026-10-15T00:00:00+00:00")
    assert set(ids) == {"AAA", "BBB"}
    assert conn.execute("SELECT count(*) FROM content_item").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM metric_snapshot").fetchone()[0] == 2  # same instant → first wins
    assert conn.execute("SELECT count(*) FROM audio_track").fetchone()[0] == 1
    row = conn.execute("SELECT audio_id FROM content_item WHERE id=?", (ids["AAA"],)).fetchone()
    assert row[0] is not None
    snap = conn.execute("SELECT plays, likes FROM metric_snapshot WHERE content_id=?", (ids["AAA"],)).fetchone()
    assert tuple(snap) == (12000, None)


def test_brave_parses_results():
    def handler(req):
        assert req.headers["X-Subscription-Token"] == "k"
        return httpx.Response(200, json={"web": {"results": [{"url": "https://a", "title": "A", "description": "d", "age": "1y"}]}})
    b = websearch.BraveSearch("k", "https://brave.test/search", client=httpx.Client(transport=httpx.MockTransport(handler)))
    (r,) = b.search("gym trend october 2025", count=3)
    assert r.url == "https://a" and r.snippet == "d"


def _client_serving(data: bytes):
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=data)))


@pytest.fixture(scope="module")
def video_only(tmp_path_factory):
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("vo") / "vo.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x568:d=1:r=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def _fake_ytdlp(tmp_path, src):
    """A stand-in yt-dlp that writes `src` to its -o path."""
    script = tmp_path / "fake-ytdlp"
    script.write_text(f"#!/bin/sh\nwhile [ \"$1\" != \"-o\" ]; do shift; done\ncp {src} \"$2\"\n")
    script.chmod(0o755)
    return str(script)


def test_download_direct(tmp_path, tiny_video):
    d = fetch.LiveDownloader(client=_client_serving(tiny_video.read_bytes()), ytdlp="false")
    assert d.download("https://cdn/x.mp4", "https://ig/reel/X", tmp_path / "x.mp4") == "direct"
    assert (tmp_path / "x.mp4").read_bytes() == tiny_video.read_bytes()


def test_video_only_direct_falls_back_to_merged_ytdlp(tmp_path, tiny_video, video_only):
    from wonhee.media import has_audio
    d = fetch.LiveDownloader(client=_client_serving(video_only.read_bytes()), ytdlp=_fake_ytdlp(tmp_path, tiny_video))
    assert d.download("https://cdn/x.mp4", "https://ig/reel/X", tmp_path / "x.mp4") == "yt-dlp"
    assert has_audio(tmp_path / "x.mp4")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fake-ytdlp", "x.mp4"]


def test_truly_muted_reel_keeps_direct_file(tmp_path, video_only):
    d = fetch.LiveDownloader(client=_client_serving(video_only.read_bytes()), ytdlp=_fake_ytdlp(tmp_path, video_only))
    assert d.download("https://cdn/x.mp4", "https://ig/reel/X", tmp_path / "x.mp4") == "direct (no audio)"
    assert (tmp_path / "x.mp4").read_bytes() == video_only.read_bytes()


def test_download_falls_back_then_errors(tmp_path, monkeypatch):
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(403)))
    d = fetch.LiveDownloader(client=client, ytdlp="false")  # `false` exits 1
    with pytest.raises(fetch.DownloadError, match="direct: 403"):
        d.download("https://cdn/x.mp4", "https://ig/reel/X", tmp_path / "x.mp4")


def test_download_missing_ytdlp_is_download_error(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(403)))
    d = fetch.LiveDownloader(client=client, ytdlp="/nonexistent/yt-dlp")
    with pytest.raises(fetch.DownloadError) as ei:
        d.download("https://cdn/x.mp4", "https://ig/reel/X", tmp_path / "x.mp4")
    assert "direct" in str(ei.value) and "yt-dlp" in str(ei.value)


def test_download_midstream_failure_leaves_no_part(tmp_path):
    class Boom(httpx.SyncByteStream):
        def __iter__(self):
            yield b"abc"
            raise httpx.ReadError("cut")

    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=Boom())))
    d = fetch.LiveDownloader(client=client, ytdlp="false")
    with pytest.raises(fetch.DownloadError, match="direct"):
        d.download("https://cdn/x.mp4", "https://ig/reel/X", tmp_path / "x.mp4")
    assert list(tmp_path.iterdir()) == []


class _Run:
    def __init__(self, status):
        self.status, self.id, self.default_dataset_id, self.usage_total_usd = status, "r1", "d1", 0.5


class _Handle:
    def __init__(self, run):
        self.run = run

    def call(self, run_input=None, **kw):
        self.kw = kw
        return self.run

    def abort(self):
        self.aborted = True

    def get(self):
        return self.run

    def iterate_items(self):
        return iter([{"a": 1}])


class _FakeClient:
    def __init__(self, status):
        self.h = _Handle(_Run(status))

    def actor(self, name):
        return self.h

    def dataset(self, i):
        return self.h

    def run(self, i):
        return self.h


def test_apify_failed_run_raises():
    with pytest.raises(RuntimeError, match="finished FAILED"):
        apify.LiveApify("t", 1.0, client=_FakeClient("FAILED")).run("a", {})


def test_apify_succeeded_run_returns_items():
    items, cost = apify.LiveApify("t", 1.0, client=_FakeClient("SUCCEEDED")).run("a", {})
    assert items == [{"a": 1}] and cost == 0.5


def test_apify_passes_caps_and_wait():
    from datetime import timedelta
    from decimal import Decimal
    c = _FakeClient("SUCCEEDED")
    apify.LiveApify("t", 1.0, client=c).run("a", {}, max_total_charge_usd=1.5, max_items=40)
    assert c.h.kw["max_total_charge_usd"] == Decimal("1.5") and c.h.kw["max_items"] == 40
    assert c.h.kw["wait_duration"] == timedelta(minutes=10)


def test_apify_failed_run_carries_cost():
    with pytest.raises(apify.ApifyError) as ei:
        apify.LiveApify("t", 1.0, client=_FakeClient("FAILED")).run("a", {})
    assert ei.value.cost_usd == 0.5


def test_apify_timeout_aborts_and_names_run():
    c = _FakeClient("RUNNING")
    with pytest.raises(apify.ApifyError, match=r"a still RUNNING.*run r1"):
        apify.LiveApify("t", 1.0, client=c).run("a", {})
    assert c.h.aborted


def test_muted_audio_id_zero_is_none():
    item = {**FIX[0], "musicInfo": {"audio_id": "0", "song_name": "", "artist_name": "",
                                    "should_mute_audio": True}}
    assert apify.parse_item(item).audio_id is None


def test_parse_photo_posts():
    from tests.fakes import make_photo_item
    from wonhee.sources.apify import parse_item
    one = parse_item(make_photo_item("ONE", "x", "2026-10-01", 5))
    assert one.post_type == "Image" and one.children == [("image", "https://cdn/ONE_0.jpg")] and one.video_url is None
    mix = parse_item(make_photo_item("MIX", "x", "2026-10-01", 5, slides=3, video_slide=2))
    assert [k for k, _ in mix.children] == ["image", "image", "video"]
    item = make_photo_item("NOURL", "x", "2026-10-01", 5)
    del item["url"]
    assert parse_item(item).url == "https://www.instagram.com/p/NOURL/"
