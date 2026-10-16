from wonhee import media
from wonhee.db import repo


def test_store_bytes_is_content_addressed(tmp_path):
    sha = media.store_bytes(tmp_path, b"hello")
    p = media.path_for(tmp_path, sha)
    assert p == tmp_path / sha[:2] / sha and p.read_bytes() == b"hello"
    assert media.store_bytes(tmp_path, b"hello") == sha  # no-op second time


def test_probe(tiny_video):
    pr = media.probe(tiny_video)
    assert pr.width == 320 and pr.height == 568 and 2900 <= pr.duration_ms <= 3100 and round(pr.fps) == 30


def test_save_file_idempotent(conn, settings, tiny_video):
    a = repo.save_file(conn, settings.media_dir, tiny_video, "video", "video/mp4")
    b = repo.save_file(conn, settings.media_dir, tiny_video, "video", "video/mp4")
    assert a == b
    assert conn.execute("SELECT count(*) FROM media").fetchone()[0] == 1
    assert conn.execute("SELECT width FROM media").fetchone()[0] == 320


def test_add_phash_idempotent(conn, settings, tiny_video):
    sha = repo.save_file(conn, settings.media_dir, tiny_video, "video", "video/mp4")
    repo.add_phash(conn, sha, 500, "0123456789abcdef")
    repo.add_phash(conn, sha, 500, "0123456789abcdef")
    assert conn.execute("SELECT count(*) FROM media_phash").fetchone()[0] == 1


def test_save_bytes_round_trip(conn, settings):
    sha = repo.save_bytes(conn, settings.media_dir, b"payload", "text/plain")
    assert repo.read_bytes(settings.media_dir, sha) == b"payload"
    assert conn.execute("SELECT bytes, mime FROM blob WHERE sha256=?", (sha,)).fetchone()[:] == (7, "text/plain")


def test_save_media_bytes_transcript(conn, settings):
    sha = repo.save_media_bytes(conn, settings.media_dir, b"hi there", "transcript", "text/plain")
    row = conn.execute("SELECT kind, duration_ms, width, height, fps FROM media WHERE blob_sha256=?", (sha,)).fetchone()
    assert tuple(row) == ("transcript", None, None, None, None)


def test_save_file_unprobed_kind(conn, settings, tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("words")
    sha = repo.save_file(conn, settings.media_dir, f, "transcript", "text/plain")
    row = conn.execute("SELECT kind, duration_ms, width, height, fps FROM media WHERE blob_sha256=?", (sha,)).fetchone()
    assert tuple(row) == ("transcript", None, None, None, None)


def test_probe_non_media_raises_runtime_error(tmp_path):
    import pytest
    f = tmp_path / "junk.bin"
    f.write_bytes(b"not media at all")
    with pytest.raises(RuntimeError, match="junk.bin"):
        media.probe(f)


def test_place_cleans_temp_on_failure(tmp_path):
    import pytest
    def boom(p):
        raise OSError("disk")
    with pytest.raises(OSError):
        media._place(tmp_path, "ab" * 32, boom)
    assert list((tmp_path / "ab").iterdir()) == []
