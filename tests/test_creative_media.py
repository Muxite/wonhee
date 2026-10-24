"""Real CPU media derivatives and concurrent receipt safety for the creative integration.

Design: docs/specs/2026-10-24-moeka-integration-design.md, approved contact sheet and preview nodes.
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from tests.fakes import fake_deps
from wonhee import creative, media, service
from wonhee.db import queries, repo
from wonhee.db.connect import open_db


def test_contact_sheet_samples_real_frames_with_timestamp_bands(tiny_video, tmp_path):
    """design: contact_sheet, real frame samples retain the clip's changing visual content."""
    from wonhee.analyze.contact import contact_sheet

    output = tmp_path / "contact.jpg"
    contact_sheet(tiny_video, output, frames=6)
    with Image.open(output) as image:
        assert image.format == "JPEG" and image.size == (960, 528)
        first = image.getpixel((160, 119))
        middle = image.getpixel((800, 119))
        last = image.getpixel((800, 383))
        assert first[0] > first[1] + 80 and first[0] > first[2] + 80
        assert middle[2] > middle[0] + 80 and middle[2] > middle[1] + 80
        assert last[1] > last[0] + 50 and last[1] > last[2] + 50
        # Timestamp lettering occupies the label band, rather than being painted into a frame.
        label = image.crop((0, 244, 100, 262)).convert("L")
        assert label.getextrema()[1] > 180


def test_contact_sheet_node_is_cpu_cached_and_rejects_invalid_params(conn, settings, tiny_video):
    """design: contact_sheet, inspection is a typed cached CPU node with bounded frame counts."""
    from wonhee.nodes import registry

    node = registry.get("contact_sheet")
    assert node.lane == "none"
    assert node.version == 1
    with pytest.raises(ValueError):
        node.params.model_validate({"frames": 0})
    with pytest.raises(ValueError):
        node.params.model_validate({"frames": 6, "unexpected": True})
    ref = repo.save_file(conn, settings.media_dir, tiny_video, "video", "video/mp4")
    run_id, _ = service.node_run(conn, settings, fake_deps(), "contact_sheet", {"video": ref})
    output = service._step_out(conn, run_id, "contact_sheet")["image"]
    assert media.path_for(settings.media_dir, output).is_file()
    again, _ = service.node_run(conn, settings, fake_deps(), "contact_sheet", {"video": ref})
    assert service._step_out(conn, again, "contact_sheet")["image"] == output
    step = queries.current_step(conn, again, "contact_sheet")
    assert "CACHED" in {row["code"] for row in queries.step_flags(conn, step["id"])}


def test_media_info_analysis_returns_image_content_and_saved_asset_transcript(conn, settings, tiny_video, monkeypatch):
    """design: What agents see, explicit inspection returns actual image bytes and attributed transcripts."""
    from wonhee import mcp_server

    monkeypatch.setattr(creative, "_docstore", lambda *a, **k: None)
    asset = creative.inbox_add(conn, settings, source=str(tiny_video))
    creative.memory_remember(conn, settings, kind="transcript", content="A saved transcript for this take",
                             source={"asset_id": asset["id"]})
    creative.memory_remember(conn, settings, kind="transcript", content="Another asset's transcript",
                             source={"asset_id": asset["id"] + 100})
    unopened = creative.media_info(conn, settings, asset_id=asset["id"])
    assert "contact_sheet" not in unopened
    assert [row["content"] for row in unopened["transcripts"]] == ["A saved transcript for this take"]

    result = mcp_server.media_info(asset["id"], analyze=True)
    assert not result.isError
    assert result.structuredContent["contact_sheet"]["run_id"]
    images = [block for block in result.content if block.type == "image"]
    assert len(images) == 1 and images[0].mimeType == "image/jpeg"
    with Image.open(io.BytesIO(base64.b64decode(images[0].data))) as image:
        assert image.size == (960, 528)
    records = json.loads(next(block.text for block in result.content if block.type == "text"))
    assert records["id"] == asset["id"]


def test_preview_node_encodes_playable_bounded_cpu_derivative_without_changing_original(conn, settings, tiny_video):
    """design: delivery_preview, bounded playable derivatives keep the original ref intact."""
    from wonhee.nodes import registry

    node = registry.get("delivery_preview")
    assert node.lane == "none"
    original = media.sha256_file(tiny_video)
    ref = repo.save_file(conn, settings.media_dir, tiny_video, "video", "video/mp4")
    run_id, _ = service.node_run(conn, settings, fake_deps(), "delivery_preview", {"video": ref},
                                 {"max_seconds": 1, "width": 128, "max_bytes": 1_000_000})
    output = service._step_out(conn, run_id, "delivery_preview")
    assert output is not None
    derivative = media.path_for(settings.media_dir, output["video"])
    probe = media.probe(derivative)
    assert probe.width == 128 and 900 <= probe.duration_ms <= 1100
    assert derivative.stat().st_size < 1_000_000
    assert media.has_audio(derivative)
    assert media.sha256_file(tiny_video) == original == ref
    assert output["video"] != ref


def test_receipt_lock_serializes_same_source_before_remote_download(conn, settings, tiny_video, monkeypatch):
    """design: Intake, reconnecting clients cannot concurrently download the same durable receipt."""
    from wonhee import intake

    first_downloading = threading.Event()
    second_attempted_lock = threading.Event()
    release_download = threading.Event()
    count_lock = threading.Lock()
    download_calls, lock_calls = [], []
    original_flock = creative.fcntl.flock

    def flock(fd, operation):
        with count_lock:
            lock_calls.append(fd)
            if len(lock_calls) == 2:
                second_attempted_lock.set()
        return original_flock(fd, operation)

    def download(url, dest, **kwargs):
        download_calls.append(url)
        first_downloading.set()
        assert release_download.wait(5), "test did not release the first download"
        shutil.copyfile(tiny_video, dest)
        return {"filename": "take.mp4", "mime": "video/mp4", "bytes": dest.stat().st_size}

    def add():
        other = open_db(settings)
        try:
            return creative.inbox_add(other, settings, source="https://drive.google.com/file/d/testtake/view",
                                      source_key="telegram:owner:drive:1")
        finally:
            other.close()

    monkeypatch.setattr(creative.fcntl, "flock", flock)
    monkeypatch.setattr(intake, "download_drive", download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(add)
        assert first_downloading.wait(5)
        second = pool.submit(add)
        try:
            assert second_attempted_lock.wait(5)
            assert len(download_calls) == 1
        finally:
            release_download.set()
        a, b = first.result(timeout=5), second.result(timeout=5)
    assert a["id"] == b["id"] and len(download_calls) == 1
    assert conn.execute("SELECT count(*) FROM creative_intake").fetchone()[0] == 1


def test_original_document_flag_and_persisted_download_endpoint_survive_new_settings(conn, settings):
    """design: Delivery, an original can be requested as a document using the daemon's private endpoint."""
    ref = repo.save_bytes(conn, settings.media_dir, b"original-bytes", "video/mp4")
    root = settings.home / "creative-outbox"
    root.mkdir(parents=True, exist_ok=True)
    (root / "endpoint.json").write_text(json.dumps({"url": "http://100.66.112.22:17420"}))
    output = creative.outbox_add(conn, settings, ref=ref, filename="original.mp4", as_document=True)
    assert output["as_document"] == 1
    assert output["url"] == f"http://100.66.112.22:17420/files/{output['token']}"
    assert Path(output["path"]).read_bytes() == b"original-bytes"


def test_index_lock_prevents_concurrent_rewriters_from_duplicating_chunks(conn, settings, monkeypatch):
    """design: Creative memory, background maintenance and querying share one derived-index writer."""
    monkeypatch.setattr(creative, "_docstore", lambda *a, **k: None)
    creative.memory_remember(conn, settings, content="A stable creator preference")
    clearing = threading.Event()
    second_attempted_lock = threading.Event()
    release_index = threading.Event()
    count_lock = threading.Lock()
    lock_calls, cleared, added = [], [], []
    original_flock = creative.fcntl.flock

    def flock(fd, operation):
        with count_lock:
            lock_calls.append(fd)
            if len(lock_calls) == 2:
                second_attempted_lock.set()
        return original_flock(fd, operation)

    class Store:
        available = True

        def clear(self, *, source, **kwargs):
            cleared.append(source)
            clearing.set()
            assert release_index.wait(5), "test did not release the first index writer"

        def add(self, content, **kwargs):
            added.append(content)
            return 1

    def index():
        other = open_db(settings)
        try:
            return creative._index_pending(other, settings)
        finally:
            other.close()

    monkeypatch.setattr(creative.fcntl, "flock", flock)
    monkeypatch.setattr(creative, "_docstore", lambda *a, **k: Store())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(index)
        assert clearing.wait(5)
        second = pool.submit(index)
        try:
            assert second_attempted_lock.wait(5)
            assert len(cleared) == 1
        finally:
            release_index.set()
        assert first.result(timeout=5) == second.result(timeout=5) == "hybrid"
    assert len(cleared) == len(added) == 1
    assert conn.execute("SELECT count(*) FROM creative_memory WHERE indexed=0").fetchone()[0] == 0
