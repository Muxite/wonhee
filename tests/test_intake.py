"""Public Drive download fixtures exercise the transport boundary without live Google requests.

Design: docs/specs/2026-10-24-moeka-integration-design.md, Public Google Drive links.
"""
from __future__ import annotations

from collections import namedtuple

import httpx
import pytest

from wonhee import intake


@pytest.fixture(autouse=True)
def no_dns(monkeypatch):
    # Fixture HTTP transports never resolve Google; retain the adapter's Google-host allowlist.
    monkeypatch.setattr("nanobot.security.network.validate_url_target", lambda url: (True, None))


def _download(tmp_path, handler, **kwargs):
    dest = tmp_path / "download.bin"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        detail = intake.download_drive(
            "https://drive.google.com/file/d/file_id/view?resourcekey=shared-key", dest,
            max_bytes=kwargs.get("max_bytes", 1000), reserve_bytes=kwargs.get("reserve_bytes", 0), client=client,
        )
    return dest, detail


def test_drive_preserves_resource_key_and_validates_confirmation_redirect(tmp_path):
    requests = []

    def response(request):
        requests.append(request)
        if len(requests) == 1:
            assert request.url.params["resourcekey"] == "shared-key"
            return httpx.Response(200, headers={"Content-Type": "text/html"}, text=(
                '<form id="download-form" action="https://drive.usercontent.google.com/download">'
                '<input name="id" value="file_id"><input name="confirm" value="approved-token"></form>'
            ))
        if len(requests) == 2:
            assert request.url.params["resourcekey"] == "shared-key"
            assert request.url.params["confirm"] == "approved-token"
            return httpx.Response(302, headers={"Location": "https://download.googleusercontent.com/raw"})
        return httpx.Response(200, headers={"Content-Type": "video/mp4", "Content-Disposition": 'attachment; filename="take.mp4"'},
                              content=b"video-bytes")

    dest, detail = _download(tmp_path, response)
    assert dest.read_bytes() == b"video-bytes"
    assert detail == {"filename": "take.mp4", "mime": "video/mp4", "bytes": 11}
    assert len(requests) == 3


@pytest.mark.parametrize("location", ["https://evil.example/raw", "http://drive.google.com/raw", "https://127.0.0.1/raw"])
def test_drive_refuses_unsafe_redirect_before_request(tmp_path, location):
    calls = []

    def response(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": location})

    with pytest.raises(intake.DownloadError):
        _download(tmp_path, response)
    assert len(calls) == 1
    assert not (tmp_path / "download.bin").exists()


@pytest.mark.parametrize("mime,body", [
    ("text/html", b"<html>Please sign in</html>"),
    ("application/octet-stream", b"<!DOCTYPE html><html>Quota exceeded</html>"),
])
def test_drive_never_stores_html_as_a_video(tmp_path, mime, body):
    with pytest.raises(intake.DownloadError, match="HTML|sharing/login/quota"):
        _download(tmp_path, lambda request: httpx.Response(200, headers={"Content-Type": mime}, content=body))
    assert not (tmp_path / "download.bin").exists()


@pytest.mark.parametrize("code", [401, 403, 429, 500])
def test_drive_access_and_server_errors_are_actionable(tmp_path, code):
    with pytest.raises(intake.DownloadError, match="access|quota|HTTP"):
        _download(tmp_path, lambda request: httpx.Response(code))
    assert not (tmp_path / "download.bin").exists()


def test_drive_refuses_declared_oversize_before_writing(tmp_path):
    with pytest.raises(intake.DownloadError, match="size limit"):
        _download(tmp_path, lambda request: httpx.Response(200, headers={"Content-Length": "2000"}, content=b"small"),
                  max_bytes=1000)
    assert not (tmp_path / "download.bin").exists()


class _Chunks(httpx.SyncByteStream):
    def __init__(self, *, interrupted=False):
        self.interrupted = interrupted

    def __iter__(self):
        yield b"a" * (1 << 20)
        if self.interrupted:
            raise httpx.ReadError("connection lost")
        yield b"b" * (1 << 20)


def test_drive_enforces_streaming_size_limit_without_content_length(tmp_path):
    with pytest.raises(intake.DownloadError, match="size limit"):
        _download(tmp_path, lambda request: httpx.Response(200, stream=_Chunks()), max_bytes=(1 << 20) + 5)
    assert not (tmp_path / "download.bin").exists()


def test_drive_interrupted_transfer_removes_partial_file(tmp_path):
    with pytest.raises(intake.DownloadError, match="interrupted"):
        _download(tmp_path, lambda request: httpx.Response(200, stream=_Chunks(interrupted=True)), max_bytes=1 << 22)
    assert not (tmp_path / "download.bin").exists()


def test_drive_retains_disk_reserve(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(intake.shutil, "disk_usage", lambda path: usage(100, 95, 5))
    with pytest.raises(intake.DownloadError, match="free disk"):
        _download(tmp_path, lambda request: httpx.Response(200, content=b"binary"), reserve_bytes=5)
    assert not (tmp_path / "download.bin").exists()


def test_cli_reports_permanent_intake_failure_as_safe_json(settings, capsys):
    """design: live intake error correction, unsupported sources must not become opaque retryable errors."""
    import json
    from wonhee.cli import main
    source = 'https://drive.google.com/drive/unsupported/example?resourcekey=private-resource-key'
    result = main(['creative', 'inbox_add', '--data', json.dumps({'source': source})])
    out = capsys.readouterr()
    assert result == 1
    error = json.loads(out.out)['error']
    assert error['code'] == 'invalid_request'
    assert error['retryable'] is False
    assert 'folder' in error['message']
    log = (settings.home / 'logs' / 'wonhee.log').read_text()
    assert 'private-resource-key' not in log


@pytest.mark.parametrize('status,retryable', [(403, False), (429, True), (500, True)])
def test_download_error_distinguishes_permanent_and_transient_failures(tmp_path, status, retryable):
    """design: live intake error correction, sharing errors and transient server failures need different recovery."""
    with pytest.raises(intake.DownloadError) as error:
        _download(tmp_path, lambda request: httpx.Response(status))
    assert error.value.retryable is retryable
