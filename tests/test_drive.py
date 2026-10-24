"""Design: moeka integration §Public Google Drive links, recursive folder extension."""
from __future__ import annotations

import json
from pathlib import Path
from html import escape

import httpx
import pytest

from wonhee import drive, creative, service


ROOT = 'https://drive.google.com/drive/folders/root?resourcekey=root-key'


@pytest.fixture(autouse=True)
def no_dns(monkeypatch):
    monkeypatch.setattr('nanobot.security.network.validate_url_target', lambda url: (True, None))


def listing(*entries):
    return '<title>Takes</title><div id="flip-contents">' + ''.join(
        f'<div class="flip-entry"><a href="{escape(url)}"><div>{escape(name)}</div></a></div>'
        for url, name in entries) + '</div>'


def binary(name='take.mp4', body=b'video'):
    return httpx.Response(200, headers={'content-type': 'video/mp4',
                                      'content-disposition': f'attachment; filename="{name}"'}, content=body)


def test_nested_folder_preserves_paths_keys_and_bytes(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == '/embeddedfolderview':
            if request.url.params['id'] == 'root':
                assert request.url.params['resourcekey'] == 'root-key'
                return httpx.Response(200, text=listing(
                    ('https://drive.google.com/drive/folders/child?resourcekey=child-key', 'Day 1'),
                    ('https://drive.google.com/file/d/one/view', 'one.mp4')))
            assert request.url.params['resourcekey'] == 'child-key'
            return httpx.Response(200, text=listing(
                ('https://drive.google.com/file/d/two/view?resourcekey=file-key', 'two.mp4')))
        if request.url.params['id'] == 'two':
            assert request.url.params['resourcekey'] == 'file-key'
        return binary()

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = drive.download(ROOT, tmp_path / 'takes', max_bytes=100, reserve_bytes=0, client=client)
    assert [f['relative_path'] for f in result['files']] == ['Day 1/two.mp4', 'one.mp4']
    assert (tmp_path / 'takes/Day 1/two.mp4').read_bytes() == b'video'
    assert result['bytes'] == 10
    assert len(calls) == 4


@pytest.mark.parametrize('page', [
    '<title>Google Drive</title><html>Sign in</html>',
    listing(('https://docs.google.com/document/d/native/edit', 'Document')),
    listing(('https://drive.google.com/file/d/one/view', '../escape')),
    listing(('https://drive.google.com/file/d/one/view', 'same'),
            ('https://drive.google.com/file/d/two/view', 'same')),
    listing(('https://drive.google.com/drive/folders/root', 'cycle')),
    listing(('https://example.com/shortcut', 'shortcut')),
    listing() + '<a rel="next" href="?pageToken=more">Next</a>',
])
def test_invalid_listing_never_downloads_files(tmp_path, page):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=page)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(drive.DownloadError):
            drive.download(ROOT, tmp_path / 'takes', max_bytes=100, reserve_bytes=0, client=client)
    assert all(r.url.path == '/embeddedfolderview' for r in calls)
    assert not (tmp_path / 'takes').exists()


def test_existing_file_collision_is_detected_before_binary_transfer(tmp_path):
    (tmp_path / 'one.mp4').write_bytes(b'original')
    def handler(request):
        assert request.url.path == '/embeddedfolderview'
        return httpx.Response(200, text=listing(('https://drive.google.com/file/d/one/view', 'one.mp4')))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='existing file'):
            drive.download(ROOT, tmp_path, max_bytes=100, reserve_bytes=0, client=client)
    assert (tmp_path / 'one.mp4').read_bytes() == b'original'


def test_total_transfer_limit_cleans_partial_but_retains_completed_files(tmp_path):
    def handler(request):
        if request.url.path == '/embeddedfolderview':
            return httpx.Response(200, text=listing(
                ('https://drive.google.com/file/d/one/view', 'one.mp4'),
                ('https://drive.google.com/file/d/two/view', 'two.mp4')))
        return binary()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(drive.DownloadError, match='size limit'):
            drive.download(ROOT, tmp_path, max_bytes=7, reserve_bytes=0, client=client)
    assert [p.name for p in tmp_path.iterdir()] == ['one.mp4']


def test_single_file_into_directory(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda request: binary())) as client:
        result = drive.download('https://drive.google.com/open?id=one', tmp_path,
                                max_bytes=100, reserve_bytes=0, client=client)
    assert result['files'][0]['relative_path'] == 'take.mp4'
    assert (tmp_path / 'take.mp4').read_bytes() == b'video'


def test_symlink_escape_refused(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    dest = tmp_path / 'takes'
    dest.mkdir()
    (dest / 'sub').symlink_to(outside, target_is_directory=True)
    def handler(request):
        if request.url.params['id'] == 'root':
            return httpx.Response(200, text=listing(('https://drive.google.com/drive/folders/child', 'sub')))
        return httpx.Response(200, text=listing(('https://drive.google.com/file/d/one/view', 'one.mp4')))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='unsafe path'):
            drive.download(ROOT, dest, max_bytes=100, reserve_bytes=0, client=client)
    assert not list(outside.iterdir())


def test_folder_node_manifest_and_intake_replay(conn, settings, tiny_video, monkeypatch):
    calls = []
    def download(url, dest, **kwargs):
        calls.append(url)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / 'take.mp4'
        path.write_bytes(tiny_video.read_bytes())
        return {'destination': str(dest), 'bytes': path.stat().st_size, 'files': [
            {'id': 'one', 'relative_path': 'take.mp4', 'path': str(path), 'filename': 'take.mp4',
             'mime': 'video/mp4', 'bytes': path.stat().st_size}]}
    monkeypatch.setattr(drive, 'download', download)
    project = creative.project_create(conn, settings, name='Takes')
    kwargs = dict(source=ROOT, project_id=project['id'], source_key='telegram:folder:1')
    first = creative.inbox_add(conn, settings, **kwargs)
    second = creative.inbox_add(conn, settings, **kwargs)
    assert first == second
    assert len(calls) == 1
    assert len(first['assets']) == 1
    asset = first['assets'][0]
    assert asset['metadata']['relative_path'] == 'take.mp4'
    assert asset['project_id'] == project['id']
    assert asset['ref']
    manifest = service._step_out(conn, first['run_id'], 'drive_download')['manifest']
    from wonhee import media
    assert json.loads(media.path_for(settings.media_dir, manifest).read_text())['files'][0]['ref'] == asset['ref']
    with pytest.raises(ValueError, match='another source'):
        creative.inbox_add(conn, settings, **{**kwargs, 'source': ROOT.replace('root?', 'other?')})


def test_mcp_download_paths_are_guarded(settings):
    from wonhee import mcp_server
    outside = '/tmp/wonhee-drive-outside'
    assert 'output paths' in mcp_server.drive_download(ROOT, outside)
    assert 'output paths' in mcp_server.node_run('drive_download', {}, {'url': ROOT, 'dest': outside})
    assert 'output paths' in mcp_server.chain_run([
        {'id': 'd', 'node': 'drive_download', 'params': {'url': ROOT, 'dest': outside}}])


def test_cli_download_uses_shared_service(settings, monkeypatch, capsys):
    from wonhee.cli import main
    calls = []
    def download(conn, config, deps, url, dest):
        calls.append((url, dest))
        return {'destination': str(dest), 'files': []}
    monkeypatch.setattr(service, 'drive_download', download)
    assert main(['download', ROOT, str(settings.home / 'takes')]) == 0
    assert calls == [(ROOT, settings.home / 'takes')]
    assert json.loads(capsys.readouterr().out)['files'] == []


def test_empty_subfolder_is_preserved(tmp_path):
    def handler(request):
        page = listing(('https://drive.google.com/drive/folders/child', 'empty')) if request.url.params['id'] == 'root' else listing()
        return httpx.Response(200, text=page)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = drive.download(ROOT, tmp_path, max_bytes=100, reserve_bytes=0, client=client)
    assert result['files'] == []
    assert (tmp_path / 'empty').is_dir()


def test_blocked_parent_preflight_prevents_all_binary_transfers(tmp_path):
    (tmp_path / 'blocked').write_text('existing file')
    def handler(request):
        assert request.url.path == '/embeddedfolderview'
        if request.url.params['id'] == 'root':
            page = listing(('https://drive.google.com/file/d/one/view', 'one.mp4'),
                           ('https://drive.google.com/drive/folders/child', 'blocked'))
        else:
            page = listing(('https://drive.google.com/file/d/two/view', 'two.mp4'))
        return httpx.Response(200, text=page)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='existing file'):
            drive.download(ROOT, tmp_path, max_bytes=100, reserve_bytes=0, client=client)
    assert not (tmp_path / 'one.mp4').exists()


@pytest.mark.parametrize('retryable', [True, False])
def test_folder_failure_retains_retryability_through_node_service(conn, settings, monkeypatch, retryable):
    from tests.fakes import fake_deps
    def fail(*args, **kwargs):
        raise drive.DownloadError('Drive unavailable', retryable=retryable)
    monkeypatch.setattr(drive, 'download', fail)
    with pytest.raises(drive.DownloadError) as error:
        service.drive_download(conn, settings, fake_deps(), ROOT, settings.home / 'takes')
    assert error.value.retryable is retryable


def test_folder_key_cannot_reuse_file_receipt(conn, settings, tiny_video):
    creative.inbox_add(conn, settings, source=str(tiny_video), source_key='same-key')
    with pytest.raises(ValueError, match='another source'):
        creative.inbox_add(conn, settings, source=ROOT, source_key='same-key')


def test_inbox_pages_folder_children(conn, settings, tiny_video):
    for i in range(3):
        creative.inbox_add(conn, settings, source=str(tiny_video), source_key=f'take:{i}')
    first = creative.inbox_list(conn, settings, limit=2)['assets']
    second = creative.inbox_list(conn, settings, limit=2, offset=2)['assets']
    assert len(first) == 2 and len(second) == 1
    assert not {a['id'] for a in first} & {a['id'] for a in second}
    with pytest.raises(ValueError, match='limit'):
        creative.inbox_list(conn, settings, limit=1001)


def test_store_copy_respects_disk_reserve(conn, settings, tmp_path, monkeypatch):
    from collections import namedtuple
    from wonhee.nodes import drive as node
    from tests.fakes import fake_deps
    path = tmp_path / 'take.bin'
    path.write_bytes(b'binary')
    monkeypatch.setattr(drive, 'download', lambda *args, **kwargs: {'destination': str(tmp_path), 'files': [
        {'path': str(path), 'bytes': 6, 'mime': 'application/octet-stream'}]})
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(node.shutil, 'disk_usage', lambda path: usage(100, 95, 5))
    with pytest.raises(drive.DownloadError, match='free disk'):
        service.drive_download(conn, settings, fake_deps(), ROOT, settings.home / 'takes')
