"""Stream public Drive binary files to disk, keeping HTML and transport credentials out of the media store."""
from __future__ import annotations

import re
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

import httpx


class DownloadError(ValueError):
    pass


def drive_identity(url: str) -> tuple[str, str | None]:
    u = urlsplit(url)
    if u.scheme != 'https' or u.hostname not in {'drive.google.com', 'www.drive.google.com'} or u.username or u.password:
        raise ValueError('send a public Google Drive binary-file link (https://drive.google.com/file/d/ID/view)')
    q = parse_qs(u.query)
    m = re.fullmatch(r'/file/d/([A-Za-z0-9_-]+)(?:/[^/]*)?/?', u.path)
    file_id = m.group(1) if m else (q.get('id', [None])[0] if u.path in ('/open', '/uc') else None)
    if not file_id or not re.fullmatch(r'[A-Za-z0-9_-]+', file_id):
        raise ValueError('v1 supports individual public binary files; folders and Google documents are unsupported')
    return file_id, q.get('resourcekey', [None])[0]


class _Form(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form' and (a.get('id') == 'download-form' or 'download' in a.get('action', '')):
            self.action, self.active = a.get('action'), True
        if tag == 'input' and self.active and a.get('name'):
            self.fields[a['name']] = a.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.active = False


def _guard(url: str) -> None:
    from nanobot.security.network import validate_url_target
    u = urlsplit(url)
    host = u.hostname or ''
    if u.scheme != 'https' or not (host == 'google.com' or host.endswith('.google.com')
                                  or host == 'googleusercontent.com' or host.endswith('.googleusercontent.com')):
        raise DownloadError('Drive redirected outside the supported Google download hosts')
    ok, error = validate_url_target(url)
    if not ok:
        raise DownloadError(f'unsafe Drive download destination: {error}')


def download_drive(url: str, dest: Path, *, max_bytes: int, reserve_bytes: int,
                   client: httpx.Client | None = None) -> dict:
    file_id, resource_key = drive_identity(url)
    query = {'id': file_id, 'export': 'download', 'confirm': 't'}
    if resource_key:
        query['resourcekey'] = resource_key
    target = 'https://drive.usercontent.google.com/download?' + urlencode(query)
    owned = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(60, connect=15), follow_redirects=False)
    deadline = time.monotonic() + 1800
    try:
        for _ in range(10):
            _guard(target)
            with client.stream('GET', target) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    target = urljoin(target, response.headers.get('location', ''))
                    continue
                if response.status_code in (401, 403):
                    raise DownloadError('Drive denied access or download quota was reached; enable public downloads or retry later')
                if response.status_code >= 400:
                    raise DownloadError(f'Drive download failed (HTTP {response.status_code}); check sharing and retry')
                mime = response.headers.get('content-type', '').split(';')[0].lower()
                length = response.headers.get('content-length')
                if length and int(length) > max_bytes:
                    raise DownloadError('file exceeds the configured intake size limit')
                dest.parent.mkdir(parents=True, exist_ok=True)
                if mime in ('text/html', 'application/xhtml+xml'):
                    page = bytearray()
                    for chunk in response.iter_bytes():
                        page.extend(chunk)
                        if len(page) > 1_000_000:
                            raise DownloadError('Drive returned an HTML page instead of a downloadable file')
                    parser = _Form()
                    parser.feed(page.decode('utf-8', errors='replace'))
                    if parser.action:
                        next_url = urljoin(target, parser.action)
                        parsed = urlsplit(next_url)
                        fields = {**dict(parse_qs(parsed.query)), **parser.fields}
                        if resource_key:
                            fields['resourcekey'] = resource_key
                        target = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(fields, doseq=True), ''))
                        continue
                    raise DownloadError('Drive returned a sharing/login/quota page; send a public downloadable binary file')
                total = 0
                with dest.open('wb') as stream:
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise DownloadError('file exceeds the configured intake size limit')
                        if time.monotonic() > deadline:
                            raise DownloadError('Drive download timed out; retry the intake')
                        if shutil.disk_usage(dest.parent).free < reserve_bytes + len(chunk):
                            raise DownloadError('not enough free disk space for this download')
                        if total == len(chunk) and chunk.lstrip().lower().startswith((b'<!doctype html', b'<html')):
                            raise DownloadError('Drive returned HTML instead of a binary file')
                        stream.write(chunk)
                if not total:
                    raise DownloadError('Drive returned an empty file')
                cd = response.headers.get('content-disposition', '')
                match = re.search(r'filename="?([^";]+)', cd)
                filename = Path(match.group(1)).name if match else file_id
                return {'filename': filename, 'mime': mime or 'application/octet-stream', 'bytes': total}
        raise DownloadError('Drive exceeded the redirect/confirmation limit')
    except httpx.HTTPError as exc:
        dest.unlink(missing_ok=True)
        raise DownloadError(f'Drive transfer interrupted ({type(exc).__name__}); retry the intake') from None
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    finally:
        if owned:
            client.close()
