"""Public Drive folder discovery and bounded downloads without browser state or an API key."""
from __future__ import annotations

import os
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import httpx

from wonhee.intake import DownloadError, _guard, download_drive, drive_identity


_FOLDER = re.compile(r'/drive/(?:u/\d+/)?folders/([A-Za-z0-9_-]+)/?')


def folder_identity(url: str) -> tuple[str, str | None] | None:
    u = urlsplit(url)
    if u.scheme != 'https' or u.hostname not in {'drive.google.com', 'www.drive.google.com'} or u.username or u.password:
        return None
    match = _FOLDER.fullmatch(u.path)
    return (match[1], parse_qs(u.query).get('resourcekey', [None])[0]) if match else None


def _name(value: str) -> str:
    if not value or value in {'.', '..'} or any(c in value for c in '/\\') or any(ord(c) < 32 for c in value):
        raise DownloadError('Drive contains an unsafe or empty filename', retryable=False)
    return value


class _Listing(HTMLParser):
    def __init__(self):
        super().__init__()
        self.entries = []
        self.href = None
        self.label = []
        self.title = False
        self.title_text = []
        self.paginated = False
        self.listing = False
        self.entry_count = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get('id') == 'flip-contents':
            self.listing = True
        if 'flip-entry' in a.get('class', '').split():
            self.entry_count += 1
        if tag == 'title':
            self.title = True
        if tag == 'a':
            self.href, self.label = a.get('href', ''), []
            if 'next' in a.get('rel', '').split() or 'pageToken=' in self.href:
                self.paginated = True

    def handle_data(self, data):
        if self.title:
            self.title_text.append(data)
        if self.href is not None:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == 'title':
            self.title = False
        if tag == 'a' and self.href is not None:
            href = urljoin('https://drive.google.com/', self.href)
            u = urlsplit(href)
            if u.hostname == 'docs.google.com':
                raise DownloadError('folder contains unsupported native Google documents', retryable=False)
            folder = folder_identity(href)
            try:
                file = drive_identity(href)
            except ValueError:
                file = None
            if folder or file:
                self.entries.append({'id': (folder or file)[0], 'name': _name(''.join(self.label).strip()),
                                     'folder': folder is not None, 'url': href})
            self.href = None


def list_folder(url: str, client: httpx.Client) -> list[dict]:
    """Fail closed when a listing cannot be interpreted; never return a login page as an empty folder."""
    result, seen = [], set()
    count = 0

    def walk(link, relative, depth):
        nonlocal count
        identity = folder_identity(link)
        if identity is None or depth > 32 or identity[0] in seen:
            raise DownloadError('Drive folder cycle or nesting limit reached', retryable=False)
        seen.add(identity[0])
        query = {'id': identity[0]}
        if identity[1]:
            query['resourcekey'] = identity[1]
        target = 'https://drive.google.com/embeddedfolderview?' + urlencode(query)
        for _ in range(10):
            _guard(target)
            with client.stream('GET', target) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    target = urljoin(target, response.headers.get('location', ''))
                    continue
                if response.status_code != 200:
                    raise DownloadError('cannot list Drive folder; enable Anyone with the link and check the quota',
                                        retryable=response.status_code == 429 or response.status_code >= 500)
                page = bytearray()
                for chunk in response.iter_bytes():
                    page.extend(chunk)
                    if len(page) > 8 * 1024 * 1024:
                        raise DownloadError('Drive folder listing exceeds the size limit', retryable=False)
                parser = _Listing()
                parser.feed(page.decode('utf-8', errors='replace'))
                title = ''.join(parser.title_text).strip().lower()
                if not title or not parser.listing:
                    raise DownloadError('cannot list Drive folder; enable Anyone with the link', retryable=False)
                if parser.entry_count != len(parser.entries):
                    raise DownloadError('Drive folder contains unsupported entries or its listing format changed', retryable=False)
                if parser.paginated:
                    raise DownloadError('Drive folder listing is paginated; split it into smaller folders', retryable=False)
                names = set()
                for entry in parser.entries:
                    count += 1
                    if count > 1000:
                        raise DownloadError('Drive folder exceeds the 1000-entry limit', retryable=False)
                    if entry['name'] in names:
                        raise DownloadError('Drive folder has duplicate filenames', retryable=False)
                    names.add(entry['name'])
                    path = relative / entry['name']
                    if entry['folder']:
                        result.append({**entry, 'relative_path': str(path)})
                        walk(entry['url'], path, depth + 1)
                    else:
                        result.append({**entry, 'relative_path': str(path)})
                return
        raise DownloadError('Drive folder exceeded the redirect limit', retryable=False)

    walk(url, Path(), 0)
    return result


def download(url: str, dest: Path, *, max_bytes: int, reserve_bytes: int,
             client: httpx.Client | None = None) -> dict:
    """Download into a folder, publishing each complete file exclusively; preserve earlier completed files on failure."""
    folder = folder_identity(url)
    if folder is None:
        drive_identity(url)
    dest = dest.expanduser().resolve()
    if dest.exists() and not dest.is_dir():
        raise ValueError('destination must be a folder')
    owned = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(60, connect=15), follow_redirects=False)
    try:
        entries = list_folder(url, client) if folder else [{'id': drive_identity(url)[0], 'url': url}]
        for entry in entries:
            if 'relative_path' in entry:
                output = dest / entry['relative_path']
                blocked = any(p.exists() and not p.is_dir() for p in output.parents if p != dest.parent)
                collision = output.exists() and not (entry['folder'] and output.is_dir())
                if not output.resolve().is_relative_to(dest) or collision or blocked or output.is_symlink():
                    raise ValueError('download destination contains an existing file or unsafe path')
        dest.mkdir(parents=True, exist_ok=True)
        files, total = [], 0
        for entry in entries:
            if entry.get('folder'):
                (dest / entry['relative_path']).mkdir(parents=True, exist_ok=True)
                continue
            with tempfile.TemporaryDirectory(prefix='.drive-', dir=dest) as td:
                temp = Path(td) / 'download'
                detail = download_drive(entry['url'], temp, max_bytes=max_bytes - total,
                                        reserve_bytes=reserve_bytes, client=client)
                relative = entry.get('relative_path') or _name(detail['filename'])
                output = dest / relative
                if not output.resolve().is_relative_to(dest):
                    raise ValueError('download destination escapes the destination folder')
                output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # Same-filesystem hard linking publishes complete bytes and never replaces another file.
                    os.link(temp, output)
                except FileExistsError:
                    raise ValueError('download destination already exists; choose another folder') from None
                total += detail['bytes']
                files.append({'id': entry['id'], 'relative_path': relative, 'path': str(output), **detail})
        return {'destination': str(dest), 'files': files, 'bytes': total}
    except httpx.HTTPError as exc:
        raise DownloadError(f'Drive folder transfer interrupted ({type(exc).__name__}); retry') from None
    finally:
        if owned:
            client.close()
