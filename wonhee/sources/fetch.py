from __future__ import annotations

import ipaddress
import os
import subprocess
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import httpx

from wonhee.procenv import clean_env


class DownloadError(Exception):
    pass


# Grabbed media is untrusted input. Apify hands us signed Instagram CDN links, so anything else (another host, plain
# http, an IP literal, an internal name) is refused instead of fetched, and redirects are re-checked hop by hop.
CDN_SUFFIXES = ("cdninstagram.com", "fbcdn.net", "instagram.com")
MAX_VIDEO_BYTES = 300 * 1024 * 1024
MAX_IMAGE_BYTES = 40 * 1024 * 1024
MAX_REDIRECTS = 3


def check_media_url(url: str) -> None:
    """Raise DownloadError unless url is https on an Instagram CDN host."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or parts.username or parts.password:
        raise DownloadError(f"refused {parts.scheme or 'no'}:// media URL (https only, no credentials)")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise DownloadError(f"refused IP-literal media host {host}")
    if not any(host == s or host.endswith("." + s) for s in CDN_SUFFIXES):
        raise DownloadError(f"refused media host {host!r}: not an Instagram CDN")


class Downloader(Protocol):
    def download(self, video_url: str | None, page_url: str, dest: Path) -> str: ...

    def fetch(self, url: str, dest: Path) -> None: ...


class LiveDownloader:
    """Direct GET of Apify's signed videoUrl first; yt-dlp on the page URL as fallback."""

    def __init__(self, client: httpx.Client | None = None, ytdlp: str = "yt-dlp"):
        self.client = client or httpx.Client(timeout=120, follow_redirects=False)
        self.ytdlp = ytdlp

    def download(self, video_url: str | None, page_url: str, dest: Path) -> str:
        """Returns how it got the file: 'direct', 'yt-dlp', or 'direct (no audio)'.

        Instagram sometimes hands Apify a video-only DASH stream as videoUrl. When the direct file has no
        audio track, yt-dlp (which merges the separate audio stream) gets a second try; a genuinely muted
        reel stays video-only either way."""
        from wonhee.media import has_audio

        def audio(path: Path) -> bool:
            try:
                return has_audio(path)
            except RuntimeError:  # unreadable file: let yt-dlp try; the check stage reports corruption
                return False

        errors: list[str] = []
        if video_url and self._direct(video_url, dest, errors, MAX_VIDEO_BYTES):
            if audio(dest):
                return "direct"
            alt = dest.with_name(dest.stem + ".ytdlp.mp4")
            try:
                if self._ytdlp(page_url, alt, errors) and audio(alt):
                    os.replace(alt, dest)
                    return "yt-dlp"
            finally:
                alt.unlink(missing_ok=True)
            return "direct (no audio)"
        if self._ytdlp(page_url, dest, errors):
            return "yt-dlp"
        raise DownloadError("; ".join(errors))

    def fetch(self, url: str, dest: Path) -> None:
        """Plain GET of a signed media URL (photo-post images, carousel slides)."""
        errors: list[str] = []
        if not self._direct(url, dest, errors, MAX_IMAGE_BYTES):
            raise DownloadError("; ".join(errors))

    def _direct(self, video_url: str, dest: Path, errors: list[str], max_bytes: int) -> bool:
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            url = video_url
            for _ in range(MAX_REDIRECTS + 1):
                check_media_url(url)
                with self.client.stream("GET", url) as r:
                    if r.is_redirect:
                        url = urljoin(url, r.headers.get("location", ""))
                        continue
                    if r.status_code != 200:
                        raise DownloadError(f"direct: {r.status_code}")
                    size = 0
                    with open(part, "wb") as f:
                        for chunk in r.iter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise DownloadError(f"direct: larger than {max_bytes >> 20} MB, refused")
                            f.write(chunk)
                os.replace(part, dest)
                return True
            raise DownloadError("direct: too many redirects")
        except (httpx.HTTPError, DownloadError, OSError) as e:
            errors.append(str(e) if isinstance(e, DownloadError) else f"direct: {e}")
            part.unlink(missing_ok=True)
            dest.unlink(missing_ok=True)
            return False

    def _ytdlp(self, page_url: str, dest: Path, errors: list[str]) -> bool:
        try:
            # --ignore-config: no user config can add exec hooks; "--" keeps the URL from being read as an option
            p = subprocess.run([self.ytdlp, "--ignore-config", "-q", "--no-warnings", "--no-playlist",
                                "--max-filesize", f"{MAX_VIDEO_BYTES >> 20}M", "-f", "bv*+ba/b",
                                "--merge-output-format", "mp4", "-o", str(dest), "--", page_url],
                               capture_output=True, text=True, timeout=300, env=clean_env())
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"yt-dlp: {e}")
            return False
        if p.returncode == 0 and dest.exists():
            return True
        errors.append(f"yt-dlp: exit {p.returncode} {p.stderr.strip()[-200:]}")
        return False
