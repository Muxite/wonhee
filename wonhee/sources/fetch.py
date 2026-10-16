from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Protocol

import httpx

from wonhee.procenv import clean_env


class DownloadError(Exception):
    pass


class Downloader(Protocol):
    def download(self, video_url: str | None, page_url: str, dest: Path) -> str: ...


class LiveDownloader:
    """Direct GET of Apify's signed videoUrl first; yt-dlp on the page URL as fallback."""

    def __init__(self, client: httpx.Client | None = None, ytdlp: str = "yt-dlp"):
        self.client = client or httpx.Client(timeout=120, follow_redirects=True)
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
        if video_url and self._direct(video_url, dest, errors):
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

    def _direct(self, video_url: str, dest: Path, errors: list[str]) -> bool:
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            with self.client.stream("GET", video_url) as r:
                if r.status_code != 200:
                    raise DownloadError(f"direct: {r.status_code}")
                with open(part, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            os.replace(part, dest)
            return True
        except (httpx.HTTPError, DownloadError, OSError) as e:
            errors.append(str(e) if isinstance(e, DownloadError) else f"direct: {e}")
            part.unlink(missing_ok=True)
            dest.unlink(missing_ok=True)
            return False

    def _ytdlp(self, page_url: str, dest: Path, errors: list[str]) -> bool:
        try:
            p = subprocess.run([self.ytdlp, "-q", "--no-warnings", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
                                "-o", str(dest), page_url],
                               capture_output=True, text=True, timeout=300, env=clean_env())
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"yt-dlp: {e}")
            return False
        if p.returncode == 0 and dest.exists():
            return True
        errors.append(f"yt-dlp: exit {p.returncode} {p.stderr.strip()[-200:]}")
        return False
