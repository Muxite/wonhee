"""Content-addressed file store. The path is derived from the sha256; it is never stored."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from wonhee.fsutil import run_timed
from wonhee.procenv import clean_env


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path_for(media_dir: Path, sha: str) -> Path:
    return Path(media_dir) / sha[:2] / sha


def _place(media_dir: Path, sha: str, write) -> str:
    dest = path_for(media_dir, sha)
    if dest.exists():
        return sha
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent)
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return sha


def store_bytes(media_dir: Path, data: bytes) -> str:
    return _place(media_dir, sha256_bytes(data), lambda p: p.write_bytes(data))


def store_file(media_dir: Path, src: Path) -> str:
    return _place(media_dir, sha256_file(src), lambda p: shutil.copyfile(src, p))


@dataclass(frozen=True)
class Probe:
    duration_ms: int | None
    width: int | None
    height: int | None
    fps: float | None


def has_audio(path: Path) -> bool:
    """True if the file has at least one audio stream (IG strips muted-song audio entirely)."""
    proc = run_timed(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, env=clean_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()[-200:]}")
    return bool(proc.stdout.strip())


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def probe(path: Path) -> Probe:
    proc = run_timed(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, env=clean_env(),
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffprobe failed on {path}: {tail}")
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"ffprobe gave unreadable output for {path}") from None
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    dur = _num(info.get("format", {}).get("duration"))
    if dur is None:
        dur = _num(v.get("duration"))
    fps = None
    rate = str(v.get("avg_frame_rate") or "")
    n, _, d = rate.partition("/")
    n, d = _num(n), _num(d) if d else 1.0
    if n and d:
        fps = n / d
    return Probe(
        duration_ms=int(dur * 1000) if dur else None,
        width=v.get("width"), height=v.get("height"), fps=fps,
    )
