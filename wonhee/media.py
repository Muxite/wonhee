"""Content-addressed file store. The path is derived from the sha256; it is never stored."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, env=clean_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()[-200:]}")
    return bool(proc.stdout.strip())


def probe(path: Path) -> Probe:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, env=clean_env(),
        )
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found on PATH (install ffmpeg; see `wonhee doctor`)") from None
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffprobe failed on {path}: {tail}")
    out = proc.stdout
    info = json.loads(out)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    dur = info.get("format", {}).get("duration")
    fps = None
    if v.get("avg_frame_rate") and v["avg_frame_rate"] != "0/0":
        n, d = v["avg_frame_rate"].split("/")
        fps = float(n) / float(d) if float(d) else None
    return Probe(
        duration_ms=int(float(dur) * 1000) if dur else None,
        width=v.get("width"), height=v.get("height"), fps=fps,
    )
