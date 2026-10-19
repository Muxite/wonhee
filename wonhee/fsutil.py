"""Small filesystem helpers: atomic writes and never-overwrite directories."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic_bytes(path: Path | str, data: bytes) -> Path:
    """Write via a temp file in the same directory, then rename over the target: readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def write_atomic(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    return write_atomic_bytes(path, text.encode(encoding))


def unique_dir(parent: Path | str, name: str) -> Path:
    """Create and return parent/name, or parent/name-2, -3, ... when that exists and is not empty."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        d = parent / (name if n == 1 else f"{name}-{n}")
        try:
            d.mkdir()
            return d
        except FileExistsError:
            if d.is_dir() and not any(d.iterdir()):
                return d
            n += 1


def run_timed(argv: list[str], *, timeout: float | None = None, **kw):
    """subprocess.run with a timeout (default: the ffmpeg_timeout_s setting); a hang becomes a clear RuntimeError."""
    import subprocess

    if timeout is None:
        from wonhee.config import get_settings
        timeout = get_settings().ffmpeg_timeout_s
    try:
        return subprocess.run(argv, timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{argv[0]} timed out after {timeout:.0f}s (raise WONHEE_FFMPEG_TIMEOUT_S)") from None
    except FileNotFoundError:
        raise RuntimeError(f"{argv[0]} not found on PATH (install ffmpeg; see `wonhee doctor`)") from None
