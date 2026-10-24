"""Bound delivery previews by dimensions, duration and bytes while preserving the original separately."""
from pathlib import Path

from wonhee.fsutil import run_timed
from wonhee.procenv import clean_env


def preview(src: Path, dest: Path, *, max_seconds: int = 60, width: int = 720, max_bytes: int = 45_000_000):
    bitrate = max(128_000, min(3_000_000, int(max_bytes * 8 / max_seconds * .75) - 96_000))
    result = run_timed(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-t', str(max_seconds),
        '-map', '0:v:0', '-map', '0:a:0?', '-vf', f'scale=min({width}\\,iw):-2,setsar=1',
        '-r', '24', '-c:v', 'libx264', '-preset', 'fast', '-b:v', str(bitrate), '-maxrate', str(bitrate),
        '-bufsize', str(bitrate * 2), '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96000',
        '-movflags', '+faststart', str(dest)], env=clean_env(), capture_output=True, text=True)
    if result.returncode or not dest.is_file():
        raise RuntimeError('preview encode failed: ' + result.stderr[-300:])
    if dest.stat().st_size > max_bytes:
        dest.unlink()
        raise ValueError('preview exceeds the configured delivery size limit')
