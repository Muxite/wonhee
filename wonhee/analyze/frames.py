import subprocess
from pathlib import Path

from wonhee.procenv import clean_env


def extract_frame(path: Path, at_ms: int, dest: Path) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{at_ms / 1000:.3f}", "-i", str(path),
                    "-frames:v", "1", "-q:v", "3", str(dest)], check=True, env=clean_env())
    if not dest.exists():
        raise RuntimeError(f"ffmpeg produced no frame at {at_ms} ms (past end of {path}?)")
    return dest
