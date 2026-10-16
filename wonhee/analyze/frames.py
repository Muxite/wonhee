import subprocess
from pathlib import Path

from wonhee.procenv import clean_env


def extract_frame(path: Path, at_ms: int, dest: Path) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{at_ms / 1000:.3f}", "-i", str(path),
                    "-frames:v", "1", "-q:v", "3", str(dest)], check=True, env=clean_env())
    if not dest.exists():
        raise RuntimeError(f"ffmpeg produced no frame at {at_ms} ms (past end of {path}?)")
    return dest


def extract_every(path: Path, every_s: float, dest_dir: Path, max_side: int = 1280) -> list[tuple[int, Path]]:
    """One ffmpeg pass: a frame every `every_s` seconds, as (ms, file), ms = i * every_s * 1000."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-vf",
                    f"fps=1/{every_s},scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'",
                    "-q:v", "3", str(dest_dir / "f%05d.jpg")], check=True, env=clean_env())
    files = sorted(dest_dir.glob("f*.jpg"))
    return [(round(i * every_s * 1000), f) for i, f in enumerate(files)]
