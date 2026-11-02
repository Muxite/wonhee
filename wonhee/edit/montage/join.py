"""Two whole videos, one named transition: `join`, and one named fx on a whole video: `apply_fx`.

The montage renderer builds its own segments from stills; these are the video-in counterparts the `join_videos` and
`video_fx` nodes use. Same ffmpeg conventions as render.py (wonhee's timeout and minimal env, LOCAL_ONLY on every
input), same transition names (`transitions.ALL`, built-in or custom expression). Plain functions on files: no DB.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from wonhee import media
from wonhee.fsutil import run_timed
from wonhee.media import LOCAL_ONLY
from wonhee.procenv import clean_env

from .transitions import ALL, CUSTOM

AUDIO_RATE = 48000
# bitexact: no encoder-version or timestamp tags, so the same inputs give the same bytes.
ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
          "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact"]


class JoinError(RuntimeError):
    """ffmpeg failed while joining clips or applying an fx."""


def _run(cmd: list[str], graph: str, tmp: Path) -> None:
    script = tmp / "graph.txt"   # long custom expressions can exceed the per-argument limit
    script.write_text(graph, encoding="utf-8")
    k = cmd.index("@GRAPH")
    cmd[k:k + 1] = ["-filter_complex_script", str(script)]
    proc = run_timed(cmd, capture_output=True, text=True, env=clean_env())
    if proc.returncode != 0:
        raise JoinError("ffmpeg failed:\n" + "\n".join(proc.stderr.strip().splitlines()[-12:]))


def _even(n: int) -> int:
    return max(2, n - n % 2)


def _info(path: Path) -> tuple[int, int, int, float]:
    """(width, height, fps, seconds) of a video, or ValueError."""
    p = media.probe(Path(path))
    if not p.width or not p.height or not p.duration_ms:
        raise ValueError(f"{path}: not a readable video (no size or duration)")
    return p.width, p.height, max(1, round(p.fps or 30)), p.duration_ms / 1000


def _norm(size: tuple[int, int], fps: int) -> str:
    return f"scale={size[0]}:{size[1]},setsar=1,fps={fps},format=yuv420p,settb=1/{fps},setpts=PTS-STARTPTS"


def join(a_path, b_path, out_path, transition: str, duration_s: float = 0.5, seed: int = 0) -> Path:
    """Crossfade video A into video B with `transition`; the result is len(A) + len(B) - duration_s long.

    Size, fps and pixel format follow A (even sizes). Audio is crossfaded when both have it; a clip without audio
    gets silence so the other's sound is kept; with none on either side the output has no audio. `seed` is accepted
    for the node's contract; every transition here is a pure function of the pixels, so it changes nothing.
    """
    if transition not in ALL:
        raise ValueError(f"unknown transition {transition!r}; choose from {ALL}")
    if not duration_s > 0:
        raise ValueError("transition duration must be > 0")
    a_path, b_path, out_path = Path(a_path), Path(b_path), Path(out_path)
    w, h, fps, a_len = _info(a_path)
    _, _, _, b_len = _info(b_path)
    d = round(duration_s * fps) / fps   # a whole number of frames
    d = max(d, 1 / fps)
    if d >= min(a_len, b_len):
        raise ValueError(f"transition {duration_s}s is not shorter than the clips ({a_len:.2f}s, {b_len:.2f}s)")
    size = (_even(w), _even(h))
    kind = f"custom:expr='{CUSTOM[transition]}'" if transition in CUSTOM else transition
    parts = [f"[0:v]{_norm(size, fps)}[va]", f"[1:v]{_norm(size, fps)}[vb]",
             f"[va][vb]xfade=transition={kind}:duration={d:.6f}:offset={a_len - d:.6f},"
             f"format=yuv420p,settb=1/{fps}[v]"]
    has = [media.has_audio(a_path), media.has_audio(b_path)]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
           # one thread: custom exprs keep st()/ld() scratch variables, which xfade's slice threads would race on
           "-filter_complex_threads", "1", *LOCAL_ONLY, "-i", str(a_path), *LOCAL_ONLY, "-i", str(b_path),
           "@GRAPH", "-map", "[v]"]
    if any(has):
        fmt = f"aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo"
        for k, (src, length) in enumerate(((0, a_len), (1, b_len))):
            if has[k]:
                parts.append(f"[{src}:a]{fmt},asetpts=PTS-STARTPTS[a{k}]")
            else:
                parts.append(f"anullsrc=r={AUDIO_RATE}:cl=stereo,atrim=duration={length:.6f},"
                             f"asetpts=PTS-STARTPTS[a{k}]")
        parts.append(f"[a0][a1]acrossfade=d={d:.6f}:c1=tri:c2=tri[aout]")
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-t", f"{a_len + b_len - d:.6f}", "-r", str(fps), *ENCODE, str(out_path)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wonhee-join-") as tmp:
        _run(cmd, ";".join(parts), Path(tmp))
    return out_path


def apply_fx(src, dst, name: str, seed: int = 0, intensity: float = 1.0) -> Path:
    """Run montage fx `name` over the whole video `src` -> `dst`; size, fps and frame count are kept.

    The fx is built for the video's own size, fps and exact frame count (rounded from duration), then the result is
    forced back to that size / fps / yuv420p / frame count, like render.py does after an fx stack. Audio is copied.
    """
    from .fx import Ctx, expand, get, slot_seed
    get(name)   # unknown name -> ValueError listing the known ones, before any ffmpeg
    if not 0.5 <= intensity <= 2:
        raise ValueError("fx intensity must be between 0.5 and 2")
    src, dst = Path(src), Path(dst)
    w, h, fps, length = _info(src)
    size = (_even(w), _even(h))
    n = max(2, round(length * fps))
    ctx = Ctx(size[0], size[1], fps, n, slot_seed(seed, 0, 0), intensity)
    norm = (f"scale={size[0]}:{size[1]},setsar=1,fps={fps},format=yuv420p,"
            f"tpad=stop_mode=clone:stop_duration={n / fps:.3f},trim=end_frame={n},"
            f"setpts=PTS-STARTPTS,settb=1/{fps}")
    parts = [f"[0:v]scale={size[0]}:{size[1]},setsar=1,fps={fps},format=yuv420p,settb=1/{fps}[s]",
             expand(name, ctx, "s", "f", "x"), f"[f]{norm}[v]"]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-filter_complex_threads", "1",
           *LOCAL_ONLY, "-i", str(src), "@GRAPH", "-map", "[v]", "-map", "0:a?", "-c:a", "copy",
           "-r", str(fps), *ENCODE, str(dst)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wonhee-fx-") as tmp:
        _run(cmd, ";".join(parts), Path(tmp))
    return dst
