"""Render a plan to mp4 in passes, so cost scales with clips instead of with one giant graph.

1. clips: one small ffmpeg per clip (zoompan, caption, fx stack), run in parallel with 2 threads
   each, written as lossless segments: head (eaten by the incoming transition), body, tail.
2. transitions: tail_i + head_i+1 -> xfade segment, single-threaded (custom exprs use st()/ld()), many at once.
3. final: concat body0, tr0, body1, tr1, ... -> global fx -> audio -> x264.

Segments are cached by content hash (plan item, source file, fx seeds, frame split, code version),
so a re-render after an edit only redoes the clips that changed.
WONHEE_MONTAGE_CACHE=0 disables the cache, WONHEE_MONTAGE_CACHE_DIR moves it (default: <home>/cache/montage),
WONHEE_MONTAGE_CACHE_GB caps it (default 10).

Every ffmpeg input carries the LOCAL_ONLY protocol whitelist and runs with the minimal environment, like the rest of
wonhee: a plan names local files, and nothing in them may reach out to other files or hosts.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from wonhee.edit import text as wtext
from wonhee.fsutil import run_timed
from wonhee.media import LOCAL_ONLY
from wonhee.procenv import clean_env

from ..sfx.cues import render_track
from .effects import filter_for
from .fx import Ctx, expand, slot_seed
from .plan import to_text_spec, total_length, validate_plan
from .transitions import CUSTOM


class MontageError(RuntimeError):
    """ffmpeg failed (or dropped undecodable packets) while rendering a montage."""


# ffmpeg exits 0 after dropping packets it can't decode; the final pass treats these as failures.
DECODE_ERRORS = ("Error submitting packet to decoder", "Invalid data found when processing input")


def run_ffmpeg(cmd: list[str], strict: bool = False) -> None:
    """Run ffmpeg with wonhee's timeout and clean env; strict=True also fails on skipped decode errors."""
    proc = run_timed(cmd, capture_output=True, text=True, env=clean_env())
    bad = strict and next((ln for ln in proc.stderr.splitlines() if any(e in ln for e in DECODE_ERRORS)), None)
    if proc.returncode != 0 or bad:
        tail = bad or "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MontageError(f"ffmpeg failed ({proc.returncode}):\n{tail}")


def _sfx_track(plan: dict, dest: Path) -> Path | None:
    """Synthesized sound cues for the plan, or None (sfx off, or nothing in the plan makes a sound)."""
    return render_track(plan, dest) if plan.get("sfx", True) else None


def _slot_seed(plan: dict, i: int, k: int) -> int:
    return slot_seed(plan.get("seed", 0), i, k)


SUPERSAMPLE = 2  # zoompan on a larger source avoids sub-pixel jitter
# Lossless and fast both ways; nut keeps the 1/fps timebase. The concat demuxer decodes every
# segment with the FIRST file's codec parameters, so all segments must share one pixel format
# (xfade alone negotiates yuv444p, which then fails to decode and is silently dropped).
SEGMENT = ["-c:v", "utvideo", "-pix_fmt", "yuv420p", "-f", "nut"]
JOB_THREADS = 2


def _code_version() -> str:
    """Hash of every source file that shapes pixels: editing an fx invalidates cached segments."""
    root = Path(__file__).resolve().parent
    files = sorted(root.rglob("*.py")) + [Path(wtext.__file__)]
    h = hashlib.sha1()
    for f in files:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def _cache_dir() -> Path | None:
    if os.environ.get("WONHEE_MONTAGE_CACHE", "1") == "0":
        return None
    base = os.environ.get("WONHEE_MONTAGE_CACHE_DIR")
    if not base:
        from wonhee.config import get_settings
        base = get_settings().home / "cache" / "montage"
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prune(cache: Path, limit_gb: float) -> None:
    """Drop least-recently-used segments until the cache fits."""
    files = sorted(cache.glob("*.nut"), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    for f in files:
        if total <= limit_gb * 1e9:
            break
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _normalise(src: str, size: tuple[int, int], dest: Path) -> None:
    big = (size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE)
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    ImageOps.fit(img, big, Image.LANCZOS, centering=(0.5, 0.5)).save(dest, quality=95)


def _norm(size: tuple[int, int], fps: int, n: int) -> str:
    """Force size / fps / yuv420p and exactly n frames (segment splits depend on it)."""
    return (f"scale={size[0]}:{size[1]},setsar=1,fps={fps},format=yuv420p,"
            f"tpad=stop_mode=clone:stop_duration={n / fps:.3f},trim=end_frame={n},"
            f"setpts=PTS-STARTPTS,settb=1/{fps}")


def _stack(plan: dict, names: list[str], label: str, i: int, size, fps: int, n: int,
           tag: str, out: str) -> list[str]:
    """Filtergraph parts running fx `names` in order from `label`, then normalising to n frames."""
    parts = []
    for k, name in enumerate(names):
        ctx = Ctx(size[0], size[1], fps, n, _slot_seed(plan, i, k))
        if k:   # fx may change size (tile_grid: 640 -> 636); the next one's overlays expect w x h
            parts.append(f"[{label}]scale={size[0]}:{size[1]},setsar=1[{tag}r{k}]")
            label = f"{tag}r{k}"
        parts.append(expand(name, ctx, label, f"{tag}f{k}", f"{tag}x{k}"))
        label = f"{tag}f{k}"
    parts.append(f"[{label}]{_norm(size, fps, n)}[{out}]")
    return parts


def timeline(plan: dict) -> tuple[list[int], list[int]]:
    """Frames per clip and per transition. Transition lengths are rounded with error diffusion so
    every clip starts within half a frame of the seconds timeline the sound cues use."""
    fps, items = plan["fps"], plan["items"]
    n = [max(2, round(it["duration"] * fps)) for it in items]
    k: list[int] = []
    ideal, start = 0.0, 0
    for i, it in enumerate(items[:-1]):
        ideal += it["duration"] - it["transition"]["duration"]
        k_in = k[-1] if k else 0
        kk = start + n[i] - round(ideal * fps)
        kk = max(1, min(kk, n[i] - k_in, n[i + 1] - 1))   # body >= 0, next clip keeps a frame
        k.append(kk)
        start += n[i] - kk
    return n, k


def _clip_graph(plan: dict, i: int, n: int, has_text: bool) -> list[str]:
    """[0:v] still (+ [1:v] caption) -> [v]: motion effect, caption, fx stack, normalised.
    The caption goes under the fx (they mangle it too) unless text "layer" is "top"."""
    size, fps, it = tuple(plan["size"]), plan["fps"], plan["items"][i]
    base = f"{filter_for(it['effect'], n, fps, size)},format=yuv420p,setsar=1,settb=1/{fps}"
    fx = it.get("fx") or []
    if not has_text:
        return [f"[0:v]{base}[s]"] + _stack(plan, fx, "s", i, size, fps, n, "c", "v")
    spec, d = it["text"], n / fps
    start, end = float(spec.get("start", 0.0)), float(spec.get("end", d))
    fade = spec.get("fade")
    fade = min(0.25, max(0.05, (end - start) / 4)) if fade is None else float(fade)
    if fade > 0:
        cap = (f"[1:v]format=rgba,settb=1/{fps},fade=t=in:st={start:.3f}:d={fade:.3f}:alpha=1,"
               f"fade=t=out:st={max(start, end - fade):.3f}:d={fade:.3f}:alpha=1[t]")
        over = "overlay=format=auto:shortest=1,format=yuv420p"
    else:   # hard on/off: a caption carried across several cuts must not blink at each one
        cap = f"[1:v]format=rgba,settb=1/{fps}[t]"
        over = f"overlay=format=auto:shortest=1:enable='between(t,{start:.3f},{end:.3f})',format=yuv420p"
    if spec.get("layer") == "top":
        return ([f"[0:v]{base}[s]"] + _stack(plan, fx, "s", i, size, fps, n, "c", "f")
                + [cap, f"[f][t]{over}[v]"])
    return ([f"[0:v]{base}[b]", cap, f"[b][t]{over}[s]"]
            + _stack(plan, fx, "s", i, size, fps, n, "c", "v"))


def _ffmpeg(cmd: list[str], graph: str, tmp: Path, name: str, strict: bool = False) -> None:
    """Run with the graph in a script file (big stacks exceed the 128KB per-argument limit)."""
    script = tmp / f"{name}.graph"
    script.write_text(graph, encoding="utf-8")
    k = cmd.index("@GRAPH")
    cmd[k:k + 1] = ["-filter_complex_script", str(script)]
    run_ffmpeg(cmd, strict)


def _part(path: Path) -> str:
    """Per-process temp name, so concurrent renders sharing the cache never write the same file."""
    return f"{path}.{os.getpid()}.part"


def _render_clip(plan: dict, i: int, n: int, k_in: int, k_out: int, outs: dict[str, Path], tmp: Path) -> None:
    it, size, fps = plan["items"][i], tuple(plan["size"]), plan["fps"]
    still = tmp / f"{i:03d}.jpg"
    _normalise(it["src"], size, still)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin", "-filter_complex_threads", str(JOB_THREADS),
           *LOCAL_ONLY, "-i", str(still)]
    if it.get("text"):
        png = tmp / f"text{i:03d}.png"
        wtext.render_layer(to_text_spec(it["text"]), size).save(png)
        cmd += [*LOCAL_ONLY, "-loop", "1", "-framerate", str(fps), "-t", f"{n / fps + 1:.2f}", "-i", str(png)]
    spans = {"head": (0, k_in), "body": (k_in, n - k_out), "tail": (n - k_out, n)}
    want = [p for p in ("head", "body", "tail") if p in outs]
    parts = _clip_graph(plan, i, n, bool(it.get("text")))
    parts.append(f"[v]split={len(want)}" + "".join(f"[w{p}]" for p in want))
    for p in want:
        a, b = spans[p]
        parts.append(f"[w{p}]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[o{p}]")
    cmd.append("@GRAPH")
    for p in want:
        cmd += ["-map", f"[o{p}]", *SEGMENT, "-threads", str(JOB_THREADS), _part(outs[p])]
    _ffmpeg(cmd, ";".join(parts), tmp, f"clip{i:03d}")
    for p in want:
        os.replace(_part(outs[p]), outs[p])


def _render_transition(plan: dict, i: int, k: int, tail: Path, head: Path, out: Path, tmp: Path) -> None:
    t, fps = plan["items"][i]["transition"]["type"], plan["fps"]
    kind = f"custom:expr='{CUSTOM[t]}'" if t in CUSTOM else t
    graph = (f"[0:v][1:v]xfade=transition={kind}:duration={k / fps:.6f}:offset=0,"
             f"trim=end_frame={k},settb=1/{fps}[o]")
    # One thread: custom exprs keep st()/ld() scratch vars, which xfade's slice threads would race on.
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin", "-filter_complex_threads", "1",
           *LOCAL_ONLY, "-i", str(tail), *LOCAL_ONLY, "-i", str(head), "@GRAPH", "-map", "[o]", *SEGMENT,
           "-threads", "1", _part(out)]
    _ffmpeg(cmd, graph, tmp, f"tr{i:03d}")
    os.replace(_part(out), out)


def _segments(plan: dict, store: Path, tmp: Path, jobs: int, log: Callable[[str], None]) -> list[Path]:
    """Passes 1 and 2: render (or reuse) every segment; return the concat order."""
    items, size, fps = plan["items"], tuple(plan["size"]), plan["fps"]
    n, k = timeline(plan)
    version = _code_version()
    clip_keys, todo, queued = [], [], set()
    for i, it in enumerate(items):
        src = Path(it["src"]).stat()
        spec = {key: v for key, v in it.items() if key != "transition"}
        seeds = [_slot_seed(plan, i, s) for s in range(len(it.get("fx") or []))]
        key = _key(version, spec, src.st_size, src.st_mtime_ns, seeds, size, fps, n[i],
                   k[i - 1] if i else 0, k[i] if i < len(k) else 0)
        clip_keys.append(key)
        outs = {p: store / f"{key}.{p}.nut" for p in ("head", "body", "tail")}
        if i == 0:
            outs.pop("head")
        if i == len(items) - 1:
            outs.pop("tail")
        if n[i] - (k[i - 1] if i else 0) - (k[i] if i < len(k) else 0) <= 0:
            outs.pop("body")
        if all(p.exists() for p in outs.values()):
            for p in outs.values():
                os.utime(p)
        elif key not in queued:          # identical clips (same image, effect, split) render once
            queued.add(key)
            todo.append((i, outs))

    def run(tasks, what, workers):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(*t) for t in tasks]
            for done, f in enumerate(futures, 1):
                f.result()
                log(f"{what} {done}/{len(tasks)}")

    log(f"clips: {len(items) - len(todo)} cached, {len(todo)} to render on {jobs} jobs")
    run([(_render_clip, plan, i, n[i], k[i - 1] if i else 0, k[i] if i < len(k) else 0, outs, tmp)
         for i, outs in todo], "clips", jobs)

    order, tr_todo = [], []
    for i in range(len(items)):
        body = store / f"{clip_keys[i]}.body.nut"
        if body.exists():
            order.append(body)
        if i < len(k):
            t = items[i]["transition"]["type"]
            tr = store / f"{_key(version, clip_keys[i], clip_keys[i + 1], t, k[i])}.tr.nut"
            order.append(tr)
            if tr.exists():
                os.utime(tr)
            elif tr not in queued:
                queued.add(tr)
                tr_todo.append((_render_transition, plan, i, k[i], store / f"{clip_keys[i]}.tail.nut",
                                store / f"{clip_keys[i + 1]}.head.nut", tr, tmp))
    run(tr_todo, "transitions", jobs * JOB_THREADS)   # single-threaded jobs: twice as many
    return order


def _final_command(plan: dict, concat: Path, out: Path, audio: str | None,
                   sfx: Path | None) -> tuple[list[str], str]:
    """Pass 3: concat the segments, run global fx, mix audio, encode."""
    size, fps = tuple(plan["size"]), plan["fps"]
    length = total_length(plan)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin", *LOCAL_ONLY, "-f", "concat", "-safe", "0", "-i", str(concat)]
    audio_idx = 1
    if audio:
        cmd += [*LOCAL_ONLY, "-i", str(audio)]
    sfx_idx = audio_idx + bool(audio)
    if sfx:
        cmd += [*LOCAL_ONLY, "-i", str(sfx)]
    parts, label = [f"[0:v]settb=1/{fps},setpts=N/({fps}*TB)[cat]"], "cat"
    if plan.get("global_fx"):
        total = round(length * fps)
        parts += _stack(plan, plan["global_fx"], label, len(plan["items"]), size, fps, total, "g", "gout")
        label = "gout"
    if audio:
        fade_out = max(0.0, length - 0.8)
        duck = ",volume=0.7" if sfx else ""   # leave room for the effects on top
        parts.append(f"[{audio_idx}:a]afade=t=in:d=0.3,afade=t=out:st={fade_out:.2f}:d=0.8{duck}[am]")
    if audio and sfx:
        parts.append(f"[am][{sfx_idx}:a]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.85:level=0[aout]")
    elif sfx:
        parts.append(f"[{sfx_idx}:a]alimiter=limit=0.85:level=0[aout]")
    cmd += ["@GRAPH", "-map", f"[{label}]"]
    if audio or sfx:
        cmd += ["-map", "[aout]" if sfx else "[am]", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-t", f"{length:.3f}", "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-movflags", "+faststart", str(out)]
    return cmd, ";".join(parts)


def render(plan: dict, out: Path, audio: str | Path | None = None, jobs: int | None = None,
           log: Callable[[str], None] | None = None) -> Path:
    """Render `plan` to `out` (h264 mp4; AAC only if there is audio or sound cues). Returns `out`."""
    out = Path(out)
    validate_plan(plan)
    log = log or (lambda msg: None)
    jobs = jobs or max(1, (os.cpu_count() or 2) // JOB_THREADS)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = _cache_dir()
    with tempfile.TemporaryDirectory(prefix="wonhee-montage-") as tmp:
        tmp = Path(tmp)
        order = _segments(plan, cache or tmp, tmp, jobs, log)
        concat = tmp / "segments.txt"
        concat.write_text("".join(f"file '{p}'\n" for p in order), encoding="utf-8")
        sfx = _sfx_track(plan, tmp / "sfx.wav")
        log("final encode")
        cmd, graph = _final_command(plan, concat, out, str(audio) if audio else None, sfx)
        _ffmpeg(cmd, graph, tmp, "final", strict=True)   # a segment that fails to decode must not vanish
    if cache:
        _prune(cache, float(os.environ.get("WONHEE_MONTAGE_CACHE_GB", "10")))
    return out
