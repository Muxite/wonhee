"""Shot-level analysis of the primary trend's example videos: shots, keyframes, phashes, transcript, shot text."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

from wonhee import llm
from wonhee.analyze.frames import extract_frame
from wonhee.analyze.phash import phash_hex
from wonhee.analyze.scenes import detect_shots
from wonhee.analyze.speech import first_word_ms
from wonhee.db import queries, repo
from wonhee.db.repo import ShotAnalysis
from wonhee.media import path_for
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.scout import prompts

ANALYZER_VERSION = 1
MIN_SAMPLE = 3
MAX_VISION_FRAMES = 8


class ShotText(BaseModel):
    ord: int
    on_screen_text: str | None = None
    visual: str


class ShotTexts(BaseModel):
    shots: list[ShotText]


def _sample(n: int) -> list[int]:
    """Indices (0-based) of up to MAX_VISION_FRAMES shots, spread evenly."""
    if n <= MAX_VISION_FRAMES:
        return list(range(n))
    return sorted({round(i * (n - 1) / (MAX_VISION_FRAMES - 1)) for i in range(MAX_VISION_FRAMES)})


def analyze_video(ctx: Ctx, sha: str) -> None:
    """Shots, keyframes, phashes, transcript and shot text for one stored video; saves a media_analysis row."""
    media_dir = ctx.settings.media_dir
    shots = detect_shots(path_for(media_dir, sha))
    keyframes: list[str | None] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (start, end) in enumerate(shots):
            mid = (start + end) // 2
            dest = Path(tmp) / f"f{i}.jpg"
            try:
                extract_frame(path_for(media_dir, sha), mid, dest)
            except (RuntimeError, subprocess.CalledProcessError):  # keep the shot, without a keyframe
                ctx.flag("FRAME_FAILED", f"{sha[:12]} shot {i + 1}")
                keyframes.append(None)
                continue
            ksha = repo.save_file(ctx.conn, media_dir, dest, "image", "image/jpeg")
            repo.add_phash(ctx.conn, sha, mid, phash_hex(dest))
            keyframes.append(ksha)

    transcript = ctx.deps.transcriber.transcribe(path_for(media_dir, sha))  # RuntimeError (CUDA) propagates
    tsha = repo.save_media_bytes(
        ctx.conn, media_dir,
        json.dumps({"language": transcript.language,
                    "segments": [{"start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text,
                                  # word timings only when the transcriber gave them: keeps old JSON shape otherwise
                                  **({"words": [{"start_ms": w.start_ms, "end_ms": w.end_ms, "text": w.text}
                                                for w in s.words]} if s.words else {})}
                                 for s in transcript.segments]}).encode(),
        "transcript", "application/json")

    sampled = [i for i in _sample(len(shots)) if keyframes[i]]
    texts: dict[int, ShotText] = {}
    if sampled:
        listing = "\n".join(f"image {n + 1} = shot ord {i + 1} ({shots[i][0]}-{shots[i][1]} ms)"
                            for n, i in enumerate(sampled))
        reply = ctx.llm(ShotTexts, schema_name="ShotTexts", system=prompts.SYSTEM_SHOTS_TEXT,
                        prompt="For each attached keyframe give the on-screen text (null if none) and a short "
                               "visual description. Reply with the shot ord of each.\n" + listing,
                        images=[path_for(media_dir, keyframes[i]) for i in sampled])
        allowed = {i + 1 for i in sampled}
        for t in reply.shots:
            if t.ord in allowed:
                texts.setdefault(t.ord, t)

    rows = [ShotAnalysis(s, e, keyframes[i],
                         texts[i + 1].on_screen_text if i + 1 in texts else None,
                         texts[i + 1].visual if i + 1 in texts else None)
            for i, (s, e) in enumerate(shots)]
    repo.save_analysis(ctx.conn, sha=sha, analyzer_version=ANALYZER_VERSION, step_id=ctx.step_id,
                       first_word_ms=first_word_ms(transcript), language=transcript.language,
                       transcript_sha=tsha, shots=rows)


def run(ctx: Ctx) -> str:
    trend = queries.primary_trend(ctx.conn, ctx.project_id)
    if trend is None:
        raise StageFailed("EXCEPTION", "analyze needs a primary trend")
    done: set[str] = set()
    ready = False
    analysed = failed = 0
    for ex in queries.trend_examples(ctx.conn, trend["trend_id"]):
        sha = ex["media_sha256"]
        if not sha:
            continue
        if sha not in done:
            done.add(sha)
            if not queries.has_analysis(ctx.conn, sha, ANALYZER_VERSION):
                if not ready:  # a broken GPU setup fails the stage once, up front, not once per video
                    ensure = getattr(ctx.deps.transcriber, "ensure_ready", None)
                    if ensure:
                        ensure()
                    ready = True
                try:
                    analyze_video(ctx, sha)
                except llm.LLMError:
                    raise
                except Exception as e:  # noqa: BLE001 - one bad video must not sink the others
                    if isinstance(e, RuntimeError) and "CUDA" in str(e):
                        raise
                    ctx.flag("ANALYZE_FAILED", f"{sha[:12]}: {e}"[:300])
                    failed += 1
                    continue
        analysed += 1
    if analysed == 0:
        raise StageFailed("LOW_SAMPLE", f"all {failed} video(s) failed analysis (see ANALYZE_FAILED flags)"
                          if failed else "no videos to analyze")
    if analysed < MIN_SAMPLE:
        ctx.flag("LOW_SAMPLE", f"only {analysed} analysed example video(s)")
        return "weak"
    return "ok"
