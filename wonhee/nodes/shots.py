"""shot_index: cut a video into shots, embed one keyframe per shot with SigLIP2, tag it, and store the vectors so
`wonhee shots search "bus at sunset"` finds shots across all indexed footage (service.shots_search)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wonhee import marks
from wonhee.analyze import embed as EMBED  # tests swap this for a fake with the same three functions
from wonhee.analyze.frames import extract_frame
from wonhee.analyze.scenes import detect_shots
from wonhee.db import repo
from wonhee.media import path_for
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import save_text

# Zero-shot tags: short phrases scored against each keyframe. Setting, light and subject are what the planner asks
# for when picking b-roll; add phrases here (it changes the node's output, so bump the node version).
TAGS = ["street", "cafe", "restaurant", "kitchen", "bedroom", "office", "subway", "bus", "car interior", "park",
        "beach", "mountains", "rooftop", "shop", "crowd", "night", "sunset", "daytime", "neon lights", "rain",
        "person talking to camera", "hands close-up", "food close-up", "text on screen", "screen recording",
        "selfie", "mirror", "walking", "dancing", "pet"]
_tag_vecs: dict[str, np.ndarray] = {}


class ShotParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tags: int = Field(default=3, ge=0, le=10)  # top tags kept per shot
    batch: int = Field(default=16, ge=1, le=128)
    keyframe_px: int = Field(default=640, ge=128, le=1920)  # stored keyframe long side


def _keyframes(src: Path, shots: list[tuple[int, int]], tmp: Path, px: int) -> list[Path]:
    out = []
    for i, (s, e) in enumerate(shots):
        f = extract_frame(src, (s + e) // 2, tmp / f"k{i:04d}.jpg")
        im = cv2.imread(str(f))
        k = px / max(im.shape[:2])
        if k < 1:
            cv2.imwrite(str(f), cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(f)
    return out


def tag_vectors() -> np.ndarray:
    key = EMBED.MODEL_ID
    if key not in _tag_vecs:
        _tag_vecs[key] = EMBED.text_vecs([f"a photo of {t}" for t in TAGS])
    return _tag_vecs[key]


def shot_index_fn(ctx, inputs, p: ShotParams) -> dict[str, str]:
    if not EMBED.available():
        raise ValueError("shot_index needs the gpu dependency group (uv sync --group gpu)")
    d = ctx.settings.media_dir
    sha = inputs["video"]
    src = path_for(d, sha).resolve()
    shots = detect_shots(src)
    with tempfile.TemporaryDirectory(prefix="wonhee-shots-") as tmp:
        frames = _keyframes(src, shots, Path(tmp), p.keyframe_px)
        marks.mark("compute_started")
        vecs = EMBED.image_vecs([cv2.imread(str(f)) for f in frames], batch=p.batch)
        scores = vecs.astype(np.float32) @ tag_vectors().astype(np.float32).T if p.tags else None
        marks.mark("compute_finished")
        keys = [repo.save_file(ctx.conn, d, f, "image", "image/jpeg") for f in frames]
    rows = []
    for i, ((s, e), k) in enumerate(zip(shots, keys)):
        top = [] if scores is None else [[TAGS[j], round(float(scores[i, j]), 4)]
                                         for j in np.argsort(-scores[i])[:p.tags]]
        rows.append({"ord": i, "start_ms": s, "end_ms": e, "keyframe": k, "tags": top})
        ctx.conn.execute("INSERT OR REPLACE INTO shot_embedding(media_sha256, model, ord, start_ms, end_ms,"
                         " keyframe_sha256, tags, vec) VALUES (?,?,?,?,?,?,?,?)",
                         (sha, EMBED.MODEL_ID, i, s, e, k, json.dumps(top), vecs[i].astype("<f2").tobytes()))
    manifest = {"video": sha, "model": EMBED.MODEL_ID, "shots": rows}
    return {"index": save_text(ctx.conn, d, json.dumps(manifest, indent=1))}


registry.register(Node("shot_index", 1, {"video": "video"}, {"index": "text"}, ShotParams, shot_index_fn,
                       "Cut a video into shots, embed each keyframe (SigLIP2) and tag it, and store the vectors so "
                       "`shots search` finds shots by description across all footage.",
                       salt=lambda s: {"model": EMBED.MODEL_ID, "tags": len(TAGS)}, lane="cuda"))
