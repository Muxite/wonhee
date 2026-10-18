"""Subject masks on the box (ONNX, CPU, about a second a photo): who or what to keep apart from the background.

Models are the public U²-Net family weights that rembg publishes (Apache-2.0). They are downloaded on first use
into <home>/models and checked by size, never committed. `person` is trained on people; `object` (IS-Net) cuts
out any salient thing (a plush on a desk, a laptop, a drink).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import httpx
import numpy as np

Kind = Literal["person", "object"]
_BASE = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"


@dataclass(frozen=True)
class Spec:
    file: str
    size: int  # input side the model was trained at
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


MODELS: dict[str, Spec] = {
    "person": Spec("u2net_human_seg.onnx", 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "object": Spec("isnet-general-use.onnx", 1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
}
_sessions: dict[str, object] = {}


def model_path(models_dir: Path, kind: str) -> Path:
    spec = MODELS[kind]
    p = models_dir / spec.file
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    models_dir.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    with httpx.stream("GET", _BASE + spec.file, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.replace(p)
    return p


def _session(models_dir: Path, kind: str):
    if kind not in _sessions:
        import onnxruntime as ort
        _sessions[kind] = ort.InferenceSession(str(model_path(models_dir, kind)), providers=["CPUExecutionProvider"])
    return _sessions[kind]


def subject(im: np.ndarray, models_dir: Path, kind: Kind = "person", feather: float = 0.004) -> np.ndarray:
    """A float32 0..1 mask the size of `im` (1 = subject), edges feathered by `feather` x the long side."""
    spec = MODELS[kind]
    h, w = im.shape[:2]
    x = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (spec.size, spec.size), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / max(float(x.max()), 1.0)
    x = (x - np.array(spec.mean, np.float32)) / np.array(spec.std, np.float32)
    sess = _session(models_dir, kind)
    out = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    m = cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, feather * max(h, w)))
    return np.clip(m, 0, 1)
