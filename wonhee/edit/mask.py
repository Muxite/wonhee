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


# --- GPU: a mask from a text prompt ("person", "plush toy", "laptop") ---------------------------------------------
# Grounding-DINO finds boxes for the words, SAM 2.1 turns each box into an exact mask. Both are Apache-2.0 and loaded
# through `transformers` (pure PyTorch, so no custom CUDA kernels to build on Blackwell). Needs the `gpu` dependency
# group (`uv sync --group gpu`); without torch, `gpu_available()` is False and callers use the ONNX models above.
DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam2.1-hiera-base-plus"
_gpu: dict[str, object] = {}


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _pretrained(cls, model_id: str):
    try:
        return cls.from_pretrained(model_id, local_files_only=True)  # cached: no network round trip
    except OSError:
        return cls.from_pretrained(model_id)


def _gpu_models():
    if not _gpu:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        _gpu["torch"] = torch
        _gpu["dp"] = _pretrained(AutoProcessor, DINO_ID)
        _gpu["dm"] = _pretrained(AutoModelForZeroShotObjectDetection, DINO_ID).to("cuda").eval()
        _gpu["sp"] = _pretrained(Sam2Processor, SAM_ID)
        _gpu["sm"] = _pretrained(Sam2Model, SAM_ID).to("cuda").eval()
    return _gpu


def boxes_for(im: np.ndarray, prompt: str, threshold: float = 0.3, top: int = 3) -> list[tuple[list[float], float]]:
    """Grounding-DINO boxes (pixel xyxy, score) for a text prompt, best first."""
    from PIL import Image
    g = _gpu_models()
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    text = prompt.strip().lower().rstrip(".") + "."  # Grounding-DINO wants lower-case phrases ending in a dot
    inputs = g["dp"](images=pil, text=text, return_tensors="pt").to("cuda")
    with g["torch"].no_grad():
        out = g["dm"](**inputs)
    res = g["dp"].post_process_grounded_object_detection(out, inputs["input_ids"], threshold=threshold,
                                                         text_threshold=threshold,
                                                         target_sizes=[(pil.height, pil.width)])[0]
    pairs = sorted(zip(res["boxes"].tolist(), res["scores"].tolist()), key=lambda bs: -bs[1])
    # keep boxes near the best one: weak extras are usually a second guess at the same thing or a background
    # patch. Scores can't tell present from absent (pure noise scores 0.43 for "person"), so callers only ask
    # for things a reviewer saw in the photo.
    return [bs for bs in pairs[:top] if bs[1] >= 0.75 * pairs[0][1]] if pairs else []


def subject_text(im: np.ndarray, prompt: str, feather: float = 0.003) -> np.ndarray | None:
    """A 0..1 mask of everything matching `prompt` (union of the top boxes' SAM masks), or None if nothing found."""
    from PIL import Image
    found = boxes_for(im, prompt)
    if not found:
        return None
    g = _gpu_models()
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    inputs = g["sp"](images=pil, input_boxes=[[b for b, _ in found]], return_tensors="pt").to("cuda")
    with g["torch"].no_grad():
        out = g["sm"](**inputs, multimask_output=False)
    masks = g["sp"].post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"], binarize=True)[0]
    m = np.asarray(masks).astype(np.float32).reshape(-1, im.shape[0], im.shape[1]).max(axis=0)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, feather * max(im.shape[:2])))
    return np.clip(m, 0, 1)
