"""SigLIP2 image and text vectors in one space, for searching shots by description ("bus at sunset").

google/siglip2-so400m-patch14-384 (Apache-2.0) through `transformers`, fp16 on the GPU (the `gpu` dependency
group), fp32 on the CPU otherwise. Vectors are L2-normalized float16, so a dot product is the cosine. The model is
loaded once per process behind a lock, like the mask models (edit/mask.py)."""
from __future__ import annotations

import threading

import cv2
import numpy as np

MODEL_ID = "google/siglip2-so400m-patch14-384"
TEXT_LEN = 64  # SigLIP was trained on max_length-padded text; other paddings shift the scores
_m: dict[str, object] = {}
_lock = threading.Lock()


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load() -> dict:
    if _m:
        return _m
    with _lock:
        if not _m:
            import torch
            from transformers import AutoModel, AutoProcessor

            from wonhee.edit.mask import _pretrained
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = _pretrained(AutoModel, MODEL_ID).to(dev, torch.float16 if dev == "cuda" else torch.float32)
            _m.update({"torch": torch, "dev": dev, "proc": _pretrained(AutoProcessor, MODEL_ID),
                       "model": model.eval()})  # published together, last: never half-loaded
    return _m


def _unit(x) -> np.ndarray:
    x = getattr(x, "pooler_output", x)  # transformers 5 returns an output object; 4.x returned the tensor
    v = x.float().cpu().numpy()
    return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)).astype(np.float16)


def image_vecs(images: list[np.ndarray], batch: int = 16) -> np.ndarray:
    """BGR images (any size) -> (n, d) float16 unit vectors."""
    from PIL import Image
    m = load()
    out = []
    for i in range(0, len(images), batch):
        pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images[i:i + batch]]
        inputs = m["proc"](images=pil, return_tensors="pt").to(m["dev"])
        inputs["pixel_values"] = inputs["pixel_values"].to(m["model"].dtype)
        with m["torch"].no_grad():
            out.append(_unit(m["model"].get_image_features(**inputs)))
    return np.concatenate(out) if out else np.zeros((0, 0), np.float16)


def text_vecs(texts: list[str]) -> np.ndarray:
    m = load()
    inputs = m["proc"](text=[t.lower() for t in texts], padding="max_length", max_length=TEXT_LEN,
                       truncation=True, return_tensors="pt").to(m["dev"])
    with m["torch"].no_grad():
        return _unit(m["model"].get_text_features(**inputs))
