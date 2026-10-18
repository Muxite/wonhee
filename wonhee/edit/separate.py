"""Voice isolation on the GPU: split a clip's audio into voice and everything else (music, traffic, room).

Demucs (htdemucs, MIT, weights from the HF hub on first use) separates vocals / drums / bass / other; the voice is
the `vocals` stem and the bed is the sum of the rest. Needs the `gpu` dependency group; on a CPU it still works,
about 10x slower. Audio goes in and out through ffmpeg as raw float32, so any container works.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from wonhee.procenv import clean_env

_sep: dict[str, object] = {}


def available() -> bool:
    try:
        import demucs.api  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _separator(model: str):
    if model not in _sep:
        import torch
        from demucs.api import Separator
        _sep[model] = Separator(model=model, device="cuda" if torch.cuda.is_available() else "cpu")
    return _sep[model]


def _decode(src: Path, sr: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(src), "-map", "0:a:0", "-ac", "2", "-ar", str(sr),
                          "-f", "f32le", "-"], env=clean_env(), capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def _encode(wav: np.ndarray, sr: int, dest: Path) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
                    "-c:a", "pcm_f32le", str(dest)], input=np.ascontiguousarray(wav.T).tobytes(), env=clean_env(),
                   check=True)


def split_voice(src: Path, voice: Path, bed: Path, model: str = "htdemucs") -> None:
    """Write the voice stem and the rest (music, noise) as float WAVs at the model's rate (44.1 kHz)."""
    import torch
    sep = _separator(model)
    sr = sep.samplerate
    wav = torch.from_numpy(_decode(src, sr))
    with torch.no_grad():
        _, stems = sep.separate_tensor(wav, sr)
    v = stems["vocals"].cpu().numpy()
    rest = sum(t for k, t in stems.items() if k != "vocals").cpu().numpy()
    _encode(v, sr, voice)
    _encode(rest, sr, bed)
