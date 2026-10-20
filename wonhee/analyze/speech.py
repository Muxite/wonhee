from __future__ import annotations

import ctypes
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class SpokenWord:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str
    words: list[SpokenWord] | None = None  # faster-whisper word timestamps; None = not computed


@dataclass
class Transcript:
    language: str | None
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


def first_word_ms(t: Transcript) -> int | None:
    return next((s.start_ms for s in t.segments if s.text.strip()), None)


class Transcriber(Protocol):
    def transcribe(self, path: Path) -> Transcript: ...


_CUDA_LIBS = (("nvidia.cublas", "nvidia-cublas-cu12", ("libcublasLt.so.12", "libcublas.so.12")),
              ("nvidia.cudnn", "nvidia-cudnn-cu12", ("libcudnn*.so.9",)))
_preloaded = False


def _preload_cuda_libs() -> None:
    """Load pip-installed cuBLAS/cuDNN globally so ctranslate2 can find them. Idempotent."""
    global _preloaded
    if _preloaded:
        return
    import importlib.util
    for module, pip_name, patterns in _CUDA_LIBS:
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            spec = None
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"CUDA library package missing: pip install {pip_name}")
        libdir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        for pat in patterns:
            libs = sorted(glob.glob(os.path.join(libdir, pat)))
            if not libs:
                raise RuntimeError(f"{pip_name}: no files matching {pat} in {libdir}")
            for lib in libs:
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError as e:
                    raise RuntimeError(f"{pip_name}: failed to load {lib}: {e}") from e
    _preloaded = True


class WhisperTranscriber:
    def __init__(self, model: str, device: str, compute_type: str):
        self.model_name, self.device, self.compute_type = model, device, compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            if self.device == "cuda":
                import ctranslate2
                if ctranslate2.get_cuda_device_count() == 0:
                    raise RuntimeError("whisper_device=cuda but no CUDA device is visible to ctranslate2 "
                                       "(run `wonhee doctor`; set WONHEE_WHISPER_DEVICE=cpu and "
                                       "WONHEE_WHISPER_COMPUTE_TYPE=int8 to use the CPU deliberately)")
                _preload_cuda_libs()
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        return self._model

    def ensure_ready(self) -> None:
        self._load()

    def transcribe(self, path: Path) -> Transcript:
        from wonhee.media import has_audio

        if not has_audio(path):
            return Transcript(None, [])  # no audio stream: nothing said, not an error
        segments, info = self._load().transcribe(str(path), beam_size=5, vad_filter=True,
                                                word_timestamps=True)
        segs = [Segment(int(s.start * 1000), int(s.end * 1000), s.text,
                        [SpokenWord(int(w.start * 1000), int(w.end * 1000), w.word.strip())
                         for w in (s.words or []) if w.word.strip()] or None)
                for s in segments]
        return Transcript(info.language, segs)
