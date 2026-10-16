"""wonhee's local-hardware nodes, run through wonhee's own `node run --no-cache` in a throwaway home.

Only speech_subs (faster-whisper, GPU) and burn_subtitles (ffmpeg libx264, CPU) use local hardware.
grab, translate and screen_text's LLM half are network-bound; they are not swept here.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from gauge.workloads.base import Workload, register
from gauge.workloads.media import MEDIA_DIR

MB = 2**20
IMAGE = "gauge/wonhee:1"
WONHEE_REPO = Path("/home/muk/projects/wonhee")
HF_CACHE = Path.home() / ".cache" / "huggingface"
CLIP_SECONDS = (10, 20, 40, 60, 100)
MANIFEST = MEDIA_DIR / "wonhee_clips.json"


def prepare_clips(source: Path) -> dict:
    """Cut speech clips from one local reel (third-party media: stays under gitignored data/), plus
    synthetic subtitles for burn. Durations are re-measured, since stream-copy cuts land on keyframes."""
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(source)], capture_output=True, text=True, check=True).stdout.split()
    if "audio" not in streams:
        raise ValueError(f"{source} has no audio stream: speech_subs would short-circuit on it")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for sec in CLIP_SECONDS:
        clip = MEDIA_DIR / f"wonhee_{sec}s.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-t", str(sec), "-c", "copy",
                        str(clip)], check=True)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                    "csv=p=0", str(clip)], capture_output=True, text=True, check=True).stdout)
        lines = [{"id": i, "start_ms": t * 1000, "end_ms": t * 1000 + 1800, "text": f"subtitle line number {i}",
                  "source": "speech"} for i, t in enumerate(range(0, int(dur), 2))]
        (MEDIA_DIR / f"wonhee_{sec}s.subs.json").write_text(json.dumps({"language": "en", "lines": lines}))
        out[str(sec)] = {"file": clip.name, "duration_s": dur}
    MANIFEST.write_text(json.dumps(out, indent=1))
    return out


def clip_duration(sec: int) -> float:
    try:
        return json.loads(MANIFEST.read_text())[str(sec)]["duration_s"]
    except (OSError, KeyError):
        return float(sec)


COMMON = dict(image=IMAGE, build="wonhee", build_context=str(WONHEE_REPO), family="media",
              mounts=(f"{MEDIA_DIR}:/media:ro", f"{HF_CACHE}:/root/.cache/huggingface:ro"))


class SpeechParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_s: Literal[CLIP_SECONDS] = 20  # type: ignore[valid-type]
    whisper_model: Literal["tiny", "base", "small", "turbo"] = "turbo"
    compute_type: Literal["float16", "int8_float16"] = "float16"


WHISPER = {"tiny": (39e6, 0.004), "base": (74e6, 0.006), "small": (244e6, 0.012), "turbo": (809e6, 0.02)}


def speech_prior(p) -> dict:
    params, rtf = WHISPER[p.whisper_model]
    audio = clip_duration(p.clip_s)
    bytes_per = 2 if p.compute_type == "float16" else 1.2
    load = 6 + params * bytes_per / 1.5e9  # python + ctranslate2 + CUDA init, then weights
    return {"runtime_s": load + rtf * audio * 3 + 1.0,  # ×3: beam 5 + VAD + import of the clip
            "cpu_time_s": load + 0.3 * audio, "ram_peak_bytes": 900 * MB + params * 2,
            "vram_peak_bytes": 450 * MB + params * bytes_per * 1.3}


register(Workload(
    name="wonhee.speech_subs", params=SpeechParams, gpu=True, **COMMON,
    param_env=lambda p: {"WONHEE_WHISPER_MODEL": p.whisper_model, "WONHEE_WHISPER_COMPUTE_TYPE": p.compute_type},
    command=lambda p: ["wonhee", "node", "run", "speech_subs", "--in", f"video=/media/wonhee_{p.clip_s}s.mp4",
                       "--no-cache"],
    size_params=("clip_s", "whisper_model", "compute_type"), size=lambda p: clip_duration(p.clip_s),
    prior=speech_prior, features=lambda p: {**p.model_dump(), "audio_s": clip_duration(p.clip_s)},
    doc="wonhee speech_subs node: faster-whisper on the clip's audio"))


class BurnParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_s: Literal[CLIP_SECONDS] = 20  # type: ignore[valid-type]


def burn_prior(p) -> dict:
    frames = clip_duration(p.clip_s) * 30
    cpu = frames * 1920 * 1080 / 2.5e7 + 2  # libx264 default preset at 1080p, same guess as ffmpeg.transcode
    return {"runtime_s": cpu / 12 + 4, "cpu_time_s": cpu + 3, "ram_peak_bytes": 1300 * MB}


register(Workload(
    name="wonhee.burn_subtitles", params=BurnParams, **COMMON,
    command=lambda p: ["wonhee", "node", "run", "burn_subtitles", "--in", f"video=/media/wonhee_{p.clip_s}s.mp4",
                       "--in", f"subtitles=/media/wonhee_{p.clip_s}s.subs.json", "--no-cache"],
    size_params=("clip_s",), size=lambda p: clip_duration(p.clip_s), prior=burn_prior,
    features=lambda p: {**p.model_dump(), "video_s": clip_duration(p.clip_s), "width": 1080, "height": 1920},
    doc="wonhee burn_subtitles node: ffmpeg ass filter + libx264 re-encode"))
