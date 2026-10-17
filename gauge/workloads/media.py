"""Media family: ffmpeg transcode (CPU libx264 vs GPU NVENC) over a fixed pre-generated source clip."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from gauge.workloads.base import Workload, register
from gauge import paths

MB = 2**20
FFMPEG_IMAGE = "linuxserver/ffmpeg:7.1.1"
MEDIA_DIR = paths.DATA / "media"
SOURCE = "src_1080p30_120s.mp4"  # noisy test pattern, so encoders can't cheat on flat frames


def prepare_media() -> Path:
    """Generate the shared source clip once (CPU, outside any measured run)."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out = MEDIA_DIR / SOURCE
    if not out.exists():
        subprocess.run(["docker", "run", "--rm", "-v", f"{MEDIA_DIR}:/media", FFMPEG_IMAGE, "-hide_banner", "-y",
                        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30,noise=alls=20:allf=t+u",
                        "-t", "120", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", f"/media/{SOURCE}"],
                       check=True)
    return out


class TranscodeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seconds: float = 20
    height: int = 1080  # output height; width follows 16:9
    codec: Literal["libx264", "h264_nvenc"] = "libx264"
    preset: str = "medium"  # libx264: ultrafast..veryslow; nvenc: p1..p7


def _px(p) -> float:
    return p.seconds * 30 * (p.height * 16 / 9) * p.height  # output pixels


# Rough engineering guesses, deliberately uncalibrated: the calibrated predictor's job is to fix them.
X264_PX_PER_CPU_S = {"ultrafast": 1.2e8, "veryfast": 6e7, "fast": 3.5e7, "medium": 2.5e7, "slow": 1.2e7}
DECODE_CPU_S_PER_SRC_S = 0.25  # 1080p30 h264 decode
SCALE_CPU_PER_PX = 2e-9


def transcode_prior(p) -> dict:
    px = _px(p)
    decode = DECODE_CPU_S_PER_SRC_S * p.seconds + SCALE_CPU_PER_PX * px
    if p.codec == "libx264":
        cpu = px / X264_PX_PER_CPU_S.get(p.preset, 2.5e7) + decode
        runtime = cpu / 12 + 0.5  # x264 frame threads keep ~12 of 16 cores busy
        return {"runtime_s": runtime, "cpu_time_s": cpu,
                "ram_peak_bytes": 150 * MB + 60 * (p.height * 16 / 9) * p.height * 1.5}
    # NVENC: encoder ~ 1.2e9 px/s; decode+scale on CPU usually becomes the limit
    runtime = max(px / 1.2e9, decode / 4) + 0.8
    return {"runtime_s": runtime, "cpu_time_s": decode + 0.3 * runtime,
            "ram_peak_bytes": 250 * MB, "vram_peak_bytes": 350 * MB}


register(Workload(
    name="ffmpeg.transcode", family="media", image=FFMPEG_IMAGE, params=TranscodeParams,
    gpu=True,  # harmless for libx264, and keeps the two codecs on identical container config
    env={"NVIDIA_DRIVER_CAPABILITIES": "compute,video,utility"},
    mounts=(f"{MEDIA_DIR}:/media:ro",),
    command=lambda p: ["ffmpeg", "-hide_banner", "-nostats", "-i", f"/media/{SOURCE}", "-t", str(p.seconds),
                       "-vf", f"scale=-2:{p.height}", "-c:v", p.codec, "-preset", p.preset, "-an", "-f", "null", "-"],
    size_params=("seconds", "height"), size=_px, prior=transcode_prior,
    doc="decode the 1080p source, scale, encode with libx264 or NVENC, discard output"))
