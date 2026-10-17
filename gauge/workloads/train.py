"""Bounded training: fixed steps on synthetic data. Batch size is the classic VRAM/OOM axis."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from gauge.workloads.base import Workload, register

MB = 2**20
# params, and rough saved-activation bytes per 224px sample in fp32 training (forward activations kept for backward)
NETS = {"resnet18": dict(params=11.7e6, act_bytes=60 * MB, gflops=1.8),
        "resnet50": dict(params=25.6e6, act_bytes=190 * MB, gflops=4.1)}


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["resnet18", "resnet50"] = "resnet18"
    batch: int = 64
    image: int = 224
    steps: int = 30
    amp: bool = False


def train_prior(p) -> dict:
    n = NETS[p.model]
    scale = (p.image / 224) ** 2
    act = n["act_bytes"] * scale * p.batch * (0.55 if p.amp else 1.0)
    state = n["params"] * 4 * 3  # weights + grads + momentum, fp32
    flops = 3 * n["gflops"] * 1e9 * scale * p.batch * (p.steps + 1)  # fwd + 2x bwd
    tput = 60e12 if p.amp else 25e12  # effective, well below peak
    runtime = 8 + flops / tput  # 8 s: python + torch + CUDA init
    return {"runtime_s": runtime, "cpu_time_s": 6 + runtime,
            "ram_peak_bytes": 2500 * MB, "vram_peak_bytes": 500 * MB + state + act}


register(Workload(
    name="train.small", family="training", image="gauge/torch:1", build="torch", params=TrainParams, gpu=True,
    shm_size="2g",
    command=lambda p: ["python", "/opt/gauge/train.py", "--model", p.model, "--batch", str(p.batch),
                       "--image", str(p.image), "--steps", str(p.steps), "--amp", str(int(p.amp))],
    size_params=("batch", "steps", "image"), size=lambda p: p.batch * (p.image / 224) ** 2,
    prior=train_prior, doc="torchvision CNN, SGD, synthetic batch, fixed steps"))
