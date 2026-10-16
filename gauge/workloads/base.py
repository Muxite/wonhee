"""Workload definitions: image + command + params + pre-launch features + analytic prior."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from gauge.spec import Prediction, digest

PRIOR_VERSION = 1
PRIOR_MARGIN = 0.25  # an uncalibrated formula earns a wide bound


@dataclass(frozen=True)
class Workload:
    name: str
    family: str  # media | inference | training | sanity
    image: str
    params: type[BaseModel]
    command: Callable[[Any], list[str]]  # validated params -> argv inside the container
    size_params: tuple[str, ...]  # params that scale the job; excluded from the config hash
    size: Callable[[Any], float]  # the prior's driving variable (frames*pixels, audio_s, batch, ctx...)
    prior: Callable[[Any], dict]  # -> point estimates keyed by TARGETS (runtime_s, cpu_time_s, ...)
    features: Callable[[Any], dict] = lambda p: p.model_dump()
    build: str | None = None  # images/<dir> holding a Dockerfile for `image`
    gpu: bool = False
    mounts: tuple[str, ...] = ()  # docker -v specs
    env: dict[str, str] = field(default_factory=dict)
    mem_limit: str = "24g"
    shm_size: str | None = None
    doc: str = ""

    def validate(self, raw: dict) -> BaseModel:
        return self.params.model_validate(raw)

    def config_hash(self, p: BaseModel) -> str:
        cfg = {k: v for k, v in p.model_dump().items() if k not in self.size_params}
        return digest({"workload": self.name, "config": cfg})

    def prior_prediction(self, p: BaseModel) -> Prediction:
        e = self.prior(p)
        up = lambda k: e[k] * (1 + PRIOR_MARGIN) if e.get(k) is not None else None  # noqa: E731
        return Prediction(predictor="prior", predictor_version=PRIOR_VERSION, status="supported",
                          runtime_estimate_s=e.get("runtime_s"), runtime_upper_s=up("runtime_s"),
                          cpu_time_estimate_s=e.get("cpu_time_s"), ram_peak_upper_bytes=up("ram_peak_bytes"),
                          vram_peak_upper_bytes=up("vram_peak_bytes"))


REGISTRY: dict[str, Workload] = {}


def register(w: Workload) -> Workload:
    REGISTRY[w.name] = w
    return w


def get(name: str) -> Workload:
    from gauge import workloads  # noqa: F401  populate registry

    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown workload {name!r}; known: {sorted(REGISTRY)}") from None


# Workload stdout protocol. Optional: a workload that prints nothing still gets measured.
MARK_RE = re.compile(r"^GAUGE_MARK (\w+)")
OUTCOME_RE = re.compile(r"^GAUGE_OUTCOME (\{.*\})\s*$")
DONE_RE = re.compile(r"^GAUGE_DONE rc=(-?\d+)")
CUDA_OOM_RE = re.compile(r"CUDA out of memory|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|"
                         r"out of memory.*cuda|cudaMalloc failed", re.I)


def parse_outcome_line(line: str) -> dict | None:
    m = OUTCOME_RE.match(line)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
