"""Job and prediction records. A job is finite; a prediction is frozen before the job launches."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

ExecutionMode = Literal["isolated", "shared_service"]
Warmness = Literal["cold", "warm", "unknown"]
FailureClass = Literal["success", "host_oom", "cuda_oom", "timeout", "application_error", "unknown"]
CENSORED = {"host_oom", "cuda_oom", "timeout"}

# Envelope targets every predictor may fill. Upper bounds are bounds, not calibrated quantiles.
TARGETS = ("runtime_s", "cpu_time_s", "ram_peak_bytes", "vram_peak_bytes")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:16]


class JobSpec(BaseModel):
    workload: str  # registry name, e.g. "ffmpeg.transcode"
    params: dict[str, Any] = Field(default_factory=dict)
    execution_mode: ExecutionMode = "isolated"
    warmness: Warmness = "unknown"
    mem_limit: str | None = None  # docker --memory; None = workload default
    timeout_s: float = 1800
    provenance: dict[str, Any] = Field(default_factory=dict)  # wonhee run/node/attempt, moeka task id
    # Nsight Systems inside the container. "trace": this process tree's CUDA kernels/copies (process scope).
    # "metrics": trace + GPU hardware counters at 10 kHz (device scope; only one job at a time can hold them).
    profile: Literal["none", "trace", "metrics"] = "none"
    # How co-running jobs share the GPU: time-sliced contexts (the driver default) or one MPS server (gauge.mps).
    sharing: Literal["timeslice", "mps"] = "timeslice"

    def spec_hash(self, validated_params: dict | None = None) -> str:
        """Hash of the job as run. Pass the validated params (defaults filled in) so `batch=128` and
        `{model: resnet18, steps: 30, batch: 128}` hash the same."""
        return digest({"workload": self.workload, "params": validated_params if validated_params is not None else self.params,
                       "mode": self.execution_mode, "mem_limit": self.mem_limit,
                       **({"sharing": self.sharing} if self.sharing != "timeslice" else {})})  # old hashes stay put


class Prediction(BaseModel):
    predictor: str
    predictor_version: int
    status: Literal["supported", "insufficient_data"]
    runtime_estimate_s: float | None = None
    runtime_upper_s: float | None = None
    cpu_time_estimate_s: float | None = None
    ram_peak_upper_bytes: float | None = None
    vram_peak_upper_bytes: float | None = None
    evidence_run_ids: list[int] = Field(default_factory=list)
    history_cutoff: float | None = None  # unix time; only runs finished before this were used
    note: str = ""

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_run_ids)
