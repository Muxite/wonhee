"""Known-answer workloads that check the probes, not the predictors."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from gauge.workloads.base import Workload, register

MB = 2**20


class CpuParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workers: int = 4
    seconds: int = 10  # stress-ng reads fractional timeouts as 0 = forever


class VmParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mb: int = 2048
    seconds: int = 5


class AllocParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gib: float = 4.0
    hold_s: float = 3.0
    child: bool = False  # allocate from a forked child, to test descendant attribution


register(Workload(
    name="stress.cpu", family="sanity", image="gauge/stress:1", build="stress", params=CpuParams,
    command=lambda p: ["stress-ng", "--cpu", str(p.workers), "--cpu-method", "matrixprod",
                       "--timeout", f"{p.seconds}s", "--quiet"],
    size_params=("workers", "seconds"), size=lambda p: p.workers * p.seconds,
    prior=lambda p: {"runtime_s": p.seconds + 0.2, "cpu_time_s": p.workers * p.seconds, "ram_peak_bytes": 32 * MB},
    doc="N busy cores for S seconds: cpu_time_s should be N*S"))

register(Workload(
    name="stress.vm", family="sanity", image="gauge/stress:1", build="stress", params=VmParams,
    command=lambda p: ["stress-ng", "--vm", "1", "--vm-bytes", f"{p.mb}M", "--vm-keep", "--vm-populate",
                       "--timeout", f"{p.seconds}s", "--quiet"],
    size_params=("mb", "seconds"), size=lambda p: p.mb,
    prior=lambda p: {"runtime_s": p.seconds + 0.5, "cpu_time_s": p.seconds, "ram_peak_bytes": p.mb * MB + 16 * MB},
    doc="hold M MiB of anonymous memory: memory.peak should be ~M MiB"))

ALLOC = r"""
import os, sys, time, torch
gib, hold, child = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3] == "1"
def work():
    x = torch.empty(int(gib * 2**30), dtype=torch.uint8, device="cuda"); x.fill_(1); torch.cuda.synchronize()
    print("GAUGE_MARK compute_started", flush=True); time.sleep(hold)
if child:
    pid = os.fork()
    if pid == 0:
        work(); os._exit(0)
    _, st = os.waitpid(pid, 0); sys.exit(os.waitstatus_to_exitcode(st))
work()
"""

register(Workload(
    name="torch.alloc", family="sanity", image="gauge/torch:1", build="torch", params=AllocParams, gpu=True,
    command=lambda p: ["python", "-c", ALLOC, str(p.gib), str(p.hold_s), "1" if p.child else "0"],
    size_params=("gib", "hold_s"), size=lambda p: p.gib,
    prior=lambda p: {"runtime_s": p.hold_s + 4, "vram_peak_bytes": p.gib * 2**30 + 400 * MB,
                     "ram_peak_bytes": 1500 * MB},
    doc="allocate G GiB on the GPU: attributed VRAM should be G GiB + CUDA context"))


class CeilParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rep: int = 0  # no knobs: the benchmark is fixed; rep only distinguishes repeats


register(Workload(
    name="micro.ceilings", family="micro", image="gauge/torch:1", build="torch", params=CeilParams, gpu=True,
    command=lambda p: ["python", "/opt/gauge/ceilings.py"], size_params=("rep",), size=lambda p: 1.0,
    prior=lambda p: {"runtime_s": 40, "vram_peak_bytes": 2.5 * 2**30, "ram_peak_bytes": 4 * 2**30},
    doc="peak matmul TFLOPS per format, gemv/device/PCIe bandwidth, kernel launch cost"))


class IdleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seconds: int = 10


register(Workload(
    name="micro.idle", family="micro", image="gauge/torch:1", build="torch", params=IdleParams, gpu=True,
    command=lambda p: ["sleep", str(p.seconds)], size_params=("seconds",), size=lambda p: p.seconds,
    prior=lambda p: {"runtime_s": p.seconds + 0.5, "ram_peak_bytes": 16 * 2**20},
    doc="does nothing: under --profile metrics it records what the GPU does without our work (background floor)"))


class StressParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["bw", "compute", "launch"] = "bw"
    seconds: float = 30


register(Workload(
    name="stress.gpu", family="micro", image="gauge/torch:1", build="torch", params=StressParams, gpu=True,
    command=lambda p: ["python", "/opt/gauge/stressor.py", p.mode, str(p.seconds)],
    size_params=("seconds",), size=lambda p: p.seconds,
    prior=lambda p: {"runtime_s": p.seconds + 6, "cpu_time_s": p.seconds + 4, "ram_peak_bytes": 2 * 2**30,
                     "vram_peak_bytes": (2.6 if p.mode == "bw" else 0.6) * 2**30},
    doc="one GPU resource held busy (DRAM bandwidth, tensor compute, or kernel launches) as a targeted co-runner"))
