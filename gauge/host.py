"""Host fingerprint and initial-state snapshot, so runs on different setups or under load stay comparable."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

from gauge.probe.gpu import Gpu, pid_cgroup


def _meminfo() -> dict[str, int]:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        parts = v.split()
        if parts:
            out[k] = int(parts[0]) * (1024 if len(parts) > 1 else 1)
    return out


def _docker_version() -> str | None:
    try:
        return subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def fingerprint(gpu: Gpu) -> dict:
    return {"hostname": platform.node(), "kernel": platform.release(), "cpus": os.cpu_count(),
            "ram_total": _meminfo().get("MemTotal"), "docker": _docker_version(), **gpu.info()}


def initial_state(gpu: Gpu, overlapping_run_ids: list[int]) -> dict:
    mi = _meminfo()
    dev = gpu.device()
    others = [{"pid": pid, "vram": mem, "cgroup": pid_cgroup(pid)} for pid, mem in gpu.processes()]
    return {"ram_available": mi.get("MemAvailable"), "swap_used": mi.get("SwapTotal", 0) - mi.get("SwapFree", 0),
            "vram_free": dev.get("dev_mem_free"), "vram_used": dev.get("dev_mem_used"),
            "gpu_temp_c": dev.get("dev_temp_c"), "gpu_processes": others,
            "loadavg": os.getloadavg()[0], "overlapping_run_ids": overlapping_run_ids}
