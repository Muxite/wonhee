"""NVML wrapper: device-scope readings plus process VRAM attributed to a cgroup by host pid.

Adapted from ggez/obs/sampler.py's Gpu. `nvml` is a pynvml-like module, injectable for tests.
Unsupported metrics are recorded once in `capabilities` with a reason and read back as None, never 0.
"""
from __future__ import annotations

from pathlib import Path

PROC = Path("/proc")


def pid_cgroup(pid: int, proc: Path = PROC) -> str | None:
    """cgroup v2 path of a host pid ('0::/system.slice/docker-x.scope' -> '/system.slice/docker-x.scope')."""
    try:
        for line in (proc / str(pid) / "cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return line[3:]
    except OSError:
        return None
    return None


def in_scope(pid: int, scope_rel: str, proc: Path = PROC) -> bool:
    cg = pid_cgroup(pid, proc)
    return cg is not None and (cg == scope_rel or cg.startswith(scope_rel.rstrip("/") + "/"))


class Gpu:
    def __init__(self, nvml=None, index: int = 0):
        self.nv = self.h = None
        self.capabilities: dict[str, str] = {}  # metric -> "ok" | reason it's unavailable
        try:
            if nvml is None:
                import pynvml as nvml
            nvml.nvmlInit()
            self.h = nvml.nvmlDeviceGetHandleByIndex(index)
            self.nv = nvml
        except Exception as e:  # no driver, no GPU, no library
            self.capabilities["nvml"] = f"unavailable: {type(e).__name__}: {e}"
            return
        self.capabilities["nvml"] = "ok"

    @property
    def ok(self) -> bool:
        return self.nv is not None

    def _call(self, metric: str, fn, *a):
        """`fn` is an NVML function or its name; a name the binding lacks marks the metric unsupported."""
        if self.capabilities.get(metric, "ok") != "ok":
            return None
        if isinstance(fn, str):
            name, fn = fn, getattr(self.nv, fn, None)
            if fn is None:
                self.capabilities[metric] = f"unsupported: binding lacks {name}"
                return None
        try:
            v = fn(*a)
            self.capabilities[metric] = "ok"
            return v
        except Exception as e:
            self.capabilities[metric] = f"unsupported: {type(e).__name__}: {e}"
            return None

    def info(self) -> dict:
        if not self.ok:
            return {}
        nv, h = self.nv, self.h
        dec = lambda v: v.decode() if isinstance(v, bytes) else v  # noqa: E731
        mem = self._call("memory", "nvmlDeviceGetMemoryInfo", h)
        return {"gpu_name": dec(self._call("name", "nvmlDeviceGetName", h)),
                "gpu_uuid": dec(self._call("uuid", "nvmlDeviceGetUUID", h)),
                "driver": dec(self._call("driver", "nvmlSystemGetDriverVersion")),
                "vram_total": int(mem.total) if mem is not None else None}

    def device(self) -> dict:
        """Device-scope reading. On a shared GPU none of this is a job's own consumption."""
        if not self.ok:
            return {}
        nv, h, out = self.nv, self.h, {}
        u = self._call("utilization", "nvmlDeviceGetUtilizationRates", h)
        if u is not None:
            out["dev_util"], out["dev_mem_util"] = float(u.gpu), float(u.memory)  # activity %, not bytes
        m = self._call("memory", "nvmlDeviceGetMemoryInfo", h)
        if m is not None:
            out["dev_mem_used"], out["dev_mem_free"] = int(m.used), int(m.free)
        p = self._call("power", "nvmlDeviceGetPowerUsage", h)
        if p is not None:
            out["dev_power_w"] = p / 1000.0
        c = self._call("sm_clock", "nvmlDeviceGetClockInfo", h, getattr(nv, "NVML_CLOCK_SM", 1))
        if c is not None:
            out["dev_sm_mhz"] = float(c)
        t = self._call("temperature", "nvmlDeviceGetTemperature", h, getattr(nv, "NVML_TEMPERATURE_GPU", 0))
        if t is not None:
            out["dev_temp_c"] = float(t)
        fn = ("nvmlDeviceGetCurrentClocksEventReasons" if hasattr(nv, "nvmlDeviceGetCurrentClocksEventReasons")
              else "nvmlDeviceGetCurrentClocksThrottleReasons")
        r = self._call("throttle", fn, h)
        if r is not None:
            out["dev_throttle"] = int(r)
        return out

    def energy_mj(self) -> int | None:
        """Cumulative device energy since driver load, millijoules. Device scope."""
        if not self.ok:
            return None
        return self._call("energy", "nvmlDeviceGetTotalEnergyConsumption", self.h)

    def processes(self) -> list[tuple[int, int | None]]:
        """(host pid, used VRAM bytes or None if NVML withholds it) for compute and graphics contexts."""
        if not self.ok:
            return []
        out: dict[int, int | None] = {}
        for metric, fn_name in (("procs_compute", "nvmlDeviceGetComputeRunningProcesses"),
                                ("procs_graphics", "nvmlDeviceGetGraphicsRunningProcesses")):
            for p in self._call(metric, fn_name, self.h) or []:
                mem = getattr(p, "usedGpuMemory", None)
                mem = None if mem is None or mem >= 2**63 else int(mem)  # NVML_VALUE_NOT_AVAILABLE
                prev = out.get(p.pid)
                out[p.pid] = mem if prev is None else max(prev, mem or 0)
        return sorted(out.items())

    def scope_vram(self, scope_rel: str, proc: Path = PROC) -> tuple[int | None, list[int]]:
        """Sum of VRAM held by pids inside `scope_rel`. Returns (bytes or None if unknown, pids)."""
        pids, total, unknown = [], 0, False
        for pid, mem in self.processes():
            if in_scope(pid, scope_rel, proc):
                pids.append(pid)
                if mem is None:
                    unknown = True
                else:
                    total += mem
        if not pids:
            return 0, []
        return (None if unknown and total == 0 else total), pids
