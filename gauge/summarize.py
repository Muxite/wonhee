"""Run -> summary envelope. Every field carries its measurement scope; unknowns are None, never 0."""
from __future__ import annotations

from gauge.spec import CENSORED

COLLECTOR_VERSION = 1


def _max(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return max(vals) if vals else None


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def peak_cores(rows) -> float | None:
    """Highest CPU-cores over any adjacent sample pair (observed, interval-limited)."""
    best = None
    for a, b in zip(rows, rows[1:]):
        if a.get("cpu_usage_usec") is None or b.get("cpu_usage_usec") is None or b["t"] <= a["t"]:
            continue
        c = (b["cpu_usage_usec"] - a["cpu_usage_usec"]) / 1e6 / (b["t"] - a["t"])
        best = c if best is None else max(best, c)
    return best


def integrate_power(rows) -> float | None:
    pts = [(r["t"], r["dev_power_w"]) for r in rows if r.get("dev_power_w") is not None]
    if len(pts) < 2:
        return None
    return sum((t1 - t0) * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(pts, pts[1:]))


def classify(*, exit_code, timed_out: bool, oom_kill_events: int, docker_oom: bool, workload_cuda_oom: bool,
             log_cuda_oom: bool) -> tuple[str, str]:
    """-> (failure_class, source). cuda_oom only from the workload's own report or a marked log match."""
    if timed_out:
        return "timeout", "runner"
    if oom_kill_events or docker_oom:
        return "host_oom", "cgroup memory.events" if oom_kill_events else "docker OOMKilled"
    if exit_code == 0:
        return "success", "exit_code"
    if workload_cuda_oom:
        return "cuda_oom", "workload"
    if log_cuda_oom:
        return "cuda_oom", "log-derived"
    if exit_code is None:
        return "unknown", "no exit code"
    return "application_error", "exit_code"


def phases(marks: list[dict], runtime_s: float | None) -> dict:
    """Marks (seconds from container start) -> phase durations. `load`: start to the first model_loaded.
    `compute`: summed compute_started..compute_finished spans (a warm worker emits one span per job, listed in
    `spans`). `tail`: last compute_finished to the end. Missing marks leave a phase None, never 0."""
    first = {}
    for m in marks:
        first.setdefault(m["mark"], m["t"])
    spans, start = [], None
    for m in marks:
        if m["mark"] == "compute_started":
            start = m["t"]
        elif m["mark"] == "compute_finished" and start is not None:
            spans.append(m["t"] - start)
            start = None
    ends = [m["t"] for m in marks if m["mark"] == "compute_finished"]
    return {"load": first.get("model_loaded"), "compute": sum(spans) if spans else None, "spans": spans,
            "tail": runtime_s - ends[-1] if ends and runtime_s is not None else None}


def summarize(*, rows: list[dict], end: dict | None, end_is_final: bool, workload_s: float | None,
              wall_s: float | None, energy_j: float | None, sampler, gpu_caps: dict, failure_class: str) -> dict:
    cpu = (end or {}).get("cpu", {})
    io = (end or {}).get("io", {})
    ev = (end or {}).get("memory_events", {})
    cpu_s = cpu.get("usage_usec") / 1e6 if cpu.get("usage_usec") is not None else None
    vram_rows = [r for r in rows if r.get("vram_proc_bytes") is not None]
    power_j = integrate_power(rows)
    s = {
        "collector_version": COLLECTOR_VERSION,
        "failure_class": failure_class,
        "censored": failure_class in CENSORED,
        # time
        "runtime_s": workload_s, "container_wall_s": wall_s,
        # cpu (scope: container, cumulative counters)
        "cpu_time_s": cpu_s,
        "cpu_user_s": cpu["user_usec"] / 1e6 if "user_usec" in cpu else None,
        "cpu_system_s": cpu["system_usec"] / 1e6 if "system_usec" in cpu else None,
        "cpu_throttled_s": cpu["throttled_usec"] / 1e6 if "throttled_usec" in cpu else None,
        "cpu_mean_cores": cpu_s / workload_s if cpu_s is not None and workload_s else None,
        "cpu_peak_cores_observed": peak_cores(rows),
        # ram (scope: container; memory.peak is the kernel's high-water mark, not sampled)
        "ram_peak_bytes": (end or {}).get("memory_peak"),
        "ram_peak_sampled_bytes": _max(rows, "mem_current"),
        "ram_anon_peak_sampled_bytes": _max(rows, "mem_anon"),
        "ram_file_peak_sampled_bytes": _max(rows, "mem_file"),
        "swap_peak_bytes": (end or {}).get("swap_peak"),
        "memory_limit_bytes": (end or {}).get("memory_max"),
        "oom_events": ev.get("oom"), "oom_kill_events": ev.get("oom_kill"),
        "pids_peak": (end or {}).get("pids_peak"),
        # pressure: stall-time totals, usec
        **{f"psi_{res}_{k}": v for res in ("cpu", "memory", "io")
           for k, v in (end or {}).get(f"{res}_pressure", {}).items()},
        # io (scope: container)
        "io_read_bytes": io.get("rbytes"), "io_write_bytes": io.get("wbytes"),
        "io_read_ops": io.get("rios"), "io_write_ops": io.get("wios"),
        # gpu memory (scope: process_sum; sampled, so an observed peak)
        "vram_peak_bytes": _max(vram_rows, "vram_proc_bytes") if vram_rows else None,
        "vram_mean_bytes": _mean(vram_rows, "vram_proc_bytes") if vram_rows else None,
        "vram_scope": "process_sum" if vram_rows else None,
        "gpu_pids": sorted(sampler.gpu_pids) if sampler else [],
        # gpu activity/power (scope: device; on a shared GPU this is not the job's own)
        "dev_util_mean": _mean(rows, "dev_util"), "dev_util_peak": _max(rows, "dev_util"),
        "dev_mem_util_mean": _mean(rows, "dev_mem_util"),
        "dev_power_mean_w": _mean(rows, "dev_power_w"), "dev_power_peak_w": _max(rows, "dev_power_w"),
        "dev_energy_j": energy_j, "dev_energy_j_integrated": power_j,
        "dev_energy_source": "nvml_counter" if energy_j is not None else ("integrated_power" if power_j else None),
        "dev_temp_peak_c": _max(rows, "dev_temp_c"),
        "dev_enc_util_mean": _mean(rows, "dev_enc_util"), "dev_dec_util_mean": _mean(rows, "dev_dec_util"),
        "dev_throttle_any": any(r.get("dev_throttle") for r in rows) if rows else None,
        "device_scope_note": "device-level readings; attributable only if no other GPU work overlapped",
        # measurement quality
        "end_snapshot": "final" if end_is_final else ("last_periodic" if end else "missing"),
        "sample_count": len(rows), "missed_samples": sampler.missed if sampler else None,
        "sampling_interval_ms": sampler.interval * 1000 if sampler else None,
        "collector_cpu_s": sampler.cpu_s if sampler else None,
        "metric_capabilities": gpu_caps,
    }
    return s
