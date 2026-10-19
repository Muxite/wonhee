"""NVIDIA MPS for co-runs. With MPS, kernels from different processes run on the GPU concurrently through one
server context, instead of time-slicing separate contexts. Comparing a co-run with and without it separates
context-switch cost from real contention for SMs and DRAM.

The control daemon runs in its own container as root: an MPS server only accepts clients of its own uid, and job
containers run as root. Clients reach it through a shared pipe directory and the host IPC namespace."""
from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from gauge import paths  # noqa: E402

ROOT = Path("/tmp/gauge-mps")  # the pipe stays short (unix socket paths max out at 108 bytes)
PIPE, LOG = ROOT / "pipe", ROOT / "log"
KEEP = paths.DATA / "mps-logs"  # daemon output per session, next to the ledger: proof clients really joined
NAME = "gauge-mps"
IMAGE = "gauge/torch:1"  # any CUDA image: the toolkit mounts the driver's MPS binaries with the compute capability
ENV = {"CUDA_MPS_PIPE_DIRECTORY": str(PIPE), "CUDA_MPS_LOG_DIRECTORY": str(LOG)}


def client_args() -> tuple[tuple[str, ...], dict, tuple[str, ...]]:
    """(mounts, env, extra docker args) for a job container that should join the MPS server."""
    return (f"{PIPE}:{PIPE}",), {"CUDA_MPS_PIPE_DIRECTORY": str(PIPE)}, ("--ipc=host",)


def _ctl(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "exec", "-i", NAME, "nvidia-cuda-mps-control"], input=cmd + "\n",
                          capture_output=True, text=True, timeout=30)


def running() -> bool:
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", NAME], capture_output=True, text=True)
    return r.stdout.strip() == "true"


def start(wait_s: float = 15) -> None:
    for d in (PIPE, LOG):
        d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    argv = ["docker", "run", "-d", "--name", NAME, "--gpus", "all", "--ipc=host", "--label", "gauge=1",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "-v", f"{ROOT}:{ROOT}"]
    for k, v in ENV.items():
        argv += ["-e", f"{k}={v}"]
    subprocess.run(argv + [IMAGE, "nvidia-cuda-mps-control", "-f"], check=True, capture_output=True, text=True)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if running() and _ctl("get_server_list").returncode == 0:
            return
        time.sleep(0.5)
    logs = subprocess.run(["docker", "logs", NAME], capture_output=True, text=True)
    stop()
    raise RuntimeError(f"MPS control daemon did not come up: {(logs.stdout + logs.stderr)[-400:]}")


def stop() -> None:
    """Quit the daemon and keep what it printed (-f logs to stdout, not to LOG) in LOG/control-<time>.log: it is
    the evidence that clients really connected, rather than silently getting their own context."""
    if running():
        _ctl("quit")
        time.sleep(1)
    logs = subprocess.run(["docker", "logs", NAME], capture_output=True, text=True)
    if logs.returncode == 0 and (logs.stdout or logs.stderr):
        KEEP.mkdir(parents=True, exist_ok=True)
        (KEEP / f"control-{int(time.time())}.log").write_text(logs.stdout + logs.stderr)
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)


@contextmanager
def daemon():
    start()
    try:
        yield
    finally:
        stop()


def last_log(n: int = 30) -> list[str]:
    logs = sorted(KEEP.glob("control-*.log"))
    return logs[-1].read_text(errors="replace").splitlines()[-n:] if logs else []
