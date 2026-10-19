"""Launch one workload container and keep its cgroup alive until the host has taken its final reading.

A container's cgroup is deleted the moment it exits, which races any end-of-run counter read. So the
workload runs under a tiny sh wrapper: run the command, print GAUGE_DONE rc=N, then block on stdin.
The host reads the cgroup, then closes stdin and the container exits with the workload's code.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from gauge.probe.cgroup import CGROUP_ROOT, scope_path

WRAPPER = '"$@" </dev/null; rc=$?; echo "GAUGE_DONE rc=$rc"; read _ ; exit $rc'


def docker_argv(name: str, image: str, cmd: list[str], *, cidfile: Path, gpu: bool, mem_limit: str,
                mounts=(), env=None, shm_size=None, cap_add=(), extra=()) -> list[str]:
    argv = ["docker", "run", "-i", "--rm=false", "--name", name, "--cidfile", str(cidfile),
            "--memory", mem_limit, "--memory-swap", mem_limit,  # no swap: a RAM overrun is a clean cgroup OOM
            "--label", "gauge=1", "--entrypoint", "/bin/sh"]
    if gpu:
        argv += ["--gpus", "all"]
    if shm_size:
        argv += ["--shm-size", shm_size]
    for cap in cap_add:
        argv += ["--cap-add", cap]
    for m in mounts:
        argv += ["-v", m]
    for k, v in (env or {}).items():
        argv += ["-e", f"{k}={v}"]
    argv += list(extra)
    return argv + [image, "-c", WRAPPER, "gauge-wrap", *cmd]


def image_digest(image: str) -> str | None:
    """The local image ID, plus the registry digest when the image came from a registry ("id@repo@sha256:..."):
    the ID alone cannot be pulled again once the image is pruned."""
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}|{{join .RepoDigests \",\"}}", image],
                       capture_output=True, text=True)
    iid, _, repo = r.stdout.strip().partition("|")
    return (f"{iid}@{repo}" if repo else iid) or None


def _ts(s: str | None) -> float | None:
    if not s or s.startswith("0001-"):
        return None
    s = s.rstrip("Z")
    main, _, frac = s.partition(".")
    return datetime.fromisoformat(main + "+00:00").timestamp() + (float("0." + frac[:6]) if frac else 0.0)


def inspect(cid: str) -> dict:
    r = subprocess.run(["docker", "inspect", cid], capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    st = json.loads(r.stdout)[0]["State"]
    return {"exit_code": st.get("ExitCode"), "oom_killed": st.get("OOMKilled"),
            "started_at": _ts(st.get("StartedAt")), "finished_at": _ts(st.get("FinishedAt")),
            "error": st.get("Error") or None}


class Container:
    """One running job container. Lines from stdout+stderr are timestamped and handed to `on_line`."""

    def __init__(self, name: str, argv_fn, on_line):
        self.name = name
        self._tmp = Path(tempfile.mkdtemp(prefix="gauge-"))
        self.cidfile = self._tmp / "cid"
        self.argv = argv_fn(self.cidfile)
        self.on_line = on_line
        self.proc: subprocess.Popen | None = None
        self.cid: str | None = None
        self.scope: Path | None = None
        self.done = threading.Event()  # GAUGE_DONE seen, or the process ended
        self.done_at: float | None = None
        self.wrapped_rc: int | None = None
        self.lines: list[tuple[float, str]] = []

    def start(self, timeout_s: float = 120) -> None:
        self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        deadline = time.time() + timeout_s  # covers an image pull on first use
        while time.time() < deadline:
            if self.cid is None and self.cidfile.exists():
                self.cid = self.cidfile.read_text().strip() or None
            if self.cid:
                self.scope = scope_path(self.cid)
                if self.scope:
                    return
            if self.proc.poll() is not None:
                break
            time.sleep(0.005)
        if self.cid is None:
            raise RuntimeError(f"container never started: {' '.join(l for _, l in self.lines[-20:])}")
        # Started and already gone (very short job that died before GAUGE_DONE): no live cgroup.

    def _pump(self) -> None:
        from gauge.workloads.base import DONE_RE

        for raw in self.proc.stdout:
            now, line = time.time(), raw.rstrip("\n")
            self.lines.append((now, line))
            m = DONE_RE.match(line)
            if m:
                self.wrapped_rc, self.done_at = int(m.group(1)), now
                self.done.set()
            else:
                self.on_line(now, line)
        self.done.set()

    def release(self) -> None:
        """Let the wrapper exit (after the final cgroup read)."""
        try:
            self.proc.stdin.write("\n")
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass

    def kill(self) -> None:
        if self.cid:
            subprocess.run(["docker", "kill", self.cid], capture_output=True)

    def wait(self, timeout: float = 60) -> int | None:
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            return self.proc.wait(timeout=30)

    def remove(self) -> None:
        if self.cid:
            subprocess.run(["docker", "rm", "-f", self.cid], capture_output=True)

    @property
    def scope_rel(self) -> str | None:
        return "/" + str(self.scope.relative_to(CGROUP_ROOT)) if self.scope else None
