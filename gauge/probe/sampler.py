"""Fixed-interval sampler for one container scope. Keeps rows in memory; the runner persists them."""
from __future__ import annotations

import threading
import time

from gauge.probe.cgroup import Cgroup
from gauge.probe.gpu import Gpu


class Sampler(threading.Thread):
    def __init__(self, cgroup: Cgroup, scope_rel: str, gpu: Gpu, t0: float, interval_s: float = 0.1):
        super().__init__(name="gauge-sampler", daemon=True)
        self.cg, self.scope_rel, self.gpu = cgroup, scope_rel, gpu
        self.t0, self.interval = t0, interval_s
        self.rows: list[dict] = []
        self.missed = 0
        self.gpu_pids: set[int] = set()
        self.gone = False  # cgroup vanished (container exited under us)
        self._halt = threading.Event()
        self.cpu_s = 0.0  # this thread's own CPU time: collector overhead
        self.last_full: dict | None = None  # fallback end snapshot if the cgroup dies before the final read
        self.full_every = max(1, int(1.0 / interval_s))

    def run(self) -> None:
        c0 = time.thread_time()
        nxt = time.monotonic()
        while not self._halt.is_set():
            if not self.sample():
                self.gone = True
                break
            nxt += self.interval
            lag = time.monotonic() - nxt
            if lag > 0:  # a read took longer than the interval: count the ticks we skipped
                skipped = int(lag // self.interval) + 1
                self.missed += skipped
                nxt += skipped * self.interval
            self._halt.wait(max(0.0, nxt - time.monotonic()))
        self.cpu_s = time.thread_time() - c0

    def sample(self) -> bool:
        row = self.cg.fast()
        if row is None:
            return False
        row["t"] = time.time() - self.t0
        if len(self.rows) % self.full_every == 0:
            self.last_full = self.cg.full() or self.last_full
        if self.gpu.ok:
            vram, pids = self.gpu.scope_vram(self.scope_rel)
            row["vram_proc_bytes"] = vram
            self.gpu_pids.update(pids)
            row.update({k: v for k, v in self.gpu.device().items() if k != "dev_mem_free"})
        self.rows.append(row)
        return True

    def stop(self) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(timeout=5)
