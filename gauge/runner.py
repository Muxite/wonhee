"""predict -> commit -> launch -> measure -> summarize -> evaluate, for one job or a deliberate co-run group."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from gauge import evaluate, host, nsight, predict, store
from gauge.docker_run import Container, docker_argv, image_digest, inspect
from gauge.probe.cgroup import Cgroup
from gauge.probe.gpu import Gpu
from gauge.probe.sampler import Sampler
from gauge.spec import JobSpec
from gauge.summarize import classify, summarize
from gauge.workloads.base import CUDA_OOM_RE, MARK_RE, get, parse_outcome_line

GPU_LOCK = os.environ.get("GAUGE_GPU_LOCK", "/home/muk/projects/gpu-lock")


@contextmanager
def gpu_lock(what: str, enabled: bool, ttl_s: int = 7200):
    """Advisory machine-wide GPU lock, so other agents' GPU work doesn't land inside our measurements."""
    if not enabled or not Path(GPU_LOCK).exists():
        yield None
        return
    r = subprocess.run([GPU_LOCK, "acquire", what, "--wait", "--ttl", str(ttl_s)], capture_output=True, text=True)
    token = None
    try:
        token = json.loads(r.stdout[r.stdout.index("{"):]).get("token")
    except (ValueError, AttributeError):
        pass
    if r.returncode != 0 or not token:
        raise RuntimeError(f"gpu-lock acquire failed: {(r.stdout + r.stderr).strip()[:300]}")
    try:
        yield token
    finally:
        for _ in range(2):
            rr = subprocess.run([GPU_LOCK, "release", f"--token={token}"], capture_output=True, text=True)
            if rr.returncode == 0:
                break
        else:
            print(f"WARNING gpu-lock release failed: {(rr.stdout + rr.stderr).strip()[:300]}", file=sys.stderr)


class Job:
    """One job's lifecycle. Split into phases so co-runs can launch several before waiting on any."""

    def __init__(self, conn, spec: JobSpec, gpu: Gpu, *, interval_s=0.1, corun_group=None, host_fp=None):
        self.conn, self.spec, self.gpu, self.interval = conn, spec, gpu, interval_s
        self.w = get(spec.workload)
        self.p = self.w.validate(spec.params)
        self.corun_group = corun_group
        self.host_fp = host_fp
        self.marks: list[dict] = []
        self.outcome: dict = {}
        self.outcomes: list[dict] = []
        self.cuda_oom_reported = False
        self.log_cuda_oom = False
        self.run_id: int | None = None
        self.c: Container | None = None
        self.sampler: Sampler | None = None
        self.energy0 = None

    # -- phase 1: freeze predictions before anything launches
    def commit(self) -> int:
        w, p, spec = self.w, self.p, self.spec
        self.run_id = store.insert_run(
            self.conn, workload=w.name, family=w.family, spec_hash=spec.spec_hash(p.model_dump()), config_hash=w.config_hash(p),
            spec_json=spec.model_dump(), features_json=w.features(p), size=w.size(p), image=w.image,
            image_digest=image_digest(w.image), execution_mode=spec.execution_mode, warmness=spec.warmness,
            corun_group=self.corun_group, provenance_json=spec.provenance, host_json=self.host_fp)
        now = time.time()
        for pred in predict.predict_all(w, p, self.conn, now):
            store.insert_prediction(self.conn, self.run_id, pred)
        store.update_run(self.conn, self.run_id, prediction_committed=time.time())
        return self.run_id

    def _on_line(self, t: float, line: str) -> None:
        m = MARK_RE.match(line)
        if m:
            self.marks.append({"mark": m.group(1), "at": t})
            if m.group(1) == "cuda_oom":
                self.cuda_oom_reported = True
            return
        o = parse_outcome_line(line)
        if o is not None:
            self.outcomes.append(o)
            if len(self.outcomes) == 1:
                self.outcome.update(o)  # first response's fields at top level; all of them under "responses"
            else:
                self.outcome["responses"] = self.outcomes
            return
        if CUDA_OOM_RE.search(line):
            self.log_cuda_oom = True

    # -- phase 2: launch and start sampling
    def launch(self, overlapping: list[int]) -> None:
        w, spec = self.w, self.spec
        store.update_run(self.conn, self.run_id, initial_json=host.initial_state(self.gpu, overlapping),
                         status="running")
        name = f"gauge-{self.run_id}-{uuid.uuid4().hex[:6]}"
        cmd, mounts, caps = w.command(self.p), w.mounts + w.param_mounts(self.p), ()
        if spec.profile != "none":
            cmd, extra, caps = nsight.wrap(cmd, self.run_id, spec.profile)
            mounts = mounts + extra
        argv_fn = lambda cid: docker_argv(name, w.image, cmd, cidfile=cid, gpu=w.gpu,  # noqa: E731
                                          mem_limit=spec.mem_limit or w.mem_limit, mounts=mounts,
                                          env={**w.env, **w.param_env(self.p)}, shm_size=w.shm_size, cap_add=caps)
        self.c = Container(name, argv_fn, self._on_line)
        self.energy0 = self.gpu.energy_mj()
        self.t_req = time.time()
        store.update_run(self.conn, self.run_id, container_start_requested=self.t_req)
        self.c.start()
        store.update_run(self.conn, self.run_id, container_id=self.c.cid)
        if self.c.scope:
            self.sampler = Sampler(Cgroup(self.c.scope), self.c.scope_rel, self.gpu, t0=self.t_req,
                                   interval_s=self.interval)
            self.sampler.start()

    # -- phase 3: wait, final read while the cgroup is still held open, release, record
    def finish(self) -> dict:
        c, spec = self.c, self.spec
        timed_out = not c.done.wait(timeout=spec.timeout_s)
        energy1 = self.gpu.energy_mj()
        end = Cgroup(c.scope).full() if c.scope else None
        end_is_final = end is not None
        if self.sampler:
            self.sampler.stop()
            end = end or self.sampler.last_full
        if timed_out:
            c.kill()
        else:
            c.release()
        c.wait()
        t_fin = time.time()
        info = inspect(c.cid) if c.cid else {}
        c.remove()

        started = info.get("started_at") or self.t_req
        workload_end = c.done_at if c.done_at else (info.get("finished_at") or t_fin)
        exit_code = c.wrapped_rc if c.wrapped_rc is not None else info.get("exit_code")
        rows = self.sampler.rows if self.sampler else []
        for r in rows:  # re-base sample times on the container's own start
            r["t"] = r["t"] - (started - self.t_req)
        ev = (end or {}).get("memory_events", {})
        fclass, fsrc = classify(exit_code=exit_code, timed_out=timed_out, oom_kill_events=ev.get("oom_kill", 0),
                                docker_oom=bool(info.get("oom_killed")), workload_cuda_oom=self.cuda_oom_reported,
                                log_cuda_oom=self.log_cuda_oom)
        energy_j = (energy1 - self.energy0) / 1000 if self.energy0 is not None and energy1 is not None else None
        summary = summarize(rows=rows, end=end, end_is_final=end_is_final, workload_s=workload_end - started,
                            wall_s=t_fin - self.t_req, energy_j=energy_j, sampler=self.sampler,
                            gpu_caps=dict(self.gpu.capabilities), failure_class=fclass)
        summary["marks"] = [{"mark": m["mark"], "t": m["at"] - started} for m in self.marks]
        summary["log_tail"] = [l[:300] for _, l in c.lines[-12:]]
        store.insert_samples(self.conn, self.run_id, rows)
        store.update_run(self.conn, self.run_id, container_started=started, workload_finished=workload_end,
                         container_finished=t_fin, marks_json=summary["marks"], outcome_json=self.outcome,
                         exit_code=exit_code, termination_reason="timeout" if timed_out else "exited",
                         docker_oom_killed=int(bool(info.get("oom_killed"))), failure_class=fclass,
                         failure_source=fsrc, status="done")
        store.put_summary(self.conn, self.run_id, summary)
        evaluate.evaluate_run(self.conn, self.run_id, summary)
        if spec.profile != "none":
            db = nsight.export(self.run_id)
            if db:
                prof = nsight.summarize(db)
                nsight.store_profile(self.conn, self.run_id, prof)
                summary["gpu_profile"] = prof
        return summary


def run(conn, spec: JobSpec, *, interval_s=0.1, lock=True) -> tuple[int, dict]:
    gpu = Gpu()
    job = Job(conn, spec, gpu, interval_s=interval_s, host_fp=host.fingerprint(gpu))
    job.commit()
    with gpu_lock(f"gauge run {job.run_id} {spec.workload}", lock):
        try:
            job.launch(overlapping=[])
            summary = job.finish()
        except BaseException:
            store.update_run(conn, job.run_id, status="error")
            if job.c:
                job.c.kill()
                job.c.remove()
            raise
    return job.run_id, summary


def corun(conn, specs: list[JobSpec], *, interval_s=0.1, lock=True) -> list[tuple[int, dict]]:
    """Launch several jobs together. Device-scope metrics in these runs are shared, never per-job."""
    gpu = Gpu()
    fp = host.fingerprint(gpu)
    group = uuid.uuid4().hex[:10]
    jobs = [Job(conn, s, gpu, interval_s=interval_s, corun_group=group, host_fp=fp) for s in specs]
    ids = [j.commit() for j in jobs]
    with gpu_lock(f"gauge corun {group}", lock):
        for j in jobs:
            j.launch(overlapping=[i for i in ids if i != j.run_id])
        results: dict[int, dict] = {}
        threads = [threading.Thread(target=lambda j=j: results.__setitem__(j.run_id, j.finish())) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return [(i, results.get(i, {})) for i in ids]
