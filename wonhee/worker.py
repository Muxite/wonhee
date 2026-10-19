"""Warm workers: one long-lived process per lane keeps its models loaded and runs chain runs from a queue.

Why: a cold GPU node pays the model load on every call (whisper ~5 s, Grounding-DINO + SAM ~10 s, demucs ~5 s,
SigLIP2 ~7 s), and the first call after loading is about 2x slower than later ones. Measured with
`python -m wonhee.bench` (docs/production/workers.md).

    wonhee worker serve --lane cuda --preload whisper,mask     # foreground; one GPU thread
    wonhee node run speech_subs --in video=clip.mp4 --detach    # queued for it while it is alive

How it works:
- `chain_run(wait=False)` enqueues a `job` when the chain has a node with that lane AND a worker of the lane has
  heartbeated within STALE_S. Otherwise it spawns a cold child process, as before.
- The worker claims a job atomically (BEGIN IMMEDIATE + UPDATE ... RETURNING), runs `service.chain_exec` with its
  own long-lived Deps, and records done/failed. A heartbeat thread (its own connection) refreshes the worker row
  and the job every HEARTBEAT_S.
- A job whose heartbeat is older than STALE_S is reclaimed by the next claim: its running steps are marked failed
  (flag WORKER_LOST) and it is requeued, up to MAX_ATTEMPTS. Then the run fails, like any other failed run.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field

from wonhee import marks
from wonhee.db import repo
from wonhee.db.connect import open_db, tx

LANES = ("cuda",)
HEARTBEAT_S = 10
STALE_S = 60
MAX_ATTEMPTS = 3


def lane_of(node_names: list[str]) -> str | None:
    """The lane a chain needs: the first non-"none" lane among its nodes."""
    from wonhee.nodes import registry
    for n in node_names:
        lane = getattr(registry.get(n), "lane", "none")
        if lane != "none":
            return lane
    return None


def live(conn, lane: str, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return conn.execute("SELECT 1 FROM worker WHERE lane=? AND heartbeat_at > ? LIMIT 1",
                        (lane, now - STALE_S)).fetchone() is not None


def enqueue(conn, run_id: int, lane: str) -> int:
    return conn.execute("INSERT INTO job(run_id, lane, status, created_at) VALUES (?,?, 'queued', ?)",
                        (run_id, lane, time.time())).lastrowid


def _orphan_steps(conn, run_id: int, detail: str) -> None:
    for (sid,) in conn.execute("SELECT id FROM step WHERE run_id=? AND status='running'", (run_id,)).fetchall():
        repo.add_flag(conn, sid, "WORKER_LOST", detail[:300])
        repo.finish_step(conn, sid, "failed")


def reclaim(conn, lane: str, now: float | None = None) -> list[int]:
    """Requeue (or fail, after MAX_ATTEMPTS) jobs whose worker stopped heartbeating. Call inside tx()."""
    now = time.time() if now is None else now
    stale = conn.execute("SELECT id, run_id, worker, attempts FROM job WHERE lane=? AND status='running'"
                         " AND coalesce(heartbeat_at, claimed_at) < ?", (lane, now - STALE_S)).fetchall()
    for j in stale:
        _orphan_steps(conn, j["run_id"], f"worker {j['worker']} lost during job {j['id']}")
        if j["attempts"] >= MAX_ATTEMPTS:
            conn.execute("UPDATE job SET status='failed', finished_at=?, error=? WHERE id=?",
                         (now, f"worker lost {j['attempts']} times", j["id"]))
            repo.finish_run(conn, j["run_id"], "failed")
        else:
            conn.execute("UPDATE job SET status='queued', worker=NULL WHERE id=?", (j["id"],))
    return [j["id"] for j in stale]


def claim(conn, lane: str, name: str, now: float | None = None):
    """The oldest queued job of the lane, now running under `name`; None when the queue is empty."""
    now = time.time() if now is None else now
    with tx(conn):
        reclaim(conn, lane, now)
        return conn.execute(
            "UPDATE job SET status='running', worker=?, attempts=attempts+1, claimed_at=?, heartbeat_at=?"
            " WHERE id=(SELECT id FROM job WHERE lane=? AND status='queued' ORDER BY id LIMIT 1)"
            " RETURNING id, run_id, attempts", (name, now, now, lane)).fetchone()


def finish(conn, job_id: int, ok: bool, error: str | None = None) -> None:
    conn.execute("UPDATE job SET status=?, finished_at=?, error=? WHERE id=?",
                 ("done" if ok else "failed", time.time(), error, job_id))


def status(conn) -> str:
    now = time.time()
    lines = []
    for w in conn.execute("SELECT * FROM worker ORDER BY lane, name"):
        age = now - w["heartbeat_at"]
        state = "live" if age < STALE_S else "gone"
        lines.append(f"{w['name']}  lane={w['lane']}  {state} ({age:.0f}s ago)  pid={w['pid']}  "
                     f"models={','.join(json.loads(w['models'])) or '-'}  jobs_done={w['jobs_done']}"
                     + (f"  on job {w['current_job']}" if w["current_job"] else ""))
    counts = conn.execute("SELECT lane, status, COUNT(*) n FROM job GROUP BY lane, status ORDER BY lane, status")
    q = ", ".join(f"{r['lane']}:{r['status']}={r['n']}" for r in counts)
    return "\n".join(lines or ["no workers have registered"]) + f"\njobs: {q or 'none'}"


# --- the serving loop --------------------------------------------------------------------------------------------

def _preloaders(deps) -> dict:
    def mask():
        from wonhee.edit import mask as mk
        mk._gpu_models()

    def separate():
        from wonhee.edit import separate as sep
        sep._separator("htdemucs")

    def embed():
        from wonhee.analyze import embed as em
        em.load()

    return {"whisper": lambda: deps.transcriber.ensure_ready(), "mask": mask, "separate": separate, "embed": embed}


@dataclass
class Worker:
    settings: object
    lane: str = "cuda"
    preload: list[str] = field(default_factory=list)
    name: str = field(default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}")
    poll_s: float = 1.0
    stop: threading.Event = field(default_factory=threading.Event)
    current: int | None = None
    done: int = 0

    def _beat(self) -> None:
        conn = open_db(self.settings)  # sqlite connections stay on their own thread
        try:
            while not self.stop.wait(HEARTBEAT_S):
                now, job = time.time(), self.current
                conn.execute("UPDATE worker SET heartbeat_at=?, current_job=?, jobs_done=? WHERE name=?",
                             (now, job, self.done, self.name))
                if job is not None:
                    conn.execute("UPDATE job SET heartbeat_at=? WHERE id=? AND worker=?", (now, job, self.name))
        finally:
            conn.close()

    def serve(self, deps, *, max_jobs: int | None = None, drain: bool = False, progress=None) -> int:
        """Run jobs until stopped, max_jobs are done, or (drain) the queue is empty. Returns jobs done."""
        from wonhee import service

        say = progress or (lambda s: None)
        conn = open_db(self.settings)
        now = time.time()
        conn.execute("INSERT INTO worker(name, lane, pid, models, started_at, heartbeat_at) VALUES (?,?,?,?,?,?)"
                     " ON CONFLICT(name) DO UPDATE SET lane=excluded.lane, pid=excluded.pid, models=excluded.models,"
                     " started_at=excluded.started_at, heartbeat_at=excluded.heartbeat_at, jobs_done=0,"
                     " current_job=NULL", (self.name, self.lane, os.getpid(), json.dumps(self.preload), now, now))
        loaders = _preloaders(deps)
        for m in self.preload:
            t = time.perf_counter()
            loaders[m]()
            say(f"loaded {m} in {time.perf_counter() - t:.1f}s")
        conn.execute("UPDATE worker SET loaded_at=? WHERE name=?", (time.time(), self.name))
        marks.mark("model_loaded")
        beat = threading.Thread(target=self._beat, daemon=True)
        beat.start()
        say(f"worker {self.name} serving lane {self.lane}")
        try:
            while not self.stop.is_set() and (max_jobs is None or self.done < max_jobs):
                job = claim(conn, self.lane, self.name)
                if job is None:
                    if drain:
                        break
                    self.stop.wait(self.poll_s)
                    continue
                self.current = job["id"]
                marks.mark("compute_started")
                t = time.perf_counter()
                try:
                    service.chain_exec(conn, self.settings, deps, job["run_id"])
                    row = conn.execute("SELECT status FROM run WHERE id=?", (job["run_id"],)).fetchone()
                    finish(conn, job["id"], row["status"] in ("done", "weak"),
                           None if row["status"] in ("done", "weak") else f"run {row['status']}")
                except Exception as e:  # noqa: BLE001 - one bad job must not take the worker down
                    finish(conn, job["id"], False, f"{type(e).__name__}: {e}"[:300])
                marks.mark("compute_finished")
                self.done += 1
                self.current = None
                conn.execute("UPDATE worker SET jobs_done=?, current_job=NULL, heartbeat_at=? WHERE name=?",
                             (self.done, time.time(), self.name))
                say(f"job {job['id']} (run {job['run_id']}) in {time.perf_counter() - t:.1f}s")
        finally:
            self.stop.set()
            beat.join(timeout=HEARTBEAT_S + 5)
            conn.execute("UPDATE worker SET heartbeat_at=0, current_job=NULL WHERE name=?", (self.name,))  # gone
            conn.close()
        return self.done
