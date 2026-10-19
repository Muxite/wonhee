import time

import pytest

from tests.fakes import fake_deps
from tests.test_chain_run import chain_of, start
from tests.test_nodes_subs import CountingTranscriber, store_video
from wonhee import service, worker
from wonhee.db import queries, repo
from wonhee.db.connect import open_db


def _run(conn, settings, video_sha):
    return start(conn, settings, chain_of([{"id": "s", "node": "speech_subs", "in": {"video": video_sha}}]))


def _beat(conn, name, lane="cuda", at=None):
    conn.execute("INSERT OR REPLACE INTO worker(name, lane, pid, started_at, heartbeat_at) VALUES (?,?,1,?,?)",
                 (name, lane, at or time.time(), at or time.time()))


def test_lane_of_finds_the_gpu_node():
    assert worker.lane_of(["to_srt", "speech_subs"]) == "cuda"
    assert worker.lane_of(["to_srt"]) is None


def test_claim_is_exclusive(conn, settings):
    job = worker.enqueue(conn, _run(conn, settings, "a" * 64), "cuda")
    other = open_db(settings)
    try:
        got = [worker.claim(conn, "cuda", "w1"), worker.claim(other, "cuda", "w2")]
    finally:
        other.close()
    assert [g["id"] if g else None for g in got] == [job, None]
    assert conn.execute("SELECT worker, attempts FROM job WHERE id=?", (job,)).fetchone()[:] == ("w1", 1)


def test_stale_job_is_reclaimed_and_its_steps_failed(conn, settings):
    rid = _run(conn, settings, "a" * 64)
    job = worker.enqueue(conn, rid, "cuda")
    t0 = time.time()
    worker.claim(conn, "cuda", "w1", now=t0)
    sid = repo.start_step(conn, rid, "s", {}, None)  # w1 died mid-step
    again = worker.claim(conn, "cuda", "w2", now=t0 + worker.STALE_S + 1)
    assert (again["id"], again["attempts"]) == (job, 2)
    step = conn.execute("SELECT status FROM step WHERE id=?", (sid,)).fetchone()
    assert step["status"] == "failed"
    assert [f["code"] for f in queries.step_flags(conn, sid)] == ["WORKER_LOST"]


def test_job_fails_after_max_attempts(conn, settings):
    rid = _run(conn, settings, "a" * 64)
    job = worker.enqueue(conn, rid, "cuda")
    t = time.time()
    for i in range(worker.MAX_ATTEMPTS):
        assert worker.claim(conn, "cuda", f"w{i}", now=t)["id"] == job
        t += worker.STALE_S + 1
    assert worker.claim(conn, "cuda", "last", now=t) is None
    assert conn.execute("SELECT status FROM job WHERE id=?", (job,)).fetchone()[0] == "failed"
    assert queries.run_row(conn, rid)["status"] == "failed"


def test_detached_gpu_chain_goes_to_a_live_worker(conn, settings, monkeypatch, tiny_video):
    from wonhee.nodes import detach
    monkeypatch.setattr(detach, "spawn", lambda *a: pytest.fail("spawned a cold child despite a live worker"))
    _beat(conn, "w1")
    sha = store_video(conn, settings, tiny_video)
    rid, text = service.node_run(conn, settings, None, "speech_subs", {"video": sha}, wait=False)
    assert "queued" in text
    assert conn.execute("SELECT run_id, status FROM job").fetchone()[:] == (rid, "queued")


@pytest.mark.parametrize("beat_age", [None, worker.STALE_S + 5])  # no worker, or one that stopped heartbeating
def test_detached_chain_spawns_without_a_live_worker(conn, settings, monkeypatch, tmp_path, tiny_video, beat_age):
    from wonhee.nodes import detach
    spawned = []
    monkeypatch.setattr(detach, "spawn", lambda s, rid: spawned.append(rid) or tmp_path / "log")
    if beat_age is not None:
        _beat(conn, "old", at=time.time() - beat_age)
    sha = store_video(conn, settings, tiny_video)
    rid, _ = service.node_run(conn, settings, None, "speech_subs", {"video": sha}, wait=False)
    assert spawned == [rid] and conn.execute("SELECT COUNT(*) FROM job").fetchone()[0] == 0


def test_serve_runs_queued_jobs_with_one_set_of_deps(conn, settings, tiny_video):
    sha = store_video(conn, settings, tiny_video)
    runs = [_run(conn, settings, sha) for _ in range(2)]
    for rid in runs:
        worker.enqueue(conn, rid, "cuda")
    tr = CountingTranscriber()
    w = worker.Worker(settings, name="t1")
    assert w.serve(fake_deps(transcriber=tr), drain=True) == 2
    assert [queries.run_row(conn, r)["status"] for r in runs] == ["done", "done"]
    assert [r[0] for r in conn.execute("SELECT status FROM job ORDER BY id")] == ["done", "done"]
    assert tr.n == 1  # the second run hit the node cache: same video, same model
    row = conn.execute("SELECT jobs_done, heartbeat_at FROM worker WHERE name='t1'").fetchone()
    assert row["jobs_done"] == 2 and row["heartbeat_at"] == 0  # marked gone on exit
    assert "t1" in worker.status(conn) and "cuda:done=2" in worker.status(conn)


def test_a_failing_job_does_not_stop_the_worker(conn, settings, monkeypatch):
    for rid in (_run(conn, settings, "a" * 64), _run(conn, settings, "b" * 64)):
        worker.enqueue(conn, rid, "cuda")

    def boom(*a, **k):
        raise RuntimeError("cuda fell over")
    monkeypatch.setattr(service, "chain_exec", boom)
    assert worker.Worker(settings, name="t2").serve(fake_deps(), drain=True) == 2
    rows = conn.execute("SELECT status, error FROM job ORDER BY id").fetchall()
    assert [r["status"] for r in rows] == ["failed", "failed"] and "cuda fell over" in rows[0]["error"]
