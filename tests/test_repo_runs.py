import pytest
from wonhee.db import queries, repo


@pytest.fixture
def project(conn):
    conn.execute("INSERT INTO intent VALUES (1,'t','scout',1,'{}','now','now')")
    conn.execute("INSERT INTO project(id,title,kind,status,intent_id,created_at) VALUES (1,'p','scout','open',1,'now')")
    return 1


def test_step_attempts_and_supersede(conn, project):
    run = repo.start_run(conn, project, 1.0)
    s1 = repo.start_step(conn, run, "discover", {}, None)
    repo.finish_step(conn, s1, "ok")
    s2 = repo.start_step(conn, run, "discover", {"count": 8}, s1)
    assert conn.execute("SELECT attempt FROM step WHERE id=?", (s2,)).fetchone()[0] == 2
    assert queries.current_step(conn, run, "discover")["id"] == s2


def test_cost_accumulates(conn, project):
    run = repo.start_run(conn, project, 1.0)
    s = repo.start_step(conn, run, "fetch", {}, None)
    repo.add_cost(conn, s, 0.1)
    repo.add_cost(conn, s, 0.15)
    assert abs(queries.run_cost(conn, run) - 0.25) < 1e-9


def test_flag_twice_appends_detail(conn, project):
    run = repo.start_run(conn, project, 1.0)
    s = repo.start_step(conn, run, "fetch", {}, None)
    repo.add_flag(conn, s, "DOWNLOAD_FAILED", "reel A")
    repo.add_flag(conn, s, "DOWNLOAD_FAILED", "reel B")
    (f,) = queries.step_flags(conn, s)
    assert f["detail"] == "reel A; reel B"


def test_unknown_flag_rejected(conn, project):
    import sqlite3
    run = repo.start_run(conn, project, 1.0)
    s = repo.start_step(conn, run, "fetch", {}, None)
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_flag(conn, s, "NOT_A_FLAG")


def test_llm_call_for_intent(conn, project):
    cid = repo.record_llm_call(conn, intent_id=1, model="m", schema_name="X", tokens_in=1, tokens_out=2, cost_usd=0.0, valid=True)
    assert cid > 0
