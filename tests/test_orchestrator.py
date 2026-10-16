import pytest
from wonhee.db import queries, repo
from wonhee.pipeline.context import Stage, StageFailed
from wonhee.pipeline.orchestrator import run_stages
from tests.fakes import fake_deps

REQ = {"mode": "discover", "niche": "x", "window": {"start": "2026-09-15", "end": "2026-10-15", "label": "now"}, "constraints": {}}


@pytest.fixture
def run_id(conn):
    i = repo.create_intent(conn, "x", "scout", 1, REQ)
    repo.confirm_intent(conn, i)
    p = repo.create_project(conn, "x", "scout", i)
    return repo.start_run(conn, p, 1.0)


def ok(ctx): return "ok"
def weak(ctx): ctx.flag("LOW_SAMPLE", "2 of 5"); return "weak"
def boom(ctx): raise ValueError("kaboom")
def fail(ctx): raise StageFailed("NO_CANDIDATES", "nothing in window")
def spend(ctx): ctx.add_cost(0.8); return "ok"


def test_happy_path_links_inputs(conn, settings, run_id):
    stages = [Stage("a", (), ok), Stage("b", ("a",), ok)]
    assert run_stages(conn, settings, fake_deps(), run_id, stages) == "done"
    b = queries.current_step(conn, run_id, "b")["id"]
    a = queries.current_step(conn, run_id, "a")["id"]
    assert conn.execute("SELECT input_step_id FROM step_input WHERE step_id=?", (b,)).fetchone()[0] == a


def test_weak_propagates(conn, settings, run_id):
    assert run_stages(conn, settings, fake_deps(), run_id, [Stage("a", (), weak)]) == "weak"


def test_exception_recorded_with_traceback(conn, settings, run_id):
    assert run_stages(conn, settings, fake_deps(), run_id, [Stage("a", (), boom), Stage("b", ("a",), ok)]) == "failed"
    s = queries.current_step(conn, run_id, "a")
    assert s["status"] == "failed"
    assert [f["code"] for f in queries.step_flags(conn, s["id"])] == ["EXCEPTION"]
    assert [a["role"] for a in queries.step_artifacts(conn, s["id"])] == ["error"]
    assert queries.current_step(conn, run_id, "b") is None


def test_stage_failed_flag(conn, settings, run_id):
    run_stages(conn, settings, fake_deps(), run_id, [Stage("a", (), fail)])
    s = queries.current_step(conn, run_id, "a")
    assert queries.step_flags(conn, s["id"])[0]["detail"] == "nothing in window"


def test_cost_cap_stops(conn, settings, run_id):
    stages = [Stage("a", (), spend), Stage("b", (), spend), Stage("c", (), ok)]
    assert run_stages(conn, settings, fake_deps(), run_id, stages) == "stopped"
    assert queries.current_step(conn, run_id, "c") is None
    b = queries.current_step(conn, run_id, "b")
    assert [f["code"] for f in queries.step_flags(conn, b["id"])] == ["COST_CAP"]


def test_resume_supersedes_and_links_current(conn, settings, run_id):
    stages = [Stage("a", (), ok), Stage("b", ("a",), ok), Stage("c", ("b",), ok)]
    run_stages(conn, settings, fake_deps(), run_id, stages)
    old_b = queries.current_step(conn, run_id, "b")["id"]
    old_c = queries.current_step(conn, run_id, "c")["id"]
    assert run_stages(conn, settings, fake_deps(), run_id, stages, start_at="b", overrides={"count": 8}) == "done"
    new_b = queries.current_step(conn, run_id, "b")
    new_c = queries.current_step(conn, run_id, "c")
    assert new_b["supersedes_step"] == old_b and new_c["supersedes_step"] == old_c
    assert '"count": 8' in new_b["params_json"]
    assert conn.execute("SELECT input_step_id FROM step_input WHERE step_id=?", (new_c["id"],)).fetchone()[0] == new_b["id"]


def test_resume_requires_earlier_stages(conn, settings, run_id):
    with pytest.raises(ValueError, match="no completed step for 'a'"):
        run_stages(conn, settings, fake_deps(), run_id, [Stage("a", (), ok), Stage("b", ("a",), ok)], start_at="b")


def test_progress_callback(conn, settings, run_id):
    lines = []
    run_stages(conn, settings, fake_deps(), run_id, [Stage("a", (), ok), Stage("b", ("a",), weak)],
               progress=lines.append)
    assert lines == ["stage 1/2 a…", "  a: ok", "stage 2/2 b…", "  b: weak"]


def test_keyboard_interrupt_marks_run_stopped_and_reraises(conn, settings, run_id):
    def ctrl_c(ctx): raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_stages(conn, settings, fake_deps(), run_id, [Stage("a", (), ctrl_c)])
    s = queries.current_step(conn, run_id, "a")
    assert s["status"] == "failed"
    (f,) = queries.step_flags(conn, s["id"])
    assert (f["code"], f["detail"]) == ("EXCEPTION", "interrupted")
    assert queries.run_row(conn, run_id)["status"] == "stopped"
