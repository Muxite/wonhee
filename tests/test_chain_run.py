import hashlib

import pytest
from pydantic import BaseModel, ConfigDict

from wonhee import media, service
from wonhee.db import queries, repo
from wonhee.nodes.base import Node
from wonhee.nodes.chain import Chain, parse_chain
from wonhee.nodes.runner import stages_for_chain
from wonhee.pipeline.orchestrator import run_stages
from wonhee.pipeline.registry import stages_for_run


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TagParams(NoParams):
    tag: str = "x"


class State:
    calls: dict[str, int]
    counter = 0
    boom = False
    cost = 0.0
    weak = False


@pytest.fixture
def toys(toy_registry):
    s = State()
    s.calls, s.counter, s.boom, s.cost, s.weak = {}, 0, False, 0.0, False

    def bump(name):
        s.calls[name] = s.calls.get(name, 0) + 1

    def src_fn(ctx, ins, p):
        bump("src")
        return {"out": sha(f"src-{s.counter}")}

    def upper_fn(ctx, ins, p):
        bump("upper")
        if s.cost:
            ctx.add_cost(s.cost)
        if s.weak:
            ctx.weak = True
            ctx.flag("NO_TEXT", "toy")
        ref = sha("upper:" + ins["in"] + p.tag)
        dest = media.path_for(ctx.settings.media_dir, ref)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return {"out": ref}

    def boom_fn(ctx, ins, p):
        bump("boom")
        if s.boom:
            raise RuntimeError("kaboom")
        return {"out": sha("boom:" + ins["in"])}

    def bad_fn(ctx, ins, p):
        return {"wrong": sha("x")}

    toy_registry(Node("src", 1, {}, {"out": "text"}, NoParams, src_fn, "toy", cacheable=False))
    s.upper = toy_registry(Node("upper", 1, {"in": "text"}, {"out": "text"}, TagParams, upper_fn, "toy"))
    toy_registry(Node("boom", 1, {"in": "text"}, {"out": "text"}, NoParams, boom_fn, "toy"))
    toy_registry(Node("bad", 1, {"in": "text"}, {"out": "text"}, NoParams, bad_fn, "toy"))
    return s


def chain_of(nodes, **kw) -> Chain:
    return parse_chain({"nodes": nodes, **kw})


def start(conn, settings, chain, cap=1.0):
    iid = repo.create_intent(conn, "toy", "chain", 1, chain.model_dump(by_alias=True, mode="json"))
    repo.confirm_intent(conn, iid)
    pid = repo.create_project(conn, "toy chain", "chain", iid)
    return repo.start_run(conn, pid, cap)


def go(conn, settings, chain, run_id, **kw):
    return run_stages(conn, settings, None, run_id, stages_for_chain(chain), **kw)


AB = [{"id": "a", "node": "src"}, {"id": "b", "node": "upper", "in": {"in": "a.out"}},
      {"id": "c", "node": "boom", "in": {"in": "b.out"}}]


def test_chain_round_trips_through_dump():
    c = chain_of(AB, no_cache=True)
    assert Chain.model_validate_json(__import__("json").dumps(c.model_dump(by_alias=True, mode="json"))) == c


def test_happy_path_records_outputs_and_wiring(conn, settings, toys):
    chain = chain_of(AB)
    rid = start(conn, settings, chain)
    assert go(conn, settings, chain, rid) == "done"
    a = queries.step_outputs(conn, queries.current_step(conn, rid, "a")["id"])["out"]["ref"]
    b = queries.step_outputs(conn, queries.current_step(conn, rid, "b")["id"])["out"]
    assert a == sha("src-0")
    assert b["ref"] == sha("upper:" + a + "x") and b["type"] == "text"
    cstep = queries.current_step(conn, rid, "c")
    assert queries.step_outputs(conn, cstep["id"])["out"]["ref"] == sha("boom:" + b["ref"])


def test_stages_for_run_dispatches_on_chain(conn, settings, toys):
    chain = chain_of(AB)
    rid = start(conn, settings, chain)
    st = stages_for_run(conn, rid)
    assert [s.name for s in st] == ["a", "b", "c"]
    assert st[1].needs == ("a",)
    assert service._stages_of_run(conn, rid)[2].name == "c"


def test_node_params_recorded_on_step(conn, settings, toys):
    chain = chain_of([{"id": "a", "node": "src"},
                      {"id": "b", "node": "upper", "in": {"in": "a.out"}, "params": {"tag": "Q"}}])
    rid = start(conn, settings, chain)
    go(conn, settings, chain, rid)
    row = queries.current_step(conn, rid, "b")
    assert __import__("json").loads(row["params_json"])["tag"] == "Q"


def test_failure_stops_downstream(conn, settings, toys):
    toys.boom = True
    chain = chain_of(AB + [{"id": "d", "node": "upper", "in": {"in": "c.out"}}])
    rid = start(conn, settings, chain)
    assert go(conn, settings, chain, rid) == "failed"
    assert queries.current_step(conn, rid, "c")["status"] == "failed"
    assert queries.current_step(conn, rid, "d") is None


def test_bad_node_output_keys_fail(conn, settings, toys):
    chain = chain_of([{"id": "a", "node": "src"}, {"id": "x", "node": "bad", "in": {"in": "a.out"}}])
    rid = start(conn, settings, chain)
    assert go(conn, settings, chain, rid) == "failed"
    flags = queries.step_flags(conn, queries.current_step(conn, rid, "x")["id"])
    assert any(f["code"] == "EXCEPTION" for f in flags)


def test_resume_reads_current_upstream_outputs(conn, settings, toys):
    chain = chain_of(AB)
    rid = start(conn, settings, chain)
    go(conn, settings, chain, rid)
    old_a = queries.step_outputs(conn, queries.current_step(conn, rid, "a")["id"])["out"]["ref"]
    toys.counter = 1
    toys.boom = True
    assert service.resume(conn, settings, None, rid, "a", {}) is not None
    # a re-ran (new attempt), b ran against it, c failed
    new_a = queries.step_outputs(conn, queries.current_step(conn, rid, "a")["id"])["out"]["ref"]
    assert new_a == sha("src-1") != old_a
    assert queries.current_step(conn, rid, "c")["status"] == "failed"
    toys.boom = False
    toys.calls.clear()
    # resume from b: must read the current a (src-1), not the superseded src-0
    service.resume(conn, settings, None, rid, "b", {})
    b = queries.step_outputs(conn, queries.current_step(conn, rid, "b")["id"])["out"]["ref"]
    assert b == sha("upper:" + new_a + "x")
    assert b != sha("upper:" + old_a + "x")
    assert "src" not in toys.calls
    assert queries.current_step(conn, rid, "c")["status"] == "ok"


def test_cache_hit_is_free_and_flagged(conn, settings, toys):
    toys.cost = 0.25
    chain = chain_of([{"id": "a", "node": "src"}, {"id": "b", "node": "upper", "in": {"in": "a.out"}}])
    r1 = start(conn, settings, chain)
    go(conn, settings, chain, r1)
    assert toys.calls["upper"] == 1 and queries.run_cost(conn, r1) == 0.25
    r2 = start(conn, settings, chain)
    go(conn, settings, chain, r2)
    assert toys.calls["upper"] == 1
    assert queries.run_cost(conn, r2) == 0
    step = queries.current_step(conn, r2, "b")
    assert any(f["code"] == "CACHED" for f in queries.step_flags(conn, step["id"]))
    assert queries.step_outputs(conn, step["id"]).keys() == {"out"}


def test_cache_skips_weak_and_version_bump(conn, settings, toys, toy_registry):
    chain = chain_of([{"id": "a", "node": "src"}, {"id": "b", "node": "upper", "in": {"in": "a.out"}}])
    toys.weak = True
    r1 = start(conn, settings, chain)
    assert go(conn, settings, chain, r1) == "weak"
    assert queries.current_step(conn, r1, "b")["status"] == "weak"
    toys.weak = False
    r2 = start(conn, settings, chain)
    assert go(conn, settings, chain, r2) == "done"
    assert toys.calls["upper"] == 2  # weak result was not cached
    r3 = start(conn, settings, chain)
    go(conn, settings, chain, r3)
    assert toys.calls["upper"] == 2  # ok result was
    from dataclasses import replace
    toy_registry(replace(toys.upper, version=2))
    r4 = start(conn, settings, chain)
    go(conn, settings, chain, r4)
    assert toys.calls["upper"] == 3  # version bump misses


@pytest.mark.parametrize("how", ["chain", "node"])
def test_no_cache_flags_skip_cache(conn, settings, toys, how):
    nodes = [{"id": "a", "node": "src"},
             {"id": "b", "node": "upper", "in": {"in": "a.out"}, **({"cache": False} if how == "node" else {})}]
    chain = chain_of(nodes, **({"no_cache": True} if how == "chain" else {}))
    for _ in range(2):
        go(conn, settings, chain, start(conn, settings, chain))
    assert toys.calls["upper"] == 2


def test_cost_cap_stops_chain(conn, settings, toys):
    toys.cost = 0.5
    chain = chain_of(AB)
    rid = start(conn, settings, chain, cap=0.4)
    assert go(conn, settings, chain, rid) == "stopped"
    assert queries.current_step(conn, rid, "c") is None


def test_no_cache_run_still_stores_for_later(conn, settings, toys):
    nodes = [{"id": "a", "node": "src"}, {"id": "b", "node": "upper", "in": {"in": "a.out"}}]
    go(conn, settings, chain_of(nodes, no_cache=True), start(conn, settings, chain_of(nodes, no_cache=True)))
    assert toys.calls["upper"] == 1
    chain = chain_of(nodes)
    rid = start(conn, settings, chain)
    go(conn, settings, chain, rid)
    assert toys.calls["upper"] == 1
    step = queries.current_step(conn, rid, "b")
    assert any(f["code"] == "CACHED" for f in queries.step_flags(conn, step["id"]))


def test_scout_stage_params_json_is_overrides(conn, settings):
    import json
    from wonhee.pipeline.context import Stage
    iid = repo.create_intent(conn, "x", "scout", 1, {"niche": "n"})
    repo.confirm_intent(conn, iid)
    rid = repo.start_run(conn, repo.create_project(conn, "t", "scout", iid), 1.0)
    run_stages(conn, settings, None, rid, [Stage("one", (), lambda ctx: "ok")])
    assert json.loads(queries.current_step(conn, rid, "one")["params_json"]) == {}


def test_resume_no_cache_override_reruns_node(conn, settings, toys):
    chain = chain_of([{"id": "a", "node": "src"}, {"id": "b", "node": "upper", "in": {"in": "a.out"}}])
    rid = start(conn, settings, chain)
    go(conn, settings, chain, rid)
    toys.counter = 0  # src is uncacheable but deterministic now, so b's cache key is unchanged
    service.resume(conn, settings, None, rid, "a", {})
    assert toys.calls["upper"] == 1  # served from cache
    service.resume(conn, settings, None, rid, "a", {"no_cache": "true"})
    assert toys.calls["upper"] == 2


def test_salt_change_misses_cache(conn, settings, toys, toy_registry):
    from dataclasses import replace
    toy_registry(replace(toys.upper, salt=lambda s: {"llm_model": s.llm_model}))
    chain = chain_of([{"id": "a", "node": "src"}, {"id": "b", "node": "upper", "in": {"in": "a.out"}}])
    go(conn, settings, chain, start(conn, settings, chain))
    go(conn, settings, chain, start(conn, settings, chain))
    assert toys.calls["upper"] == 1
    settings.llm_model = "other/model"
    go(conn, settings, chain, start(conn, settings, chain))
    assert toys.calls["upper"] == 2


def test_real_nodes_salt_their_models(settings):
    from wonhee.nodes import registry
    assert registry.get("translate").salt(settings) == {"llm_model": settings.llm_model}
    assert registry.get("screen_text").salt(settings) == {"llm_model": settings.llm_model}
    assert registry.get("speech_subs").salt(settings) == {
        "whisper_model": settings.whisper_model, "compute_type": settings.whisper_compute_type}


def test_cache_miss_when_cached_file_is_gone(conn, settings, toys):
    chain = chain_of([{"id": "a", "node": "src"}, {"id": "b", "node": "upper", "in": {"in": "a.out"}}])
    go(conn, settings, chain, start(conn, settings, chain))
    ref = queries.step_outputs(conn, queries.current_step(conn, 1, "b")["id"])["out"]["ref"]
    media.path_for(settings.media_dir, ref).unlink()
    go(conn, settings, chain, start(conn, settings, chain))
    assert toys.calls["upper"] == 2


def test_resume_chain_rejects_scout_only_keys(conn, settings, toys):
    chain = chain_of(AB)
    rid = start(conn, settings, chain)
    go(conn, settings, chain, rid)
    with pytest.raises(ValueError) as e:
        service.resume(conn, settings, None, rid, "a", {"count": 3})
    assert "unknown --set key(s) count" in str(e.value) and "no_cache, cost_cap_usd, force" in str(e.value)
