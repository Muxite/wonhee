import time

import pytest

from gauge import evaluate, predict, store
from gauge.evaluate import score
from gauge.predict import fit_affine
from gauge.summarize import classify, peak_cores
from gauge.workloads.base import get


def test_classify_order():
    base = dict(exit_code=0, timed_out=False, oom_kill_events=0, docker_oom=False, workload_cuda_oom=False,
                log_cuda_oom=False)
    assert classify(**base) == ("success", "exit_code")
    assert classify(**{**base, "timed_out": True})[0] == "timeout"
    assert classify(**{**base, "exit_code": 137, "oom_kill_events": 1})[0] == "host_oom"
    assert classify(**{**base, "exit_code": 3, "workload_cuda_oom": True}) == ("cuda_oom", "workload")
    assert classify(**{**base, "exit_code": 1, "log_cuda_oom": True}) == ("cuda_oom", "log-derived")
    assert classify(**{**base, "exit_code": 1})[0] == "application_error"


def test_peak_cores():
    rows = [{"t": 0, "cpu_usage_usec": 0}, {"t": 1, "cpu_usage_usec": 2_000_000}, {"t": 2, "cpu_usage_usec": 2_500_000}]
    assert peak_cores(rows) == pytest.approx(2.0)


def test_score_success_and_censored():
    s = score(actual=10, estimate=8, upper=12, censored=False)
    assert s["signed_error"] == 2 and s["covered"] == 1 and s["underprediction"] == 0 and s["excess_reservation"] == 2
    miss = score(actual=15, estimate=None, upper=12, censored=False)
    assert miss["covered"] == 0 and miss["underprediction"] == 3
    # an OOM at 15 under a bound of 20: we can't claim the bound covered the true need
    cen = score(actual=15, estimate=None, upper=20, censored=True)
    assert cen["covered"] is None and cen["excess_reservation"] is None
    assert score(actual=25, estimate=None, upper=20, censored=True)["covered"] == 0


def test_fit_affine():
    a, b, worst = fit_affine([1, 2, 3], [3, 5, 7.5])
    assert b == pytest.approx(2.25, rel=1e-6) and worst > 0


def _fake_run(conn, w, params, summary, fclass="success", finished=None):
    p = w.validate(params)
    rid = store.insert_run(conn, workload=w.name, family=w.family, spec_hash="x", config_hash=w.config_hash(p),
                           spec_json={"params": params}, features_json={}, size=w.size(p),
                           execution_mode="isolated", warmness="unknown", status="done", failure_class=fclass,
                           container_finished=finished or time.time() - 10)
    store.put_summary(conn, rid, summary)
    return rid


def test_predictors_use_only_prior_history():
    conn = store.connect(":memory:")
    w = get("train.small")
    now = time.time()
    p = w.validate({"batch": 64})
    assert predict.history_predictor(w, p, conn, now).status == "insufficient_data"
    assert predict.calibrated_predictor(w, p, conn, now).status == "insufficient_data"
    vram = lambda b: 1e9 + 5e7 * b  # noqa: E731
    for b in (16, 32):
        _fake_run(conn, w, {"batch": b}, {"vram_peak_bytes": vram(b), "runtime_s": 10 + b / 10})
    _fake_run(conn, w, {"batch": 128}, {"vram_peak_bytes": 1.5e10}, fclass="cuda_oom")  # censored: ignored
    _fake_run(conn, w, {"batch": 48}, {"vram_peak_bytes": 1}, finished=now + 100)  # after cutoff: ignored
    cal = predict.calibrated_predictor(w, p, conn, now)
    assert cal.status == "supported" and len(cal.evidence_run_ids) == 2
    assert cal.vram_peak_upper_bytes == pytest.approx(vram(64) * 1.1, rel=1e-6)
    hist = predict.history_predictor(w, w.validate({"batch": 32}), conn, now)
    assert hist.status == "supported" and hist.vram_peak_upper_bytes == pytest.approx(vram(32) * 1.1)


def test_predictions_immutable_and_evaluated():
    conn = store.connect(":memory:")
    w = get("stress.cpu")
    rid = _fake_run(conn, w, {"workers": 2, "seconds": 5}, {})
    pid = store.insert_prediction(conn, rid, w.prior_prediction(w.validate({"workers": 2, "seconds": 5})))
    with pytest.raises(Exception, match="immutable"):
        conn.execute("UPDATE prediction SET runtime_estimate_s=1 WHERE id=?", (pid,))
    evaluate.evaluate_run(conn, rid, {"runtime_s": 5.1, "cpu_time_s": 10.2, "ram_peak_bytes": 1e6, "censored": False})
    rows = {r["target"]: r for r in conn.execute("SELECT * FROM evaluation")}
    assert rows["cpu_time_s"]["signed_error"] == pytest.approx(0.2)
    assert rows["runtime_s"]["covered"] == 1


def test_llm_prior_kv_scales_with_ctx():
    w = get("llm.generate")
    a = w.prior(w.validate({"model": "qwen2.5:7b", "num_ctx": 4096}))["vram_peak_bytes"]
    b = w.prior(w.validate({"model": "qwen2.5:7b", "num_ctx": 8192}))["vram_peak_bytes"]
    assert b - a == 2 * 28 * 4 * 128 * 2 * 4096  # K+V, f16, GQA kv_heads
    # all models share one config, so calibration can pool across sizes
    assert w.config_hash(w.validate({"model": "qwen2.5:0.5b"})) == w.config_hash(w.validate({"model": "qwen2.5:14b"}))


def test_fit_affine_never_negative_intercept():
    # steep line through small sizes would give a < 0 and predict ~0 below the smallest size seen
    a, b, _ = fit_affine([10, 20, 40], [5, 40, 110])
    assert a == 0.0 and b > 0
    assert a + b * 5 > 0


def test_history_needs_identical_params():
    conn = store.connect(":memory:")
    w = get("wonhee.speech_subs")
    _fake_run(conn, w, {"clip_s": 40, "whisper_model": "tiny"}, {"vram_peak_bytes": 4e8, "runtime_s": 13})
    turbo = w.validate({"clip_s": 40, "whisper_model": "turbo"})
    assert predict.history_predictor(w, turbo, conn, time.time()).status == "insufficient_data"
