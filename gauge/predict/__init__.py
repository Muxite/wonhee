"""Predictors. Each returns a Prediction from facts known before launch plus finished history.

- prior:      the workload's analytic formula (weights + KV cache, frames*pixels/throughput, load + RTF*audio...)
- history:    same workload+config, same log2 size bucket: median estimate, max*margin upper bound
- calibrated: per target, actual = a + b * prior_estimate fitted on history. `a` absorbs fixed costs the
              formula misses (CUDA context, server start), `b` rescales its rate constants. Because the
              prior already encodes size, one config can pool across sizes and extrapolate to unseen ones.
"""
from __future__ import annotations

import math
import statistics

from gauge import store
from gauge.spec import Prediction
from gauge.workloads.base import Workload

MARGIN = 0.10
# summary key -> (Prediction estimate field, Prediction upper field)
FIELDS = {
    "runtime_s": ("runtime_estimate_s", "runtime_upper_s"),
    "cpu_time_s": ("cpu_time_estimate_s", None),
    "ram_peak_bytes": (None, "ram_peak_upper_bytes"),
    "vram_peak_bytes": (None, "vram_peak_upper_bytes"),
}


def _usable(runs: list[dict]) -> list[dict]:
    return [r for r in runs if r["failure_class"] == "success" and r["size"] is not None]


def history_predictor(w: Workload, p, conn, now: float) -> Prediction:
    size = w.size(p)
    runs = _usable(store.history(conn, w.family, w.config_hash(p), now, w.name))
    bucket = lambda s: round(math.log2(s)) if s and s > 0 else None  # noqa: E731
    same = [r for r in runs if bucket(r["size"]) == bucket(size)]
    if not same:
        return Prediction(predictor="history", predictor_version=1, status="insufficient_data",
                          history_cutoff=now, note=f"{len(runs)} runs of this config, none in size bucket")
    out: dict = {}
    for key, (est, up) in FIELDS.items():
        vals = [r["summary"].get(key) for r in same if r["summary"].get(key) is not None]
        if not vals:
            continue
        if est:
            out[est] = statistics.median(vals)
        if up:
            out[up] = max(vals) * (1 + MARGIN)
    return Prediction(predictor="history", predictor_version=1, status="supported", history_cutoff=now,
                      evidence_run_ids=[r["id"] for r in same], **out)


def fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b*x. Returns (a, b, max positive residual)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    worst = max(0.0, *(y - (a + b * x) for x, y in zip(xs, ys)))
    return a, b, worst


def calibrated_predictor(w: Workload, p, conn, now: float) -> Prediction:
    runs = _usable(store.history(conn, w.family, w.config_hash(p), now, w.name))
    here = w.prior(p)
    out, notes, used = {}, [], set()
    for key, (est, up) in FIELDS.items():
        if here.get(key) is None:
            continue
        pts = []
        for r in runs:
            y = r["summary"].get(key)
            x = w.prior(w.validate(r["params"])).get(key)
            if x is not None and y is not None:
                pts.append((x, y, r["id"]))
        if len({round(x, 6) for x, _, _ in pts}) < 2:
            continue
        a, b, worst = fit_affine([x for x, _, _ in pts], [y for _, y, _ in pts])
        y = max(a + b * here[key], 0.0)
        if est:
            out[est] = y
        if up:
            out[up] = (y + worst) * (1 + MARGIN)
        used.update(i for _, _, i in pts)
        notes.append(f"{key}: a={a:.4g} b={b:.4g} n={len(pts)}")
    if not out:
        return Prediction(predictor="calibrated", predictor_version=2, status="insufficient_data",
                          history_cutoff=now, note=f"need runs at >=2 distinct prior values; have {len(runs)} runs")
    return Prediction(predictor="calibrated", predictor_version=2, status="supported", history_cutoff=now,
                      evidence_run_ids=sorted(used), note="; ".join(notes), **out)


def predict_all(w: Workload, p, conn, now: float) -> list[Prediction]:
    return [w.prior_prediction(p), history_predictor(w, p, conn, now), calibrated_predictor(w, p, conn, now)]


def best(preds: list[Prediction]) -> Prediction:
    """Most data-backed supported prediction: calibrated > history > prior."""
    order = {"calibrated": 0, "history": 1, "prior": 2}
    ok = [p for p in preds if p.status == "supported"]
    return sorted(ok, key=lambda p: order.get(p.predictor, 9))[0]
