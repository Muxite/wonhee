"""Replay today's predictors over recorded isolated runs, in time order, each seeing only runs that finished
before that run's prediction was committed. Scores predictor changes without re-running hardware.
Never writes to the ledger: frozen predictions stay what they were."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict

from gauge import predict
from gauge.evaluate import score
from gauge.workloads.base import get


def backtest(conn) -> list[dict]:
    runs = conn.execute(
        "SELECT r.id, r.workload, r.spec_json, r.prediction_committed, r.failure_class, s.json FROM run r"
        " JOIN summary s ON s.run_id=r.id WHERE r.status='done' AND r.corun_group IS NULL"
        " AND r.execution_mode='isolated' ORDER BY r.id").fetchall()
    acc = defaultdict(list)
    for r in runs:
        w = get(r["workload"])
        p = w.validate(json.loads(r["spec_json"])["params"])
        summary = json.loads(r["json"])
        for pred in predict.predict_all(w, p, conn, r["prediction_committed"]):
            if pred.status != "supported":
                continue
            for target, (ef, uf) in predict.FIELDS.items():
                est, up = getattr(pred, ef) if ef else None, getattr(pred, uf) if uf else None
                if est is None and up is None:
                    continue
                acc[(r["workload"], target, pred.predictor)].append(
                    score(summary.get(target), est, up, summary.get("censored", False)))
    out = []
    for (wl, target, pr), rows in sorted(acc.items()):
        ok = [x for x in rows if not x["censored"] and x["actual"]]
        cov = [x["covered"] for x in ok if x["covered"] is not None]
        ape = [x["abs_error"] / x["actual"] for x in ok if x["abs_error"] is not None]
        ratio = [x["ratio"] for x in ok if x["ratio"] is not None]
        out.append({"workload": wl, "target": target, "predictor": pr, "n": len(rows),
                    "censored": sum(x["censored"] for x in rows), "no_actual": sum(not x["actual"] for x in rows), "coverage": statistics.mean(cov) if cov else None,
                    "mape": statistics.mean(ape) if ape else None,
                    "median_ratio": statistics.median(ratio) if ratio else None})
    return out
