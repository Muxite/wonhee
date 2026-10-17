"""Score frozen predictions against the measured summary. Censored runs are kept but flagged."""
from __future__ import annotations

from gauge.predict import FIELDS


def score(actual, estimate, upper, censored: bool) -> dict:
    point = estimate if estimate is not None else upper
    row = {"actual": actual, "estimate": estimate, "upper": upper, "censored": int(censored),
           "signed_error": None, "abs_error": None, "ratio": None,
           "underprediction": None, "excess_reservation": None, "covered": None}
    if actual is None:
        return row
    if point is not None:
        row["signed_error"] = actual - point
        row["abs_error"] = abs(actual - point)
        row["ratio"] = actual / point if point > 0 else None
    if upper is not None:
        under = max(actual - upper, 0.0)
        if censored:
            # The job died at `actual`, so its true need is >= actual. Only a miss is certain.
            row["underprediction"] = under if under > 0 else None
            row["covered"] = 0 if under > 0 else None
        else:
            row["underprediction"] = under
            row["excess_reservation"] = max(upper - actual, 0.0)
            row["covered"] = int(actual <= upper)
    return row


def evaluate_run(conn, run_id: int, summary: dict) -> None:
    preds = conn.execute("SELECT * FROM prediction WHERE run_id=?", (run_id,)).fetchall()
    for pr in preds:
        if pr["status"] != "supported":
            continue
        for target, (est_f, up_f) in FIELDS.items():
            est = pr[est_f] if est_f else None
            up = pr[up_f] if up_f else None
            if est is None and up is None:
                continue
            r = score(summary.get(target), est, up, summary.get("censored", False))
            conn.execute(
                "INSERT OR REPLACE INTO evaluation (prediction_id, run_id, target, actual, estimate, upper,"
                " signed_error, abs_error, ratio, underprediction, excess_reservation, covered, censored)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (pr["id"], run_id, target, r["actual"], r["estimate"], r["upper"], r["signed_error"],
                 r["abs_error"], r["ratio"], r["underprediction"], r["excess_reservation"], r["covered"],
                 r["censored"]))


AGG_SQL = """
SELECT r.family, r.workload, p.predictor, e.target,
       COUNT(*) AS n,
       SUM(e.censored) AS censored,
       AVG(CASE WHEN e.censored=0 THEN e.covered END) AS coverage,
       AVG(CASE WHEN e.censored=0 THEN e.abs_error / NULLIF(e.actual,0) END) AS mape,
       AVG(CASE WHEN e.censored=0 THEN e.ratio END) AS mean_ratio,
       AVG(CASE WHEN e.censored=0 THEN e.underprediction END) AS mean_under,
       AVG(CASE WHEN e.censored=0 THEN e.excess_reservation END) AS mean_excess
FROM evaluation e JOIN prediction p ON p.id=e.prediction_id JOIN run r ON r.id=e.run_id
WHERE r.corun_group IS NULL
GROUP BY r.family, r.workload, p.predictor, e.target ORDER BY r.workload, e.target, p.predictor
"""


def aggregate(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(AGG_SQL)]
