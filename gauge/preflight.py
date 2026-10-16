"""Advisory capacity check for a fresh isolated job. Not a reservation: anything can allocate after it."""
from __future__ import annotations

import time

from gauge import predict
from gauge.probe.gpu import Gpu
from gauge.spec import JobSpec
from gauge.workloads.base import get

VRAM_MARGIN = 256 * 2**20


def estimate(conn, spec: JobSpec):
    w = get(spec.workload)
    return predict.best(predict.predict_all(w, w.validate(spec.params), conn, time.time()))


def coexecution(conn, a: JobSpec, b: JobSpec) -> str:
    """measured_pair if these two configs were ever deliberately co-run; never inferred from fit alone."""
    wa, wb = get(a.workload), get(b.workload)
    ha, hb = wa.config_hash(wa.validate(a.params)), wb.config_hash(wb.validate(b.params))
    row = conn.execute(
        "SELECT 1 FROM run x JOIN run y ON x.corun_group = y.corun_group AND x.id != y.id"
        " WHERE x.config_hash=? AND y.config_hash=? AND x.status='done' AND y.status='done' LIMIT 1",
        (ha, hb)).fetchone()
    return "measured_pair" if row else "untested"


def preflight(conn, spec: JobSpec, alongside: JobSpec | None = None, gpu: Gpu | None = None) -> dict:
    gpu = gpu or Gpu()
    pred = estimate(conn, spec)
    snap_at = time.time()
    free = gpu.device().get("dev_mem_free")
    need = pred.vram_peak_upper_bytes
    if spec.execution_mode != "isolated":
        capacity, why = "unknown", "only fresh isolated jobs are checked"
    elif not get(spec.workload).gpu:
        capacity, why = "fits", "no GPU use declared"
    elif need is None or free is None:
        capacity, why = "unknown", "no VRAM bound" if need is None else "free VRAM unreadable"
    else:
        capacity = "fits" if need + VRAM_MARGIN <= free else "does_not_fit"
        why = f"need {need / 2**30:.2f} GiB + {VRAM_MARGIN / 2**30:.2f} margin vs free {free / 2**30:.2f} GiB"
    return {"capacity": capacity, "reason": why,
            "coexecution": coexecution(conn, spec, alongside) if alongside else "untested",
            "snapshot_at": snap_at, "predictor": pred.predictor, "evidence_run_ids": pred.evidence_run_ids,
            "vram_upper_bytes": need, "runtime_upper_s": pred.runtime_upper_s}
