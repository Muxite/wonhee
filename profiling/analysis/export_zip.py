"""Bundle the profiling results into one zip for analysis elsewhere: the dashboard tables, the raw `gauge dataset`
export, the audit and analysis outputs, the pre-registered experiment files, and a README describing each file.
Nsight traces (GBs) and third-party media are left out.

    uv run python profiling/analysis/export_zip.py [--out data/gauge/gpu-profiling-<date>.zip]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gauge import paths  # noqa: E402

DASH = paths.DATA / "dashboard"
RAW = paths.DATA / "dataset"

ABOUT = {
    # dashboard tables (profiling/analysis/dashboard_export.py, dashboard_extra.py)
    "findings.json": "Headline figures, each computed from the other tables.",
    "progress.json": "Planned vs finished runs per pre-registered experiment.",
    "audit.json": "The earlier 317 runs checked against what the evidence needs (AUDIT.md).",
    "profiler.json": "C1: steady-state rate and wall time with Nsight off, tracing, and on GPU metrics.",
    "warmcold.json": "C7: phases before the first result with page cache and CUDA JIT cache controlled.",
    "inputs.json": "C5: load, first and later call time and VRAM per GPU node at small and large inputs.",
    "interference.json": "C3: LLM decode beside one-resource stressors and training, both sides' share of solo speed.",
    "shapes.json": "C3 solos: each kind of work's share of the power, memory-controller, SM and clock ceilings.",
    "sharing.json": "C2: 4 requests batched vs 4 time-sliced processes vs 4 MPS clients.",
    "media.json": "C6: decode / encode / transcode on CPU or NVDEC/NVENC, with engine use as backend check.",
    "media_parallel.json": "C6: 4 parallel transcodes, CPU decode vs NVDEC.",
    "resident.json": "C0: steady VRAM of each wonhee GPU model held resident (one at a time).",
    "capacity.json": "C8: every ladder step, median over repeats: throughput and each resource's share of its ceiling.",
    "ladders.json": "C8: per ladder, peak, plateau and the resource that topped out first (pre-registered rules).",
    "packing.json": "C8: tasks with different limiters together; each one's share of its solo speed, summed.",
    "runs.csv": "Every finished run: experiment, workload, params, outcome, CPU/RAM/VRAM, power, energy, phases, work, J per unit.",
    "series.csv": "1 s time series of every run (device-scope GPU readings, container CPU and RAM).",
    "marks.csv": "Phase marks of every run, seconds from container start.",
    "kernel_mix.csv": "Nsight runs: share of kernel time by kernel category.",
    "top_kernels.csv": "Nsight runs: top 12 kernels with count and median duration.",
    "gpu_process.csv": "Nsight runs: kernel count, busy time, launches/s, kernel and gap lengths.",
    "coruns.csv": "Every co-run member vs its matched solos: slowdown, compute slowdown, throughput ratio, overlap.",
    "batching.csv": "Every llama-batched-bench level: sequences, total and per-sequence decode speed.",
    "starts.csv": "1,760 container starts: time to created, started, first output.",
    "stress_series.csv": "C3 stressors' work rate every half second, solo and beside LLM decode.",
    "capacity_raw.csv": "C8: each repeat of each ladder step.",
}


def tables_md() -> str:
    """Column list per CSV/JSON table, read from the files themselves."""
    out = []
    for f in sorted(DASH.glob("*")):
        if f.suffix == ".csv":
            with f.open() as fh:
                cols = next(csv.reader(fh), [])
            n = sum(1 for _ in f.open()) - 1
        else:
            rows = json.loads(f.read_text())
            cols = list(dict.fromkeys(k for r in rows for k in r)) if rows else []
            n = len(rows)
        out.append(f"### tables/{f.name} ({n} rows)\n\n{ABOUT.get(f.name, '')}\n\nColumns: {', '.join(f'`{c}`' for c in cols) or '(empty)'}\n")
    return "\n".join(out)


def readme(commit: str, runs: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# GPU workload profiling: data export

Exported {now} from the wonhee repo at commit {commit}. {runs} runs in the gauge ledger.
Host: RTX 5070 Ti (16 GiB, 300 W limit, PCIe Gen4 x16), Ryzen 7 3700X (16 threads), driver 595.91.07.

## Layout

| Path | What |
|---|---|
| `tables/` | The findings dashboard's tables: per-contrast summaries (JSON) and run-level data (CSV) |
| `raw/` | `gauge dataset` export: runs, outcomes, 100 ms samples, Nsight per-process and per-kernel summaries, 10 ms kernel timeline, co-run interference, container starts |
| `analysis/` | `round2.md` / `round2.json` (pre-registered rules applied to C0-C7), `capacity.md` / `capacity.json` (C8), `observer.md` / `observer.json` (how much measuring changed the results: Nsight, the host sampler, the idle floor, container launch, co-run attribution; C9), `audit.json` |
| `experiments/` | The experiment files as run: each states its question, measure, analysis rule, tolerance and stopping rule |
| `docs/` | AUDIT.md (evidence audit), DATA_CARD.md, GUIDE.md (how gauge measures) |

## Reading the numbers

- Observer effect (analysis/observer.md): Nsight adds seconds of fixed cost per run and slows steady-state work by
  0-20%, more for launch-heavy jobs; the host sampler costs ~1% of the job's CPU and is never charged to it;
  an observed idle GPU draws more than an unobserved one. Rates on the dashboard are from profiler-off runs.

- Rates come from profiler-off runs (C1: Nsight slows steady-state work by 4.5-8%).
- Device readings (power, busy, clocks, engines, PCIe) are whole-GPU; in co-runs they belong to all jobs together.
- `[min-max]` in the summaries is over repeats (2-5 per cell). Experiments ran interleaved, each round shuffled
  with seed 20261019.
- Page cache (`cache`) and CUDA JIT cache (`jit_cache`) are explicit conditions from 2026-10-19 on; earlier runs
  did not control them (AUDIT.md).
- NVML "busy" is the share of time any kernel ran, not how much of the chip it used.

## Tables

{tables_md()}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.DATA / f"gpu-profiling-{datetime.now():%Y-%m-%d}.zip"))
    a = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    runs = sqlite3.connect(paths.DATA / "gauge.sqlite").execute("SELECT COUNT(*) FROM run WHERE status='done'").fetchone()[0]
    py = [sys.executable]
    an = {}
    for name, cmd in (("round2", ["profiling/analysis/round2.py"]), ("capacity", ["profiling/analysis/capacity.py"]),
                      ("observer", ["profiling/analysis/observer.py"])):
        js = paths.DATA / f"{name}.json"
        md = subprocess.run(py + cmd + ["--json", str(js)], cwd=ROOT, capture_output=True, text=True).stdout
        an[f"analysis/{name}.md"], an[f"analysis/{name}.json"] = md, js.read_text() if js.exists() else "{}"
    audit = subprocess.run(py + ["profiling/audit/audit.py"], cwd=ROOT, capture_output=True, text=True).stdout
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", readme(commit, runs))
        for f in sorted(DASH.glob("*")):
            z.write(f, f"tables/{f.name}")
        for f in sorted(RAW.glob("*")):
            z.write(f, f"raw/{f.name}")
        for k, v in an.items():
            z.writestr(k, v)
        z.writestr("analysis/audit.json", audit)
        for f in sorted((ROOT / "profiling" / "sweeps" / "x").glob("*")):
            if f.suffix in (".yaml", ".sh"):
                z.write(f, f"experiments/{f.name}")
        for f in ("docs/profiling/AUDIT.md", "docs/profiling/DATA_CARD.md", "docs/profiling/GUIDE.md"):
            z.write(ROOT / f, f"docs/{Path(f).name}")
    size = Path(a.out).stat().st_size
    with zipfile.ZipFile(a.out) as z:
        print(f"{a.out}: {len(z.namelist())} files, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
