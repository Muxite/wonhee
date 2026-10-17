# gauge (workload profiling inside wonhee)

A workload profiler with a prediction ledger. For each job, gauge freezes a resource prediction, runs the job in its own Docker container, measures it from the host, and scores the prediction.
It is **not a scheduler**. It produces data, plus two advisory calls (`estimate`, `preflight`) for wonhee and the moeka/agentic-node side.

```
JobSpec → predict (prior, history, calibrated) → immutable prediction rows
        → docker run (sh wrapper holds the cgroup open) + host sampler (cgroup v2 + NVML, 100 ms)
        → final cgroup read → release → classify outcome → summary → evaluation
```

Later additions (Nsight traces via `profile: trace|metrics`, `startbench`, `sequence`, `trace-export`, `dataset`) are described step by step in the appendix of `profiling/report/shared-gpu-node.html`. That appendix is the full how-it-works reference, including setup on another machine.

## Where things live

| Path | What |
|---|---|
| `gauge/` | the profiler package |
| `profiling/images/` | workload Dockerfiles (`gauge build`); `wonhee/` builds from this repo |
| `profiling/sweeps/` | job grids; `x/` holds the experiment files behind the report |
| `profiling/report/` | the report page, its template and `build_report.py` |
| `data/gauge/` | ledger `gauge.sqlite`, `nsys/`, `traces/`, `dataset/`, `media/` (gitignored; override with `GAUGE_HOME`) |

## Use

```bash
uv sync --extra dev
# `uv run wonhee profile <cmd>` is the same as `uv run gauge <cmd>`
uv run gauge doctor                       # cgroup v2, docker, nvidia runtime, NVML capabilities, gpu-lock
uv run gauge build && uv run gauge prepare  # workload images (profiling/images), ffmpeg source clip
uv run gauge run train.small batch=64     # one job
uv run gauge sweep profiling/sweeps/train.yaml   # a grid, in file order (predictions use only earlier runs)
uv run gauge corun --pairs profiling/sweeps/corun.yaml --repeats 3
uv run gauge preflight llm.generate model=qwen2.5:14b num_ctx=16384
uv run gauge report && uv run gauge export
```

Library: `from gauge import estimate, preflight`. `preflight` returns
`{capacity: fits|does_not_fit|unknown, coexecution: measured_pair|untested, ...}`. It is advisory only. Another process can allocate VRAM right after the check.

## What is measured, and at what scope

| Metric | Source | Scope |
|---|---|---|
| CPU time, user/system, throttled | `cpu.stat` counters at the end of the run | container |
| RAM peak | `memory.peak` (the kernel's high-water mark) | container. **Includes page cache charged to the cgroup** |
| Anon / file split | `memory.stat`, sampled | container (observed peaks) |
| OOM, pressure stall time, IO bytes/ops | `memory.events`, `*.pressure`, `io.stat` | container |
| VRAM | NVML per-pid memory, pid → `/proc/<pid>/cgroup`, descendants included | process_sum (observed peak) |
| GPU util, power, clocks, throttle, energy | NVML device | **device**: attributable only if nothing else overlapped |

Unsupported NVML metrics are recorded as `null` with a reason in `metric_capabilities`, never as `0`.
Each run also snapshots its initial state (free VRAM, other GPU pids, swap, temperature, overlapping runs) and holds the machine's `gpu-lock`.

## Outcomes and censoring

`failure_class` is one of `success | host_oom | cuda_oom | timeout | application_error | unknown`.
- `host_oom` comes from cgroup `memory.events` or Docker's OOMKilled flag.
- `cuda_oom` comes only from the workload's own `GAUGE_MARK cuda_oom`, or from a log match labelled `log-derived`.

OOM and timeout runs are **censored**: they are kept, but never used as a "fits at X" label. In evaluation, a censored run can only prove that a bound was too low.

## Predictors

- **prior**: an analytic formula per workload. For an LLM: GGUF bytes + KV cache (2·layers·kv_heads·head_dim·2·ctx) + overhead. For ffmpeg: pixels / throughput. Its upper bound is the estimate + 25%.
- **history**: same workload and config, same log2 size bucket. Estimate = median, upper = max + 10%.
- **calibrated**: per target, fits `actual = a + b·prior` on earlier runs. `a` absorbs fixed costs the formula misses, and `b` rescales its rate constants. Because the prior already encodes size, one config can pool across sizes and models, so it can be scored on sizes it has never seen.

Upper bounds are heuristics. Don't read them as calibrated p90/p95.

## Workload protocol (optional)

Stdout lines are timestamped by the host:
- `GAUGE_MARK <name>` marks a phase (`model_loaded`, `compute_started`, `cuda_oom`, ...).
- `GAUGE_OUTCOME {json}` reports facts such as token counts. These are stored as outcomes, never as prediction features.

## Known measurement facts on this box (2026-10-16)

- **CUDA JIT**: Ollama 0.35 in a fresh container spends about 60 s JIT-compiling for the RTX 5070 Ti. A 0.5B request took 66.6 s cold vs 5.3 s with a persistent compute cache. `llm.generate` defaults to `jit_cache=shared` (volume `gauge-cuda-cache`).
- **Page cache**: file pages are charged to the cgroup that first reads them. A model that another container already read costs this job no `file` memory, so RAM peaks depend on host history. Use the anon/file split.
- **Energy**: for short runs the NVML energy counter differs a lot from integrated sampled power. The counter is primary.
