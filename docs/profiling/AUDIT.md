# P0 evidence audit (2026-10-19)

**Audited:** the gauge ledger `data/gauge/gauge.sqlite`, which holds 317 runs from 2026-10-16 and 2026-10-17. They were checked against the experiment plan's list of what the evidence needs before anything new is run. The facts come from `uv run python profiling/audit/audit.py`, plus the queries quoted here. This page interprets them. The audit reanalyses existing records only; it involves no new runs.

## What the records can and cannot support

| Need | Status | Evidence |
|---|---|---|
| Immutable run manifest | **Partly** | Every run has its validated params, a prediction frozen before launch, and its initial state (free VRAM, GPU processes, temperature). 289 of 317 runs have a host fingerprint (driver 595.91.07 throughout). **Missing:** the exact argv (it is rebuilt from today's code), input hashes (media files and GGUF blobs are named, not hashed), and cache or JIT state as a field. **The "image digest" is the local image ID**, which cannot be pulled again. Every gauge image had been pruned by 2026-10-19, and `llama.cpp:full-cuda` is a floating tag, so the exact software of the old runs cannot be restored |
| Raw exports | **Yes** | All 130 profiled runs keep their `.nsys-rep` file and a SQLite export. 313 runs have 100 ms samples |
| Failed / OOM runs kept | **Yes** | 4 `cuda_oom` and 5 abandoned runs stay in the ledger. Censored runs are never used as "fits" labels |
| Phase-level timing | **Partly** | 189 runs have phase marks, but they come from LLM and training workloads only. The wonhee nodes (whisper, subtitle burn) report one end-to-end time |
| Matching baselines | **Partly** | Of 79 co-run members, 76 have solo references with the same params and profile mode. Only 58 had them when the analysis accepted traced references alone; profiler-off runs now count too, still never mixed across modes. 26 of 36 groups are referenced on both sides |
| Alternating or randomized order | **No** | 81 of 92 repeated conditions ran back to back |
| Repeats | **Thin** | Of 109 solo conditions, 17 ran once, 76 twice and 16 three or more times. The median repeat spread is 3.3% of runtime; the 90th percentile is 22% |
| Uncertainty shown | **No** | The report shows medians, with spread only in the per-run tables |
| Experiment label per run | **No** | It is inferred afterwards from params (`gauge/dataset.py`). Runs from 2026-10-19 on record it explicitly |

## Findings that change how existing numbers may be used

1. **The profiler inflates wall time, not training throughput.**
   - The same solo condition under Nsight takes 1.2–2.5× the wall time of a profiler-off run: speech_subs ×1.24 (trace), NVENC ×1.58 (trace) and ×2.46 (metrics), ResNet-50 ×1.90 (metrics).
   - That run's own reported training throughput did not change (0.995×, n = 2), so the extra time is Nsight's fixed startup and export cost, not slower steady-state work.
   - **Rule:** compare runtimes only within one profile mode (the report already does this), and quote steady-state rates from the workload's own counters.
   - **Not yet shown:** that LLM decode is equally unaffected. No profiler-off llama.cpp run exists.
2. **Some pair slowdowns understate contention.**
   - 12 of 58 referenced co-run members overlapped their partner for less than 80% of their own run (`interference.overlap_frac`).
   - Worst: **whisper next to NVENC (0.36–0.38), training (0.66) and LLM decode (0.56–0.60)**. The partner was gone for much of whisper's work.
   - "Whisper slows ×1.0 next to anything" is therefore **weakly supported**, and `docs/production/policies.md` quotes it.
   - The NVENC pairs (0.53–0.79) are also short. LLM with training (0.80–0.84) and LLM with LLM (≥0.97) hold.
3. **Cache state is not part of the condition.**
   - Run 12 (fresh CUDA JIT, 66 s) shares a condition with four warm runs of 5.3 s.
   - 17 of 92 repeated conditions ran more than 10% slower on their first repeat (up to about 2×), all in model-loading workloads. Disk reads confirm it: `llm.generate` read 2.5 GB on its first repeat and 0.03 GB after.
   - Back-to-back repeats put that cold run first every time.
4. **"Throttled" means the power limit.** All 874 throttled samples show reason 0x4, the software power cap. None is thermal. Start temperatures ranged 31–51 °C.
5. **One light GPU tenant was resident in 280 of 317 runs.** It is the same 234 MiB process each time (display compositor and streamer). X0b found the same measurement floor without it, so it is recorded but not a confound for these workloads.

6. **Found while piloting round 2 (runs 318–332): the "cold whisper" cost is mostly CUDA JIT, not model load.**
   - In a fresh container, whisper's (ctranslate2) first call took **10.5 s** against **0.4 s** for the second. The model load took 3 s.
   - With a populated, persistent JIT cache, the first call took **1.0 s**, and a 10 s clip went from about 17 s to about 7 s end to end.
   - The PyTorch masks moved by only 0.3–0.8 s; their cold cost is framework import (about 4 s) plus weights (about 8 s).
   - So the old "whisper ≈ 15 s flat" (X3) mostly measured JIT compilation in fresh containers. JIT cache state is now a recorded condition (`jit_cache`).

## The plan's questions against existing records

| # | Question | Status | Existing evidence | Smallest contrast still needed |
|---|---|---|---|---|
| 1 | Does the profiler change what we measure? | Partly | Wall time +24–146%, training throughput unchanged (one condition) | **C1**: LLM decode and a short GPU node with the profiler off, on trace and on metrics, alternated, n = 5. Tolerance set in advance: steady-state rate within ±3% |
| 2 | Copies vs batching vs MPS | Partly | X4: batching 1/2/4/8 vs 2/4 separate processes. The work is not matched (llama-bench vs batched-bench, different token counts). No tail latency, no MPS | **C2**: 4 requests of 256 tokens, fixed. One batched process vs 4 time-sliced processes vs 4 MPS clients. Aggregate tok/s, per-request completion, VRAM |
| 3 | Does solo demand predict sensitivity? | Partly | X5 both-sided slowdowns: LLM decode loses 0.35–0.58× next to training, training loses 1.15×. Mechanism untested | **C3**: LLM decode next to a pure-bandwidth stressor and next to a pure-compute stressor. Supports a mechanism hypothesis; does not prove one |
| 4 | Does low DRAM bandwidth rule out a memory bottleneck? | Missing | Decode DRAM read share 32–78%. No working-set or locality contrast | Deferred (P2). Coverage-map gap |
| 5 | Is task identity enough to reuse a profile? | Answered for the LLM, missing for wonhee nodes | X2 splits prefill and decode across 4 model sizes. The wonhee nodes have no phase or input contrast | **C5**: masks, demucs and SigLIP2 at small and large inputs, with phase marks |
| 6 | Is a long speech/video run evidence of GPU demand? | Partly | Whisper GPU busy 3% (trace), NVENC SM 0.2%. Decode always on the CPU. No encode-only or decode-only runs | **C6**: decode-only (CPU vs NVDEC), encode-only, transcode, each with backend checks (engine utilisation, log) |
| 7 | Warm/cold vs steady state | Answered for LLM serving, partly for nodes | X6 ladder: disk 6.1 s, page cache 2.6 s, resident 0.05 s. JIT +59 s. Container start 0.15 s. Nodes: whisper ~15 s flat | **C7**: same whisper work as a cold process vs a warm worker, phases split (launch, load, first result, later results), page cache evicted vs warm |
| 8 | Can one device predict others? | Out of scope | One GPU model | None until a second device exists for a named hypothesis |

## Process changes before any new run

- **Interleave conditions.** `gauge sweep --order interleave --seed N` runs repeat 1 of every condition, then repeat 2, and so on. With `--seed`, each round is shuffled.
- **Pre-register every contrast.** Each sweep file states its question, contrast, analysis rule, tolerance and stopping condition, and the decision it changes, before it runs.
- **Treat state as part of the condition.** Page cache (`cache: cold | warm`) and the CUDA JIT cache (`jit_cache: none | shared`) are set explicitly per run, never left to run order.
- **Record what can be restored.** Each run now also records the image's registry digest, its exact argv, and sha256 of its media inputs.
- **Show the spread.** Results report n, median and min–max (or a bootstrap interval) next to every ratio.
