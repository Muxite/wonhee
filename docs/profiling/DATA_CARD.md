# gauge dataset: AI workloads on one GPU node, per process

Collected 2026-10-16 on a single host. Target use: understanding what agentic work nodes (small LLMs, speech, video, training) demand from GPU hardware, and what that implies for cloud packing and chip design. The data is descriptive; no predictor is included.

**Host:**
- RTX 5070 Ti (Blackwell GB203, 16 GB, driver 595.91, PCIe **Gen4 x16**: the board's limit, though the card supports Gen5)
- 16 CPU cores, 46 GiB RAM, kernel 6.8
- Docker 29.8 with cgroup v2

**Software:**
- llama.cpp `ghcr.io/ggml-org/llama.cpp:full-cuda` (digest per run in `runs.csv`)
- Ollama 0.35.0
- PyTorch 2.8 / CUDA 12.8
- Nsight Systems 2026.3.2

## Can data be isolated to one process?

Partly. Every column is one of three scopes:

| Scope | What | Isolated to the job? |
|---|---|---|
| `container` | cgroup v2 counters of the job's own container: CPU time, RAM peak (kernel high-water), IO, OOM events | **Yes, exact**, even when other jobs run |
| `process` | NVML VRAM per pid; Nsight CUDA trace of the job's own process tree: every kernel and copy, with start/end | **Yes, exact.** Under sharing, a kernel's duration also includes time the GPU spent on other contexts (time-slicing) |
| `device` | NVML utilization, power, energy, clocks, encoder/decoder; Nsight GPU counters at 10 kHz (SM active, tensor active, DRAM read/write bandwidth, PCIe) | **No.** Whole GPU. It equals the job only when the job ran alone. `runs.device_metrics_attributable` = 1 marks those runs |

**Measurement floor.** X0 profiles a container that only sleeps: **~27–28 W, 0% SM, ~1% graphics, 0.8–1.8% DRAM, ~1% PCIe TX**. Rerunning it after the display compositor and Sunshine streamer were stopped (X0b, runs 286–288, zero other GPU processes) gave the same floor. It comes from the act of measuring (Nsight's 10 kHz counter sampling plus an open CUDA context keep the GPU out of its low-power state), not from other tenants. The unobserved idle GPU draws 16.8 W in P8. Runs before 2026-10-16 ~18:20 had those two light tenants resident (229 MiB VRAM, no measurable SM load); every run records the other GPU processes present at start.

Profiler cost: Nsight adds ~4 s startup, ~6 CPU-s and ~0.5 GiB RAM inside the container. Use unprofiled runs for container CPU/RAM, and compare profiled runs only with profiled runs (the co-run analysis does this).

## Experiments, and what is happening in each

| ID | What runs | What is happening, in one line |
|---|---|---|
| X0 / X0b | `sleep` under the profiler, with / without display + Sunshine | The floor the measurement itself sees, and proof that the former tenants didn't raise it |
| X1 | Matmul per number format, gemv, copies, PCIe, launches | The ceilings: bf16 **102 TFLOPS**, fp8 **207**, fp32 35; DRAM ~**770–850 GB/s**; PCIe **28.8 GB/s** pinned; **6.8 µs** per kernel launch |
| X2 | llama.cpp prefill and decode as separate processes; qwen2.5 0.5B/3B/7B/14B (Q4_K_M), ×2 | **Prefill** = batched matmul on tensor cores (compute-bound). **Decode** = one matrix-vector per weight matrix per token, so every token rereads all weights (bandwidth-bound) |
| X2b | 7B decode with 99/20/12/4/0 of 28 layers on the GPU | Partial offload: the CPU and host memory become the bottleneck and the GPU sits idle |
| X3 | ResNet50 training (fp32, AMP), whisper (wonhee speech_subs), NVENC encode | Other task shapes: training is a mix of matmul and memory-bound norm/elementwise; whisper is CPU and startup bound; NVENC uses a separate engine |
| X4 | One process decoding 1/2/4/8 sequences (batched) **vs** 2/4 separate processes each decoding 1 | Consolidation: what sharing one GPU costs when work is batched vs time-sliced across processes |
| X5 | Every pair of {LLM decode, CNN training, NVENC encode, whisper}, ×2, with traced solos | Interference: who slows whom, measured per process |
| X6 | Ollama 7B, request 1 after page-cache eviction / with a warm page cache, request 2 resident | Serving cold-start ladder |
| (earlier) | Container start bench (1,760 starts), CUDA JIT cold/warm, sweeps | See `container_starts.csv`; earlier runs in `runs.csv` with `experiment = earlier` |

## Key measurements (medians of 2–3 runs; device scope from solo runs)

**LLM anatomy (X2):**

| qwen2.5 | Decode tok/s | Decode DRAM read | Decode tensor active | Implied weight bandwidth | Prefill tok/s | Prefill tensor active |
|---|---|---|---|---|---|---|
| 0.5B | 738 | 32% | 0% | 294 GB/s (launch/latency-bound) | 31,700 | 2% |
| 3B | 283 | 61% | 0% | 545 GB/s | 10,700 | 10% |
| 7B | 157 | 78% | 0% | 737 GB/s (~95% of measured copy ceiling) | 6,590 | 17.5% |
| 14B | 80 | 78% | 0% | 723 GB/s | 3,400 | 21.5% |

- **Decode kernel mix:** 77% `mul_mat_vec_q` (gemv), 10% `quantize_q8_1`, 5% attention, 4.5% RMSNorm. Median kernel 7 µs, ~190k launches/s. At these sizes the 6.8 µs launch cost is the same order as the work: small models are latency-bound.
- **Prefill kernel mix:** 72% `mul_mat_q` (batched matmul), 7.5% flash attention, 6% gated activation. Median kernel 37 µs.
- **Decode energy:** ≈ **0.15 J/token (0.5B), 0.8 (3B), 1.7 (7B), 3.6 (14B)** at 230–290 W. It scales with the weight bytes read per token. Prefill per-token energy is not reliable here (bursts shorter than NVML's ~1 s power averaging).
- **Partial offload (7B decode):** 28/28 layers on the GPU → 157 tok/s. 20 → 22 tok/s. 12 → 12.6. 4 → 8.8. 0 → 7.1. Moving ~30% of layers to the CPU loses ~86% of throughput, and GPU busy drops to 10%.

**Other tasks (X3):**
- **ResNet50 training:** matmul is only ~30% of kernel time; norm and elementwise ~46%. SM active 65–79%, tensor 12–15%. AMP cuts the median kernel from 77 to 17 µs.
- **Whisper turbo (speech_subs):** GPU busy 3% of the process's GPU window. The node is CPU and startup bound.
- **NVENC encode:** SM 0.2%, encoder engine ~7%. The compute cores are idle, so it is a good co-tenant.

**Consolidation (X4), qwen2.5 3B decode:**

| Setup | Total tok/s | Per stream |
|---|---|---|
| 1 process, 1 sequence | 274–284 | 274–284 |
| 1 process, 2 / 4 / 8 sequences batched | 406 / 614 / 1,427 | 203 / 154 / 178 |
| 2 separate processes | ~250 | ~125 |
| 4 separate processes | ~252 | ~63 |

Separate processes time-slice the GPU and get *less* total throughput than one process alone. Batching in one process gets 1.6× (2 streams) to 2.4× (4 streams) more. The jump at 8 sequences coincides with the batched-matmul kernel (`mul_mat_q`) appearing next to gemv in the trace.

**Pair interference (X5)**, as slowdown of each job vs its traced solo runs:

| Job \ alongside | LLM decode | CNN training | NVENC | Whisper |
|---|---|---|---|---|
| LLM decode (tok/s ratio) | 0.44 (2 copies) | **0.41–0.58** | 0.89 | 0.98 |
| CNN training (runtime) | 1.15× | | 1.06× | 1.00× |
| NVENC (runtime) | 1.08× | 1.08× | | 1.05× |
| Whisper (runtime) | 1.00–1.06× | 0.98× | 1.02× | |

Interference is asymmetric. Next to training, the latency-sensitive, bandwidth-bound LLM loses roughly half its throughput while training loses 15%. Jobs on different engines (NVENC) or mostly on the CPU (whisper) barely interact.

**Cold start (X6 + earlier):**
- Container start (`docker run`, no network): **~150 ms**. Warm `docker exec`: ~46 ms.
- Fresh Ollama container without a shared CUDA JIT cache: **+59 s** (64–68 s vs 5.3 s).
- 7B time to first token: **6.1–6.5 s** with the model on disk, **2.6–2.7 s** in page cache, **0.05 s** resident.

## Files (`data/dataset/`)

| File | Rows | What |
|---|---|---|
| `runs.csv` | one per job | experiment, workload, params, outcome, image digest, isolation flags, container and device summary |
| `workload_outcomes.csv` | one per reported result | llama-bench / batched-bench / Ollama timings, ceilings, training throughput |
| `gpu_process.csv` | one per profiled job | process-scope trace summary: kernel count, busy time, launches/s, gap stats, copy bandwidth |
| `gpu_kernel_categories.csv` | job × category | share of kernel time by kind of work (gemv, gemm, attention, norm, …) |
| `gpu_top_kernels.csv` | job × kernel | top 12 kernels by time, with count and median duration |
| `gpu_device_counters.csv` | one per metrics-profiled job | device counters averaged over the job's GPU window and the whole run |
| `kernel_timeline_10ms.csv.gz` | job × 10 ms bin | process-scope busy share per kernel category over time |
| `samples_100ms.csv.gz` | job × 100 ms | container counters + NVML (VRAM per pid, util, power, clocks, encoder/decoder) |
| `corun_interference.csv` | one per co-run job | slowdown, kernel stretch and tokens/s vs identical solo runs |
| `container_starts.csv` | one per container start | start-overhead phases for 12 launch variants |

Raw Nsight traces (`.nsys-rep` + SQLite export, 6.2 GB) are in `data/nsys/run<id>.*`, for anything not summarized here.

## Known limits
- **One machine, one GPU model.** The ratios (bandwidth-bound decode, interference asymmetry, batching vs time-slicing) are expected to generalize. Absolute numbers are this card's.
- **2–3 repeats per condition:** enough to see the effect, not to give tight intervals. Repeat spread is visible in the per-run rows.
- **Ollama traces are partial.** It kills its model runner, which loses buffered trace records, so the per-process anatomy uses llama.cpp directly (the same engine).
- **`kernel_stretch` under sharing** mixes real contention with time-slice preemption, which CUPTI does not separate.
- **Device counters during co-runs** belong to both jobs together.
- **Media inputs:** wonhee clips come from a third-party reel and are not included; only numbers derived from them are.
