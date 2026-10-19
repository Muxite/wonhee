#!/bin/sh
# Profiling round 2 (2026-10-19), in run order. Every step waits for the machine's gpu-lock, so this can run
# unattended next to other sessions. Each step is independent: rerun one by copying its line.
# Results land in data/gauge/gauge.sqlite; then `gauge dataset` and `python profiling/report/build_report.py`.
set -u
cd "$(dirname "$0")/../../.."
g() { echo "== gauge $*"; uv run gauge "$@" || echo "!! failed: gauge $*"; }

g build                                                    # images: wonhee-gpu (torch), torch, wonhee
g sweep profiling/sweeps/x/x8_new_nodes.yaml               # X8  load vs compute per GPU node; warm worker vs cold
g sweep profiling/sweeps/x/x9_resident.yaml                # X9a each model held: steady VRAM
g corun --pairs profiling/sweeps/x/x9_coresident.yaml      # X9b sets of models held together: what fits in 16 GB
g sweep profiling/sweeps/x/x9_solos.yaml                   # X9c traced solo references
g corun --pairs profiling/sweeps/x/x9_pairs.yaml           # X9d new nodes next to LLM decode / NVENC
g sweep profiling/sweeps/x/x10_cuvid.yaml                  # X10a NVDEC + scale_cuda vs CPU decode
g corun --pairs profiling/sweeps/x/x10_parallel.yaml       # X10b 2/4/8 parallel encodes
g sweep profiling/sweeps/x/x12_batch_limits.yaml           # X12 batching to 64 sequences, 2k prompts, 14B
if uv run python -c "from gauge import mps; mps.start(); mps.stop()"; then  # X11 only if MPS starts here
  g sweep profiling/sweeps/x/x11_mps_solos.yaml
  g corun --pairs profiling/sweeps/x/x11_mps.yaml
else echo "!! MPS daemon did not start: X11 skipped (record that as the result)"; fi
g dataset
