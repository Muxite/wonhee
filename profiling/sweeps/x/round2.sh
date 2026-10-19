#!/bin/sh
# Profiling round 2 (2026-10-19): the pre-registered contrasts C0-C7 from docs/profiling/AUDIT.md, in run order.
# Each file states its question, measure, analysis rule, tolerance and stopping rule before it runs.
# Conditions are interleaved and each round shuffled with a fixed seed (reproducible order); every run waits for
# the machine's gpu-lock, so this can run unattended next to other sessions.
# Then: `uv run gauge dataset` and `uv run python profiling/analysis/round2.py`.
set -u
cd "$(dirname "$0")/../../.."
SEED=${SEED:-20261019}
X=profiling/sweeps/x
g() { echo "== gauge $*"; uv run gauge "$@" || echo "!! failed: gauge $*"; }

g build                                                  # images: wonhee-gpu, torch (stressor), wonhee
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$gpu" -eq 0 ] || echo "!! $gpu other GPU process(es) resident: runs record them, but the GPU is not quiet"

g sweep $X/c1_profiler.yaml --seed $SEED                 # C1 profiler on/off: decides how every later run is read
g sweep $X/c7_warm_cold.yaml --seed $SEED                # C7 cold vs warm, cache controlled
g sweep $X/c5_inputs.yaml --seed $SEED                   # C5 input size per GPU node
g corun --pairs $X/c3_stressors.yaml --seed $SEED        # C3 targeted co-runners
g sweep $X/c6_media.yaml --seed $SEED                    # C6 media stages and engines
g corun --pairs $X/c6_parallel.yaml --seed $SEED         # C6 4 parallel transcodes
if uv run python -c "from gauge import mps; mps.start(); mps.stop()"; then
  g corun --pairs $X/c2_sharing.yaml --seed $SEED        # C2 copies vs batching vs MPS at fixed work
else
  echo "!! MPS daemon did not start: running C2 without the MPS arm is not meaningful; record MPS as unsupported"
fi
g sweep $X/c0_resident.yaml --seed $SEED                 # C0 operational: resident VRAM for the worker
g dataset
