#!/bin/sh
# Capacity round (C8, 2026-10-19): push each kind of work until throughput stops rising and record which resource
# topped out first, then test packing tasks with different limiters together. Run after round2.sh on a quiet GPU.
# Then: `uv run python profiling/analysis/capacity.py`.
set -u
cd "$(dirname "$0")/../../.."
SEED=${SEED:-20261019}
X=profiling/sweeps/x
g() { echo "== gauge $*"; uv run gauge "$@" || echo "!! failed: gauge $*"; }

docker build -q -t gauge/torch:1 profiling/images/torch >/dev/null || exit 1
docker build -q -t gauge/wonhee-gpu:1 -f profiling/images/wonhee-gpu/Dockerfile . >/dev/null || exit 1
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$gpu" -eq 0 ] || echo "!! $gpu other GPU process(es) resident: runs record them, but the GPU is not quiet"

g sweep $X/warmup.yaml --order file                      # JIT cache populated for the new image
g sweep $X/c8_ladders.yaml --seed $SEED                  # in-process batch ladders (+ top-rung counter runs)
g corun --pairs $X/c8_copies.yaml --seed $SEED           # concurrent-copy ladders
g corun --pairs $X/c8_mix.yaml --seed $SEED              # packing tasks with different limiters
