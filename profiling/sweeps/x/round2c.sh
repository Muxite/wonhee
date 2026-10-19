#!/bin/sh
# C8 packing supplement (NVENC groups), the C9 observer contrast, then every export: dashboard tables, observer analysis, the zip. Run after round2b.sh.
set -u
cd "$(dirname "$0")/../../.."
SEED=${SEED:-20261019}
echo "== gauge corun --pairs profiling/sweeps/x/c8_mix_nvenc.yaml"
uv run gauge corun --pairs profiling/sweeps/x/c8_mix_nvenc.yaml --seed $SEED || echo "!! failed: c8 mix nvenc"
echo "== gauge sweep profiling/sweeps/x/c9_observer.yaml"
uv run gauge sweep profiling/sweeps/x/c9_observer.yaml --seed $SEED || echo "!! failed: c9"
uv run gauge dataset
uv run python profiling/analysis/dashboard_export.py
uv run python profiling/analysis/dashboard_extra.py
uv run python profiling/analysis/export_zip.py
echo "round2c done"
