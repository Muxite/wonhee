#!/bin/sh
# C9 observer contrast, then every export: dashboard tables, observer analysis, the zip. Run after round2b.sh.
set -u
cd "$(dirname "$0")/../../.."
SEED=${SEED:-20261019}
echo "== gauge sweep profiling/sweeps/x/c9_observer.yaml"
uv run gauge sweep profiling/sweeps/x/c9_observer.yaml --seed $SEED || echo "!! failed: c9"
uv run gauge dataset
uv run python profiling/analysis/dashboard_export.py
uv run python profiling/analysis/dashboard_extra.py
uv run python profiling/analysis/export_zip.py
echo "round2c done"
