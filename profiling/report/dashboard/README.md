# Findings dashboard

Published as the claude.ai artifact https://claude.ai/artifact/1fQCvCKLUTxUj7ad8KzA7n (private unless shared).

- `index.html` is the page (the artifact's `files/index.html`). Every number on it comes from a dataset.
- Datasets are the JSON tables written by `profiling/analysis/dashboard_export.py` to `data/gauge/dashboard/`, one file
  per dataset (findings, progress, audit, profiler, warmcold, inputs, interference, shapes, sharing, media,
  media_parallel, capacity, ladders, packing). They are computed from the gauge ledger with the same code as
  `round2.py`, `capacity.py` and `audit.py`.
- Raw-level tables from `profiling/analysis/dashboard_extra.py` (CSV): runs (every run), series (1 s time series of
  every run), marks (phase marks), kernel_mix, top_kernels, gpu_process (Nsight), coruns (every co-run member with
  overlap), batching (llama-batched-bench levels), starts (container starts), stress_series (C3), capacity_raw (C8).
  The page's Explorer part draws them; three tables are computed in the page (`dash.calc`): energy per unit by
  condition, kernel mix by workload, container start by variant.
- To refresh after new runs: re-run the export, upload each changed file as an asset of the artifact, then update
  that dataset's `source.url` and `updated`.
