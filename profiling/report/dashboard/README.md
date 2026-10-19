# Findings dashboard

Published as the claude.ai artifact https://claude.ai/artifact/1fQCvCKLUTxUj7ad8KzA7n (private unless shared).

- `index.html` is the page (the artifact's `files/index.html`). Every number on it comes from a dataset.
- Datasets are the JSON tables written by `profiling/analysis/dashboard_export.py` to `data/gauge/dashboard/`, one file
  per dataset (findings, progress, audit, profiler, warmcold, inputs, interference, shapes, sharing, media,
  media_parallel, capacity, ladders, packing). They are computed from the gauge ledger with the same code as
  `round2.py`, `capacity.py` and `audit.py`.
- To refresh after new runs: re-run the export, upload each changed file as an asset of the artifact, then update
  that dataset's `source.url` and `updated`.
