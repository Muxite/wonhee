"""Where gauge keeps things inside the wonhee repo. GAUGE_HOME moves the data (ledger, traces, media) elsewhere."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("GAUGE_HOME") or REPO / "data" / "gauge")
PROFILING = REPO / "profiling"
IMAGES = PROFILING / "images"
DOCS = REPO / "docs" / "profiling"
