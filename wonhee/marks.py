"""gauge's stdout protocol, spoken only when gauge runs us (GAUGE=1): `GAUGE_MARK <name>` timestamps a phase,
`GAUGE_OUTCOME {json}` reports facts. gauge turns model_loaded / compute_started / compute_finished into load and
compute durations. Outside gauge these are no-ops, so they are safe in normal runs."""
from __future__ import annotations

import json
import os
import sys


def on() -> bool:
    return os.environ.get("GAUGE") == "1"


def mark(name: str) -> None:
    if on():
        print(f"GAUGE_MARK {name}", file=sys.stdout, flush=True)


def outcome(**kw) -> None:
    if on():
        print("GAUGE_OUTCOME " + json.dumps(kw, default=str), file=sys.stdout, flush=True)
