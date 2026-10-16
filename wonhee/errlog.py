"""Append tracebacks to the one log file every error message points to."""
from __future__ import annotations

import traceback
from pathlib import Path


def log_path(home: Path | None) -> Path:
    return (home if home is not None else Path(".")) / "logs" / "wonhee.log"


def log_traceback(home: Path | None, label: str) -> Path:
    """Call from inside an except block; returns the log path."""
    log = log_path(home)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"--- {label}\n{traceback.format_exc()}\n")
    return log


def one_line(text: str) -> str:
    return " ".join(str(text).split())
