"""Saved filters: the pack that ships with wonhee plus the ones you make (`wonhee filter make`), by name.

A saved filter is one JSON file in <home>/filters/<name>.json holding the numbers, what it was asked to be, and
how it got there (each round's filter, score and the reviewer's notes), so it can be re-applied, tweaked, or
exported as a .cube LUT later. Pack names are reserved so `film-warm` always means the same thing.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from wonhee import fsutil
from wonhee.edit.filters import PACK, Filter


def slug(name: str) -> str:
    raw = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()).strip("-")
    if not s:
        if not raw:
            raise ValueError("a filter needs a name")
        return "f-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]  # e.g. a Korean name
    return s[:48].strip("-")


def folder(home: Path) -> Path:
    return home / "filters"


def path_for(home: Path, name: str) -> Path:
    return folder(home) / f"{slug(name)}.json"


def load(home: Path, name: str) -> tuple[Filter, dict]:
    """(filter, record) for a saved filter or a pack preset; saved files cannot shadow the pack. Unknown keys in an
    old or hand-edited file are dropped (listed in rec["dropped"]) instead of failing the load."""
    key = slug(name)
    if key in PACK:
        desc, f = PACK[key]
        return f, {"name": key, "source": "pack", "description": desc}
    p = path_for(home, key)
    if not p.exists():
        known = ", ".join([*PACK, *(x["name"] for x in saved(home))])
        raise ValueError(f"no filter called '{name}'. Known: {known}")
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        raw = rec["filter"]
        keep = {k: v for k, v in raw.items() if k in Filter.model_fields}
        spec = Filter(**keep)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:  # JSONDecodeError and pydantic are ValueErrors
        raise ValueError(f"saved filter {name} is unreadable: {e}") from e
    dropped = sorted(set(raw) - set(keep))
    return spec, {**rec, "source": "saved", **({"dropped": dropped} if dropped else {})}


def save(home: Path, name: str, record: dict) -> Path:
    key = slug(name)
    if key in PACK:
        raise ValueError(f"'{key}' is a pack filter; pick another name (e.g. my-{key})")
    p = path_for(home, key)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        old = {}  # corrupt: start over rather than crash
    if not isinstance(old, dict):
        old = {}
    rec = {"name": key, "created": old.get("created", now), "updated": now, **record}
    if old.get("filter") is not None:  # keep what the filter used to be, so a tweak can be undone by hand
        rec["previous"] = (old.get("previous") or [])[-4:] + [{"filter": old["filter"], "updated": old.get("updated"),
                                                              "description": old.get("description")}]
    return fsutil.write_atomic(p, json.dumps(rec, indent=1))


def saved(home: Path) -> list[dict]:
    out = []
    for p in sorted(folder(home).glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and isinstance(r.get("filter"), dict):
            r.setdefault("name", p.stem)
            out.append(r)
    return out


# --- usage log: what intensity you actually use, so `filter apply` can default to it ---------------------------

def _log_path(home: Path, name: str) -> Path:
    key = slug(name)
    return folder(home) / ("_usage.jsonl" if key in PACK else f"{key}.uses.jsonl")


def log_use(home: Path, name: str, entry: dict) -> None:
    key = slug(name)
    entry = {**entry, **({"name": key} if key in PACK else {})}
    p = _log_path(home, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def uses(home: Path, name: str) -> list[dict]:
    key = slug(name)
    p = _log_path(home, key)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and (key not in PACK or e.get("name") == key):
            out.append(e)
    return out


def usual_intensity(home: Path, name: str) -> float | None:
    """Median of the last 10 logged intensities, or None with no history."""
    xs = [float(e["intensity"]) for e in uses(home, name)[-10:] if isinstance(e.get("intensity"), (int, float))]
    return round(statistics.median(xs), 3) if xs else None
