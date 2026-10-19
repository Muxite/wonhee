"""Saved filters: the pack that ships with wonhee plus the ones you make (`wonhee filter make`), by name.

A saved filter is one JSON file in <home>/filters/<name>.json holding the numbers, what it was asked to be, and
how it got there (each round's filter, score and the reviewer's notes), so it can be re-applied, tweaked, or
exported as a .cube LUT later. Pack names are reserved so `film-warm` always means the same thing.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from wonhee.edit.filters import PACK, Filter


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", name.strip().lower()).strip("-")
    if not s:
        raise ValueError("a filter needs a name made of letters, digits or dashes")
    return s[:48]


def folder(home: Path) -> Path:
    return home / "filters"


def path_for(home: Path, name: str) -> Path:
    return folder(home) / f"{slug(name)}.json"


def load(home: Path, name: str) -> tuple[Filter, dict]:
    """(filter, record) for a saved filter or a pack preset; saved files cannot shadow the pack."""
    key = slug(name)
    if key in PACK:
        desc, f = PACK[key]
        return f, {"name": key, "source": "pack", "description": desc}
    p = path_for(home, key)
    if not p.exists():
        known = ", ".join([*PACK, *(x["name"] for x in saved(home))])
        raise ValueError(f"no filter called '{name}'. Known: {known}")
    rec = json.loads(p.read_text(encoding="utf-8"))
    return Filter(**rec["filter"]), {**rec, "source": "saved"}


def save(home: Path, name: str, record: dict) -> Path:
    key = slug(name)
    if key in PACK:
        raise ValueError(f"'{key}' is a pack filter; pick another name (e.g. my-{key})")
    p = path_for(home, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rec = {"name": key, "created": old.get("created", now), "updated": now, **record}
    if old:  # keep what the filter used to be, so a tweak can be undone by hand
        rec["previous"] = (old.get("previous") or [])[-4:] + [{"filter": old["filter"], "updated": old["updated"],
                                                              "description": old.get("description")}]
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return p


def saved(home: Path) -> list[dict]:
    out = []
    for p in sorted(folder(home).glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out
