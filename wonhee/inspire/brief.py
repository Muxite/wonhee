"""The creator's own brief: idea lists, persona notes, phrase banks they wrote (`<home>/brief/*.md`).

Kept apart from inspiration on purpose, so every idea can say where it came from. Each bullet (or paragraph) is one
item with a stable id from its text (`b-` + 5 hex), so a reference survives the file being reordered or extended.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_BULLET = re.compile(r"^\s*(?:[*\-+]|\d+[.)])\s+(.*\S)\s*$")


@dataclass(frozen=True)
class BriefItem:
    id: str
    source: str  # file stem, e.g. 2026-10-17-niche-ideas
    section: str
    text: str


def item_id(text: str) -> str:
    norm = " ".join(text.lower().split())
    return "b-" + hashlib.sha1(norm.encode()).hexdigest()[:5]


def parse(text: str, source: str) -> list[BriefItem]:
    items, section, seen = [], "", set()
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            section = line.lstrip("# ").strip()
            continue
        m = _BULLET.match(line)
        body = m.group(1) if m else line.strip()
        if not body or (not m and line.startswith("    ")):
            continue
        iid = item_id(body)
        if iid not in seen:
            seen.add(iid)
            items.append(BriefItem(iid, source, section, body))
    return items


def load(brief_dir: Path) -> list[BriefItem]:
    """Every item from every brief file, oldest file first. Empty if there is no brief."""
    out, seen = [], set()
    for f in sorted(Path(brief_dir).glob("*.md")):
        for it in parse(f.read_text(encoding="utf-8"), f.stem):
            if it.id not in seen:
                seen.add(it.id)
                out.append(it)
    return out


def render(items: list[BriefItem]) -> str:
    """One line per item, grouped under its source and section, for the ideate prompt."""
    out, last = [], None
    for it in items:
        if (it.source, it.section) != last:
            last = (it.source, it.section)
            out.append(f"\n[{it.source} / {it.section or 'intro'}]")
        out.append(f"{it.id}: {it.text}")
    return "\n".join(out).strip()
