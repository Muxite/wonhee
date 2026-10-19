"""Shared ignore rules for the viz scanners: a `.vizkitignore` (gitignore-style) in the scanned root, plus
include/exclude globs passed as params. Dotdirs (.git,
.claude, .venv, ...) are always skipped — they're tooling/session state, never something a
presentation tool should show. Each scanner supplies its own extra default-excluded dir names
and file glob patterns on top of that, since depgraph and diskviz want different defaults
(depgraph hides node_modules/vendor as import noise; diskviz wants them visible).
"""
from __future__ import annotations

from pathlib import Path

import pathspec


def load_ignore_spec(root: Path) -> pathspec.PathSpec | None:
    ignore_file = root / ".vizkitignore"
    if not ignore_file.exists():
        return None
    lines = ignore_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitignore", lines)


def globs_spec(patterns: list[str] | tuple[str, ...] | None) -> pathspec.PathSpec | None:
    """Param globs ("profiles/", "**/ledger_*") as a gitignore-style spec; None when there are none."""
    return pathspec.PathSpec.from_lines("gitignore", list(patterns)) if patterns else None


def is_ignored(rel_path: Path, ignore_spec: pathspec.PathSpec | None, *,
               default_dirs: frozenset[str] = frozenset(),
               default_file_patterns: tuple[str, ...] = (),
               include: pathspec.PathSpec | None = None,
               exclude: pathspec.PathSpec | None = None) -> bool:
    dir_parts = rel_path.parts[:-1]
    if any(part.startswith(".") for part in dir_parts):
        return True
    if any(part in default_dirs for part in dir_parts):
        return True
    if any(rel_path.match(pat) for pat in default_file_patterns):
        return True
    if ignore_spec is not None and ignore_spec.match_file(str(rel_path)):
        return True
    if exclude is not None and exclude.match_file(str(rel_path)):
        return True
    if include is not None and not include.match_file(str(rel_path)):
        return True
    return False
