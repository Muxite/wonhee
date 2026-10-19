"""Walk a path, stat every file for real byte sizes, build a hierarchical size tree.
Unlike depgraph, everything is included by default (node_modules/build/vendor are exactly
the kind of thing a disk-usage view exists to surface) — only dotdirs are skipped by default,
plus an optional .vizkitignore for further excludes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pathspec import PathSpec

from wonhee.edit.viz.ignore import is_ignored, load_ignore_spec

DEFAULT_IGNORE_FILE_PATTERNS: tuple[str, ...] = ()


def _build_node(name: str, path: Path) -> dict[str, Any]:
    return {"name": name, "path": str(path), "value": 0, "ext": None, "children": {}}


def build_tree(root: Path, include: PathSpec | None = None, exclude: PathSpec | None = None,
               only: set[str] | None = None) -> dict[str, Any]:
    ignore_spec = load_ignore_spec(root)
    tree = _build_node(root.name or str(root), Path("."))

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if only is not None and rel.as_posix() not in only:
            continue
        if is_ignored(rel, ignore_spec, default_dirs=frozenset(),
                      default_file_patterns=DEFAULT_IGNORE_FILE_PATTERNS, include=include, exclude=exclude):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue

        node = tree
        parts = rel.parts
        for depth, part in enumerate(parts[:-1]):
            node["value"] += size
            node = node["children"].setdefault(part, _build_node(part, Path(*parts[:depth + 1])))
        node["value"] += size
        leaf_name = parts[-1]
        node["children"][leaf_name] = {
            "name": leaf_name, "path": str(rel), "value": size, "ext": path.suffix or "(none)", "children": {},
        }

    return _finalize(tree)


def _finalize(node: dict[str, Any]) -> dict[str, Any]:
    children = node.pop("children")
    if children:
        node["children"] = [_finalize(c) for c in children.values()]
    return node


def tree_stats(tree: dict[str, Any]) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    by_ext: dict[str, int] = {}

    def walk(node: dict[str, Any]) -> None:
        nonlocal file_count, total_bytes
        children = node.get("children")
        if not children:
            file_count += 1
            total_bytes += node["value"]
            ext = node.get("ext") or "(none)"
            by_ext[ext] = by_ext.get(ext, 0) + node["value"]
            return
        for c in children:
            walk(c)

    walk(tree)
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "by_ext": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])),
    }
