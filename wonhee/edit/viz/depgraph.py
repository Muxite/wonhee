"""Walk a repo, extract per-file imports and lines of code with regexes, and resolve import strings to files
in the same tree. Best effort on purpose: the graph is for a picture of a codebase's shape, not a correct
dependency analysis.

Python resolution indexes each module three ways: by its repo path, with a leading src/ or lib/ dropped, and
from its package root (the parent of the topmost directory holding __init__.py). The last one is what makes
`from awork.x import y` resolve when the package lives under backend/ (awork's first render had no edges).
`from pkg import mod` also links to pkg/mod.py when that file exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pathspec import PathSpec

from wonhee.edit.viz.ignore import is_ignored, load_ignore_spec
from wonhee.edit.viz.graph import Edge, Graph, Node

DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "dist", "build", "vendor", "venv", ".venv",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".next", ".turbo", "target", "out",
    "coverage", ".idea", ".vscode", "site-packages", "egg-info",
}
DEFAULT_IGNORE_FILE_PATTERNS = ("*.lock", "*.min.js", "*.map", "package-lock.json", "*.svg", "*.png", "*.jpg")

LANGUAGES = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cu": "cuda", ".cuh": "cuda",
    ".rs": "rust",
}

# Each pattern's capture group 1 is the raw imported-thing string.
IMPORT_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [
        re.compile(r"^\s*from\s+([.\w]+)\s+import\b", re.MULTILINE),
        re.compile(r"^\s*import\s+([.\w]+)", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"""\bimport\s+(?:[\w*{}\s,]+\sfrom\s+)?["']([^"']+)["']"""),
        re.compile(r"""\brequire\(\s*["']([^"']+)["']\s*\)"""),
    ],
    "typescript": [
        re.compile(r"""\bimport\s+(?:type\s+)?(?:[\w*{}\s,]+\sfrom\s+)?["']([^"']+)["']"""),
        re.compile(r"""\brequire\(\s*["']([^"']+)["']\s*\)"""),
    ],
    "c": [re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)],
    "cpp": [re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)],
    "cuda": [re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)],
    "rust": [
        re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE),
        re.compile(r"^\s*mod\s+(\w+)\s*;", re.MULTILINE),
    ],
}


@dataclass
class ScannedFile:
    path: Path
    language: str
    loc: int
    raw_imports: list[str]


# `from X import a, b` (or a parenthesised list): the imported names, tried as submodules X.a, X.b
_PY_FROM_NAMES = re.compile(r"^\s*from\s+([.\w]+)\s+import\s+(\([^)]*\)|[^\n#]+)", re.MULTILINE)


def _python_submodules(text: str) -> list[str]:
    out = []
    for module, names in _PY_FROM_NAMES.findall(text):
        for name in re.findall(r"\b(\w+)\b(?:\s+as\s+\w+)?", names.strip("()")):
            if name == "as":
                continue
            out.append(f"{module}{name}" if module.endswith(".") else f"{module}.{name}")
    return out


def _count_loc(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def scan_files(root: Path, include: PathSpec | None = None, exclude: PathSpec | None = None,
               only: set[str] | None = None) -> list[ScannedFile]:
    ignore_spec = load_ignore_spec(root)
    results: list[ScannedFile] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if only is not None and rel.as_posix() not in only:
            continue
        if is_ignored(rel, ignore_spec, default_dirs=DEFAULT_IGNORE_DIRS,
                      default_file_patterns=DEFAULT_IGNORE_FILE_PATTERNS, include=include, exclude=exclude):
            continue
        language = LANGUAGES.get(path.suffix)
        if language is None:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        raw_imports: list[str] = []
        for pattern in IMPORT_PATTERNS.get(language, []):
            raw_imports.extend(pattern.findall(text))
        if language == "python":
            raw_imports.extend(_python_submodules(text))
        results.append(ScannedFile(path=path, language=language, loc=_count_loc(text), raw_imports=raw_imports))
    return results


def build_python_module_index(files: list[ScannedFile], root: Path) -> dict[str, str]:
    """Map dotted module path -> rel file path (see the module docstring for the three ways each is indexed)."""
    package_dirs = {f.path.relative_to(root).parent for f in files
                    if f.language == "python" and f.path.name == "__init__.py"}
    # the scan root may itself sit inside packages (drawing repo/agent/app, whose code says `from agent.app.x`):
    # walk up on disk while each parent is a package, and also index every module under that dotted prefix
    above: list[str] = []
    d = root
    while (d / "__init__.py").is_file() and d.parent != d:
        above.insert(0, d.name)
        d = d.parent
    index: dict[str, str] = {}
    for f in files:
        if f.language != "python":
            continue
        rel = f.path.relative_to(root)
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        variants = [parts]
        if parts and parts[0] in ("src", "lib"):
            variants.append(parts[1:])
        if rel.parent in package_dirs:
            # walk up while each dir is a package; the first one that is not is the import root
            top = rel.parent
            while top in package_dirs and top != Path("."):
                top = top.parent
            variants.append(parts[len(top.parts):])
        if above:
            variants.append(above + parts)
        for variant in variants:
            if not variant:
                continue
            index.setdefault(".".join(variant), str(rel))
    return index


def _resolve_python(file: ScannedFile, root: Path, module_index: dict[str, str]) -> list[str]:
    resolved = []
    file_dir = file.path.parent
    for raw in file.raw_imports:
        dots = len(raw) - len(raw.lstrip("."))
        module = raw.lstrip(".")
        if dots == 0:
            hit = module_index.get(module)
            if hit:
                resolved.append(hit)
            continue
        base_dir = file_dir
        for _ in range(max(dots - 1, 0)):
            base_dir = base_dir.parent
        parts = module.split(".") if module else []
        candidate_dir = base_dir.joinpath(*parts) if parts else base_dir
        for c in (candidate_dir.with_suffix(".py"), candidate_dir / "__init__.py"):
            try:
                rel = str(c.relative_to(root))
            except ValueError:
                continue
            if rel in module_index.values() or c.exists():
                resolved.append(rel)
                break
    return resolved


def _resolve_relative_module(file: ScannedFile, root: Path, raw: str, extensions: list[str],
                              by_rel: dict[str, Path]) -> str | None:
    if not raw.startswith("."):
        return None
    base = (file.path.parent / raw).resolve()
    candidates = [base.with_suffix(ext) for ext in extensions]
    candidates += [base / f"index{ext}" for ext in extensions]
    for c in candidates:
        try:
            rel = str(c.relative_to(root))
        except ValueError:
            continue
        if rel in by_rel:
            return rel
    return None


def _resolve_c_family(file: ScannedFile, root: Path, by_rel: dict[str, Path]) -> list[str]:
    resolved = []
    for raw in file.raw_imports:
        candidates = [file.path.parent / raw, root / raw]
        for c in candidates:
            try:
                rel = str(c.resolve().relative_to(root))
            except (ValueError, OSError):
                continue
            if rel in by_rel:
                resolved.append(rel)
                break
    return resolved


def _resolve_rust(file: ScannedFile, root: Path, by_rel: dict[str, Path]) -> list[str]:
    resolved = []
    for raw in file.raw_imports:
        first_segment = raw.split("::")[0]
        candidates = [
            file.path.parent / f"{first_segment}.rs",
            file.path.parent / first_segment / "mod.rs",
        ]
        for c in candidates:
            try:
                rel = str(c.relative_to(root))
            except ValueError:
                continue
            if rel in by_rel:
                resolved.append(rel)
                break
    return resolved


def build_graph(root: Path, include: PathSpec | None = None, exclude: PathSpec | None = None,
                only: set[str] | None = None) -> Graph:
    files = scan_files(root, include, exclude, only)
    by_rel = {str(f.path.relative_to(root)): f.path for f in files}
    python_index = build_python_module_index(files, root)

    nodes = [Node(id=str(f.path.relative_to(root)), path=str(f.path.relative_to(root)), loc=f.loc,
                   language=f.language) for f in files]

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for f in files:
        rel = str(f.path.relative_to(root))
        targets: list[str] = []
        if f.language == "python":
            targets = _resolve_python(f, root, python_index)
        elif f.language in ("javascript", "typescript"):
            for raw in f.raw_imports:
                t = _resolve_relative_module(f, root, raw, [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"], by_rel)
                if t:
                    targets.append(t)
        elif f.language in ("c", "cpp", "cuda"):
            targets = _resolve_c_family(f, root, by_rel)
        elif f.language == "rust":
            targets = _resolve_rust(f, root, by_rel)

        for t in targets:
            if t == rel or t not in by_rel:  # a relative import can land on an excluded or non-code file
                continue
            key = (rel, t)
            if key in seen:
                continue
            seen.add(key)
            edges.append(Edge(from_=rel, to=t))

    return Graph(nodes=nodes, edges=edges)
