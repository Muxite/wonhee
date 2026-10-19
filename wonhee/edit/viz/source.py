"""Pick what a render looks at: a path as it is now, or the same path at an older commit.

`snapshot` uses `git archive` into a temp dir, so "the repo as it was in March 2026" is a param instead of a hand
step, and the working tree (uncommitted files, local data) is never read for a past-rev render. A path inside a
repo keeps its place: rendering repo/backend at a rev archives only backend/.

A working-tree render of a path inside a git repo draws only the files git would track (tracked, plus untracked
ones not ignored), so gitignored data, virtualenvs and submodule checkouts stay out of the picture, the same set a
past-rev render sees. `tracked=False` draws everything on disk, the old vizkit behaviour.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path


class SourceError(ValueError):
    pass


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SourceError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def resolve_rev(path: Path, rev: str) -> str:
    """Full commit SHA for `rev` in the repo holding `path` (recorded so a render can be reproduced)."""
    return _git(path, "rev-parse", "--verify", f"{rev}^{{commit}}")


def git_files(path: Path) -> set[str] | None:
    """Paths (relative to `path`) git would track there, or None when `path` is not inside a git work tree."""
    r = subprocess.run(["git", "-C", str(path), "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                        "--", "."], capture_output=True)
    if r.returncode != 0:
        return None
    return {f for f in r.stdout.decode(errors="replace").split("\0") if f}


@dataclasses.dataclass(frozen=True)
class Source:
    root: Path               # directory to scan
    sha: str | None          # resolved commit for a past-rev render
    only: set[str] | None    # when set, scan just these relative paths (working tree, git-tracked)


@contextlib.contextmanager
def snapshot(path: Path, rev: str | None, tracked: bool = True) -> Iterator[Source]:
    """rev=None scans `path` in place; otherwise a git-archive copy of it at `rev`."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise SourceError(f"{path}: not a directory")
    if rev is None:
        yield Source(path, None, git_files(path) if tracked else None)
        return
    sha = resolve_rev(path, rev)
    prefix = _git(path, "rev-parse", "--show-prefix")  # "backend/" when path is a subdir, "" at the repo root
    data = subprocess.run(["git", "-C", str(path), "archive", "--format=tar", sha, "--", "."],
                          capture_output=True)
    if data.returncode != 0:
        raise SourceError(f"git archive {rev}: {data.stderr.decode(errors='replace').strip()}")
    with tempfile.TemporaryDirectory(prefix="wonhee-viz-") as tmp:
        with tarfile.open(fileobj=io.BytesIO(data.stdout)) as tar:
            tar.extractall(tmp, filter="data")
        root = Path(tmp) / prefix if prefix else Path(tmp)
        yield Source(root if root.is_dir() else Path(tmp), sha, None)
