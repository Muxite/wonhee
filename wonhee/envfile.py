"""Entry points load a KEY=VALUE file into os.environ. Library modules never call this."""
from __future__ import annotations

import os
from pathlib import Path

EXPLICIT: list[str] = []  # the --env-file path this process was started with (for a detached child)
LOADED: dict[str, str] = {}  # variable name -> file it came from (names only, never values)


def load_env_file(path: Path | str) -> list[str]:
    """Set variables from path that are not already in os.environ (the caller's environment wins).
    Returns the names it set. A missing file is not an error."""
    p = Path(path)
    if not p.is_file():
        return []
    set_names: list[str] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not name or name in os.environ:
            continue
        os.environ[name] = value
        LOADED[name] = str(p)
        set_names.append(name)
    return set_names


def load_entry_env(explicit: str | Path | None = None) -> list[str]:
    """Entry-point helper: --env-file PATH, else WONHEE_ENV_FILE, else REPO_ROOT/.env."""
    from wonhee.config import REPO_ROOT
    if explicit:
        EXPLICIT[:] = [str(Path(explicit).resolve())]
    path = explicit or os.environ.get("WONHEE_ENV_FILE") or REPO_ROOT / ".env"
    return load_env_file(path)
