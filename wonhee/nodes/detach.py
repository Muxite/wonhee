"""Start `wonhee chain exec RUN` as a detached child so a long chain outlives the caller."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from wonhee import envfile
from wonhee.config import KEY_NAMES, REPO_ROOT, Settings
from wonhee.procenv import clean_env


PASS_THROUGH = ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE", "XDG_CACHE_HOME")


def child_env() -> dict[str, str]:
    """The same home, keys and GPU/model-cache settings as the parent (it already loaded .env into
    os.environ), nothing else."""
    env = clean_env()
    for k in PASS_THROUGH:
        if k in os.environ:
            env[k] = os.environ[k]
    for k, v in os.environ.items():
        if k.startswith("WONHEE_") or k in KEY_NAMES:
            env[k] = str(Path(v).resolve()) if k.startswith("WONHEE_") and v and Path(v).exists() \
                and not Path(v).is_absolute() else v
    return env


def _env_file_args() -> list[str]:
    """--env-file is a global option, so it goes before the subcommand."""
    return ["--env-file", envfile.EXPLICIT[0]] if envfile.EXPLICIT else []


def spawn(settings: Settings, run_id: int) -> Path:
    env = child_env()
    env["WONHEE_HOME"] = str(Path(settings.home).resolve())
    log = settings.home / "logs" / f"run-{run_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        try:
            subprocess.Popen([sys.executable, "-m", "wonhee.cli", *_env_file_args(), "chain", "exec", str(run_id)],
                             env=env, cwd=REPO_ROOT, start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=out, stderr=out)
        except Exception as e:
            _mark_failed(settings, run_id, e)
            raise
    return log


def _mark_failed(settings: Settings, run_id: int, err: Exception) -> None:
    """The child never started: do not leave the run 'running' forever."""
    from wonhee.db import repo
    from wonhee.db.connect import open_db

    conn = open_db(settings)
    try:
        step_id = repo.start_step(conn, run_id, "spawn", {}, None)
        repo.add_flag(conn, step_id, "EXCEPTION", f"could not start background process: {err}"[:300])
        repo.finish_step(conn, step_id, "failed")
        repo.finish_run(conn, run_id, "failed")
    finally:
        conn.close()
