"""The environment handed to every subprocess: no API keys or tokens leak to ffmpeg or yt-dlp."""
from __future__ import annotations

import os


def clean_env() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", os.defpath)}
    for k in ("LANG", "HOME", "TMPDIR"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env
