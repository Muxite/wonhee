"""GitHub for the slop machine: repo search, one repo's facts, and a shallow clone.

Public data only. `GITHUB_TOKEN` (from the project env, never ~/) raises the rate limit; without it the calls are
anonymous. The clone runs with procenv.clean_env() so the token never reaches git.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import httpx

from wonhee.fsutil import run_timed
from wonhee.procenv import clean_env

API = "https://api.github.com"
MAX_SIZE_KB = 50_000  # the API reports `size` in KB: skip repos over 50 MB
CLONE_TIMEOUT_S = 120


class RateLimited(RuntimeError):
    pass


class CloneFailed(RuntimeError):
    pass


class GitHubClient:
    def __init__(self, token: str | None = None, client: httpx.Client | None = None):
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = client or httpx.Client(timeout=30, headers=headers)

    def _get(self, path: str, **kw) -> httpx.Response:
        r = self.http.get(API + path, **kw)
        if r.status_code in (403, 429) and (r.headers.get("x-ratelimit-remaining") == "0" or r.status_code == 429):
            raise RateLimited(f"GitHub rate limit hit on {path}; set GITHUB_TOKEN in the project env")
        return r

    def search(self, query: str, page: int = 1, per_page: int = 30) -> list[dict]:
        r = self._get("/search/repositories", params={"q": query, "sort": "updated", "order": "desc",
                                                      "page": page, "per_page": per_page})
        r.raise_for_status()
        return r.json().get("items", [])

    def repo(self, full_name: str) -> dict:
        r = self._get(f"/repos/{full_name}")
        r.raise_for_status()
        return r.json()

    def languages(self, full_name: str) -> dict[str, int]:
        r = self._get(f"/repos/{full_name}/languages")
        return r.json() if r.status_code == 200 else {}

    def readme(self, full_name: str) -> str:
        r = self._get(f"/repos/{full_name}/readme", headers={"Accept": "application/vnd.github.raw+json"})
        return r.text if r.status_code == 200 else ""


def clone(full_name: str, dest: Path, timeout: float = CLONE_TIMEOUT_S) -> Path:
    """`git clone --depth 1` into dest (replacing a stale copy). Raises CloneFailed."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = {**clean_env(), "GIT_TERMINAL_PROMPT": "0"}
    try:
        p = run_timed(["git", "clone", "--depth", "1", "--quiet", f"https://github.com/{full_name}.git", str(dest)],
                      timeout=timeout, env=env, capture_output=True, text=True)
    except RuntimeError as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneFailed(str(e)) from e
    if p.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneFailed(f"git clone {full_name}: {p.stderr.strip()[-300:]}")
    return dest
