"""All runtime settings. Every field is overridable with a WONHEE_<NAME> env var,
so several wonhee instances can run side by side with different homes."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Project-scoped API keys. Entry points load them from .env; the detached child inherits these.
KEY_NAMES = ("OPENROUTER_API_KEY", "BRAVE_API_KEY", "APIFY_API_KEY")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WONHEE_", extra="ignore")

    home: Path = REPO_ROOT / "data"
    creator_path: Path = REPO_ROOT / "creator.yaml"

    llm_model: str = "openai/gpt-6-luna"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_s: int = 240
    llm_price_in_per_mtok: float = 0.25  # gpt-6-luna ceiling (awr ledger, 2026-10-12)
    llm_price_out_per_mtok: float = 0.75

    apify_price_per_1k: float = 2.70    # used only when Apify does not report usage
    brave_endpoint: str = "https://api.search.brave.com/res/v1/web/search"

    whisper_model: str = "turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    cost_cap_usd: float = 1.0
    reels_per_trend: int = 5
    hashtag_results: int = 30

    @property
    def db_path(self) -> Path:
        return self.home / "wonhee.db"

    @property
    def media_dir(self) -> Path:
        return self.home / "media"


def get_settings() -> Settings:
    return Settings()


def secret(name: str, settings: Settings | None = None) -> str | None:
    """A key from the process environment only. (entry points load .env into it; see wonhee.envfile)"""
    return os.environ.get(name) or None
