"""All runtime settings. Every field is overridable with a WONHEE_<NAME> env var,
so several wonhee instances can run side by side with different homes."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Project-scoped API keys. Entry points load them from .env; the detached child inherits these.
KEY_NAMES = ("OPENROUTER_API_KEY", "BRAVE_API_KEY", "APIFY_API_KEY")
# Only publish_muksite needs these (muksite's Supabase Storage); doctor reports them as warnings, not failures.
# Optional keys: a feature works without them (doctor reports info, detached runs still receive them).
OPTIONAL_KEY_NAMES = ("GITHUB_TOKEN",)
PUBLISH_KEY_NAMES = ("SUPABASE_URL", "SUPABASE_SERVICE_KEY")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WONHEE_", extra="ignore")

    home: Path = REPO_ROOT / "data"
    creator_path: Path = REPO_ROOT / "creator.yaml"

    llm_model: str = "openai/gpt-6-luna"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_s: int = 240
    llm_price_in_per_mtok: float = 0.25  # gpt-6-luna ceiling (awr ledger, 2026-10-12)
    llm_price_out_per_mtok: float = 0.75

    # On-box model (Ollama) for $0 work such as translation. gemma4:e4b tested best for ko/zh/ja captions (2026-10-17).
    local_llm_url: str = "http://localhost:11440"
    local_llm_model: str = "gemma4:e4b"
    local_llm_timeout_s: int = 180
    local_llm_keep_alive: str = "30m"
    local_llm_ctx: int = 32768  # one fixed window: a changed num_ctx makes Ollama reload the model (~15 s)

    ffmpeg_timeout_s: int = 1800

    apify_price_per_1k: float = 2.70    # used only when Apify does not report usage
    brave_endpoint: str = "https://api.search.brave.com/res/v1/web/search"

    whisper_model: str = "turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # viz_wavegen's docker image, built from wonhee/edit/viz/vendor/wavesim.Dockerfile
    wavesim_image: str = "wonhee-wavesim:dev"

    cost_cap_usd: float = 1.0
    intake_max_bytes: int = 10 * 1024 ** 3
    intake_reserve_bytes: int = 1024 ** 3
    memory_semantic: bool = True
    memory_embedding_model: str = "all-MiniLM-L6-v2"
    creative_download_url: str = ""
    creative_port: int = 17420
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
