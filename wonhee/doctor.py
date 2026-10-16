"""Environment checks. Details never contain '=' so a secret value can never be echoed by accident."""
from __future__ import annotations

import shutil
from dataclasses import dataclass

from wonhee.config import Settings, secret

KEYS = ("OPENROUTER_API_KEY", "BRAVE_API_KEY", "APIFY_API_KEY")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    severity: str = "required"  # "required" | "warning"

    def __post_init__(self):
        self.detail = self.detail.replace("=", ":")


def _moeka_check() -> Check:
    try:
        import moeka  # noqa: F401
    except Exception as e:  # noqa: BLE001 - any import failure is a failed check
        return Check("moeka import", False, f"import moeka failed: {type(e).__name__}; run uv sync --all-extras")
    return Check("moeka import", True, "moeka imports (pinned pip dependency)")


def _key_check(name: str) -> Check:
    from wonhee import envfile
    if not secret(name):
        return Check(f"key {name}", False, "not set; put it in .env (cp .env.example .env) or export it")
    src = envfile.LOADED.get(name)
    return Check(f"key {name}", True, f".env ({src})" if src else "shell env")


def _cuda_check(settings: Settings) -> Check:
    if settings.whisper_device != "cuda":
        return Check("cuda", True, f"whisper device is {settings.whisper_device}; CUDA not needed")
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
    except Exception as e:  # noqa: BLE001 - any import/driver failure is a failed check
        return Check("cuda", False, f"ctranslate2 could not query CUDA: {e}; "
                     "set WONHEE_WHISPER_DEVICE to cpu and WONHEE_WHISPER_COMPUTE_TYPE to int8 to use the CPU")
    if n <= 0:
        return Check("cuda", False, "no CUDA device visible to ctranslate2; set WONHEE_WHISPER_DEVICE to cpu "
                     "and WONHEE_WHISPER_COMPUTE_TYPE to int8 to use the CPU")
    try:
        from wonhee.analyze import speech
        speech._preload_cuda_libs()
    except RuntimeError as e:
        return Check("cuda", False, str(e))
    return Check("cuda", True, f"{n} CUDA device(s); cuBLAS and cuDNN load")


def checks(settings: Settings) -> list[Check]:
    out: list[Check] = []
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        out.append(Check(tool, bool(path), path or f"{tool} not found on PATH; install ffmpeg"))
    out.append(_moeka_check())
    out += [_key_check(k) for k in KEYS]
    out.append(_cuda_check(settings))
    try:
        from wonhee.db.connect import open_db
        conn = open_db(settings)
        n = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        conn.close()
        out.append(Check("database", True, f"{settings.db_path} migrated ({n} tables)"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("database", False, f"{settings.db_path}: {e}"))
    has_creator = settings.creator_path.is_file()
    out.append(Check("creator.yaml", has_creator,
                     str(settings.creator_path) if has_creator else
                     f"missing at {settings.creator_path}; copy creator.example.yaml and edit it", "warning"))
    priced = settings.llm_price_in_per_mtok > 0 and settings.llm_price_out_per_mtok > 0
    out.append(Check("llm prices", priced,
                     "configured" if priced else "LLM prices are 0 so LLM spend counts as free; set "
                     "WONHEE_LLM_PRICE_IN_PER_MTOK and WONHEE_LLM_PRICE_OUT_PER_MTOK", "warning"))
    return out
