from pathlib import Path
from wonhee.config import Settings, REPO_ROOT, secret


def test_defaults_point_inside_repo():
    s = Settings()
    assert s.home == REPO_ROOT / "data"
    assert s.db_path == s.home / "wonhee.db"
    assert s.media_dir == s.home / "media"
    assert s.cost_cap_usd == 1.0
    assert s.whisper_device == "cuda"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WONHEE_HOME", str(tmp_path))
    monkeypatch.setenv("WONHEE_COST_CAP_USD", "0.25")
    s = Settings()
    assert s.home == tmp_path and s.cost_cap_usd == 0.25


def test_secret_reads_only_the_process_environment(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    assert secret("BRAVE_API_KEY") is None
    monkeypatch.setenv("BRAVE_API_KEY", "x")
    assert secret("BRAVE_API_KEY", Settings()) == "x"


def test_llm_defaults():
    s = Settings()
    assert s.llm_model == "openai/gpt-6-luna" and s.llm_base_url == "https://openrouter.ai/api/v1"
