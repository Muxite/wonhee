"""One real 'now' scout. Spends real money (cap 0.75 USD); skipped without keys."""
import pytest

from wonhee.config import Settings, secret

pytestmark = pytest.mark.live


def test_now_scout_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("WONHEE_HOME", str(tmp_path))
    monkeypatch.setenv("WONHEE_COST_CAP_USD", "0.75")
    monkeypatch.setenv("WONHEE_REELS_PER_TREND", "3")
    monkeypatch.setenv("WONHEE_HASHTAG_RESULTS", "15")
    from wonhee import envfile
    from wonhee.config import REPO_ROOT
    envfile.load_env_file(REPO_ROOT / ".env")
    s = Settings()
    if not (secret("APIFY_API_KEY", s) and secret("BRAVE_API_KEY", s)):
        pytest.skip("APIFY_API_KEY / BRAVE_API_KEY not configured")
    from wonhee import service
    from wonhee.db.connect import open_db
    from wonhee.pipeline.deps import Deps

    conn = open_db(s)
    r = service.ask(conn, s, "scout gym humor reels right now, on camera, 20 seconds max")
    for _ in range(5):
        if r.status == "confirmed":
            break
        r = service.ask(conn, s, "go", r.intent_id)
    assert r.status == "confirmed", r.message
    run_id, text = service.go(conn, s, Deps.live(s), r.intent_id)
    print(text)
    status = conn.execute("SELECT status FROM run WHERE id=?", (run_id,)).fetchone()[0]
    assert status in ("done", "weak"), service.show_run(conn, run_id)
