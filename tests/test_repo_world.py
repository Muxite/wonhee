from wonhee.db import repo


def _step(conn):
    conn.execute("INSERT INTO intent VALUES (1,'t','scout',1,'{}','now','now')")
    conn.execute("INSERT INTO project(id,title,kind,status,intent_id,created_at) VALUES (1,'p','scout','open',1,'now')")
    run = repo.start_run(conn, 1, 1.0)
    return repo.start_step(conn, run, "discover", {}, None)


def test_upsert_content_fills_but_never_erases(conn):
    a = repo.upsert_account(conn, "instagram", "u1", "alice")
    c1 = repo.upsert_content(conn, platform="instagram", external_id="X", url="u", account_id=a, caption="hi")
    c2 = repo.upsert_content(conn, platform="instagram", external_id="X", url="u", account_id=a, caption=None, posted_at="2025-10-01T00:00:00+00:00")
    assert c1 == c2
    row = conn.execute("SELECT caption, posted_at FROM content_item WHERE id=?", (c1,)).fetchone()
    assert row["caption"] == "hi" and row["posted_at"].startswith("2025-10-01")


def test_account_handle_change_and_relation_kept(conn):
    a = repo.upsert_account(conn, "instagram", "u1", "alice")
    repo.set_relation(conn, "instagram", "alice", "friend")
    b = repo.upsert_account(conn, "instagram", "u1", "alice_new")
    assert a == b
    row = conn.execute("SELECT handle, relation FROM account WHERE id=?", (a,)).fetchone()
    assert row["handle"] == "alice_new" and row["relation"] == "friend"


def test_hashtags_normalised(conn):
    a = repo.upsert_account(conn, "instagram", "u1", "alice")
    c = repo.upsert_content(conn, platform="instagram", external_id="X", url="u", account_id=a)
    repo.add_hashtags(conn, c, ["#Gym", "gym", "", "Fit", "# gym", "#"])
    assert sorted(r[0] for r in conn.execute("SELECT tag FROM hashtag")) == ["fit", "gym"]


def test_snapshot_negative_becomes_null(conn):
    s = _step(conn)
    a = repo.upsert_account(conn, "instagram", "u1", "alice")
    c = repo.upsert_content(conn, platform="instagram", external_id="X", url="u", account_id=a)
    repo.add_snapshot(conn, c, s, "2026-10-01T00:00:00+00:00", 10, -1, 2, None)
    repo.add_snapshot(conn, c, s, "2026-10-01T00:00:00+00:00", 99, 1, 1, 1)  # same instant: ignored
    row = conn.execute("SELECT plays, likes FROM metric_snapshot").fetchone()
    assert row["plays"] == 10 and row["likes"] is None
