import sqlite3
import pytest
from wonhee.db.connect import connect, tx
from wonhee.db.migrate import migrate


def test_migrate_fresh_and_idempotent(tmp_path):
    c = connect(tmp_path / "a.db")
    assert migrate(c) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert migrate(c) == []


def test_all_tables_strict(conn):
    rows = conn.execute("SELECT name, strict FROM pragma_table_list WHERE schema='main' AND type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    assert rows and all(r["strict"] == 1 for r in rows), [r["name"] for r in rows if not r["strict"]]


def test_fk_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO media(blob_sha256, kind) VALUES (?, 'video')", ("a" * 64,))


def test_account_platform_must_match_content(conn):
    conn.execute("INSERT INTO account(id, platform, external_id, handle) VALUES (1,'instagram','u1','h')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO content_item(platform, external_id, url, account_id) VALUES ('tiktok','x','u',1)")


def test_shot_citation_exactly_one_target(conn):
    conn.executescript("""
      INSERT INTO intent(id, raw_text, schema_name, schema_version, compiled_json, created_at) VALUES (1,'t','scout',1,'{}','now');
      INSERT INTO project(id, title, kind, status, intent_id, created_at) VALUES (1,'p','scout','open',1,'now');
      INSERT INTO run(id, project_id, orchestrator, status, cost_cap_usd, started_at) VALUES (1,1,'fixed','running',1,'now');
      INSERT INTO step(id, run_id, stage, attempt, started_at) VALUES (1,1,'shotlist',1,'now');
      INSERT INTO shotlist(id, project_id, version, created_by_step) VALUES (1,1,1,1);
      INSERT INTO shot(shotlist_id, ord, duration_ms, framing, action) VALUES (1,1,1000,'wide','x');
    """)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO shot_citation(shotlist_id, shot_ord) VALUES (1,1)")
    conn.executescript("""
      INSERT INTO account(id, platform, external_id, handle) VALUES (1,'instagram','u1','h');
      INSERT INTO content_item(id, platform, external_id, url, account_id) VALUES (1,'instagram','x','u',1);
      INSERT INTO web_source(id, url, retrieved_at) VALUES (1,'http://a','now');
    """)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO shot_citation(shotlist_id, shot_ord, content_id, web_source_id) VALUES (1,1,1,1)")
    conn.execute("INSERT INTO shot_citation(shotlist_id, shot_ord, content_id) VALUES (1,1,1)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO shot_citation(shotlist_id, shot_ord, content_id) VALUES (1,1,1)")


def test_llm_call_exactly_one_owner(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO llm_call(model, schema_name, tokens_in, tokens_out, cost_usd, valid, created_at) VALUES ('m','s',1,1,0,1,'now')")


def test_flag_codes_seeded(conn):
    codes = {r[0] for r in conn.execute("SELECT code FROM flag_code")}
    assert {"LOW_SAMPLE", "SCHEMA_FAIL", "COST_CAP", "EXCEPTION", "MISSING_KEY"} <= codes


def test_tx_rolls_back(conn):
    with pytest.raises(RuntimeError):
        with tx(conn):
            conn.execute("INSERT INTO hashtag(tag) VALUES ('a')")
            raise RuntimeError
    assert conn.execute("SELECT count(*) FROM hashtag").fetchone()[0] == 0


def test_hamming_registered(conn):
    assert conn.execute("SELECT hamming('ffffffffffffffff','0000000000000000')").fetchone()[0] == 64


def test_project_rebuild_keeps_rows_and_references(tmp_path):
    import shutil
    from wonhee.db.migrate import MIGRATIONS
    old = tmp_path / "old"
    old.mkdir()
    for f in MIGRATIONS.iterdir():
        if f.name[:4] in ("0001", "0002", "0003", "0004"):
            shutil.copy(f, old / f.name)
    c = connect(tmp_path / "b.db")
    migrate(c, old)
    c.executescript("""
      INSERT INTO intent VALUES (1,'t','scout',1,'{}','now',NULL);
      INSERT INTO project(id, title, kind, status, intent_id, created_at) VALUES (7,'p','scout','open',1,'now');
      INSERT INTO run(id, project_id, orchestrator, status, cost_cap_usd, started_at) VALUES (1,7,'fixed','done',1,'now');
    """)
    assert migrate(c) == [5, 6, 7, 8, 9]
    assert c.execute("SELECT title FROM project WHERE id=7").fetchone()[0] == "p"
    assert c.execute("SELECT count(*) FROM pragma_foreign_key_check").fetchone()[0] == 0
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    c.execute("INSERT INTO project(title, kind, created_at) VALUES ('lib','library','now')")
    c.execute("SELECT * FROM v_project_overview").fetchall()
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("DELETE FROM project WHERE id=7")  # run still references it: FKs enforced on the new table


def test_0006_rebuild_keeps_projects_and_accepts_chain(tmp_path):
    import shutil
    from wonhee.db.migrate import MIGRATIONS
    old = tmp_path / "old"
    old.mkdir()
    for f in MIGRATIONS.iterdir():
        if f.name[:4] in ("0001", "0002", "0003", "0004", "0005"):
            shutil.copy(f, old / f.name)
    c = connect(tmp_path / "c.db")
    migrate(c, old)
    c.executescript("""
      INSERT INTO project(id, title, kind, created_at) VALUES (1,'s','scout','now');
      INSERT INTO project(id, title, kind, created_at) VALUES (2,'l','library','now');
    """)
    assert migrate(c) == [6, 7, 8, 9]
    assert [r[0] for r in c.execute("SELECT title FROM project ORDER BY id")] == ["s", "l"]
    assert c.execute("SELECT count(*) FROM pragma_foreign_key_check").fetchone()[0] == 0
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    c.execute("INSERT INTO project(title, kind, created_at) VALUES ('c','chain','now')")
    c.execute("SELECT * FROM v_project_overview").fetchall()
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO project(title, kind, created_at) VALUES ('x','bogus','now')")
    codes = {r[0] for r in c.execute("SELECT code FROM flag_code")}
    assert {"CACHED", "TRANSLATION_GAP", "NO_TEXT"} <= codes
