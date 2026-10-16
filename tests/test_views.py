import pytest

VIEWS = ["v_content_latest", "v_content_growth", "v_trend_popularity", "v_project_overview",
         "v_post_vs_trend", "v_shot_lineage", "v_near_duplicates", "v_run_cost",
         "v_current_steps", "v_media_style"]


@pytest.fixture
def seeded(conn):
    conn.executescript("""
      INSERT INTO intent VALUES (1,'t','scout',1,'{}','now',NULL);
      INSERT INTO project VALUES (1,'p','scout','open',1,'now');
      INSERT INTO run VALUES (1,1,'fixed','running',1.0,'now',NULL);
      INSERT INTO step(id,run_id,stage,attempt,started_at,cost_usd) VALUES (1,1,'discover',1,'t',0.2);
      INSERT INTO step(id,run_id,stage,attempt,started_at,cost_usd,supersedes_step) VALUES (2,1,'discover',2,'t',0.3,1);
      INSERT INTO account VALUES (1,'instagram','u1','alice','external');
      INSERT INTO account VALUES (2,'instagram','u2','me','ours');
      INSERT INTO content_item(id,platform,external_id,url,account_id) VALUES (1,'instagram','A','ua',1),(2,'instagram','B','ub',1),(3,'instagram','C','uc',1),(9,'instagram','Z','uz',2);
      INSERT INTO metric_snapshot VALUES (1,'2026-10-01T00:00:00+00:00',100,NULL,NULL,NULL,1);
      INSERT INTO metric_snapshot VALUES (1,'2026-10-02T00:00:00+00:00',300,NULL,NULL,NULL,2);
      INSERT INTO metric_snapshot VALUES (2,'2026-10-02T00:00:00+00:00',50,NULL,NULL,NULL,2);
      INSERT INTO metric_snapshot VALUES (3,'2026-10-02T00:00:00+00:00',NULL,NULL,NULL,NULL,2);
      INSERT INTO metric_snapshot VALUES (9,'2026-10-02T00:00:00+00:00',400,NULL,NULL,NULL,2);
      INSERT INTO trend VALUES (1,'t1',NULL,NULL,NULL,2);
      INSERT INTO trend_example VALUES (1,1),(1,2),(1,3);
      INSERT INTO project_trend VALUES (1,1,'primary');
      INSERT INTO project_output VALUES (1,9);
    """)
    return conn


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_runs(conn, view):
    conn.execute(f"SELECT * FROM {view}").fetchall()


def test_latest_snapshot_wins(seeded):
    r = seeded.execute("SELECT plays FROM v_content_latest WHERE content_id=1").fetchone()
    assert r["plays"] == 300


def test_growth_delta(seeded):
    rows = seeded.execute("SELECT d_plays, d_seconds FROM v_content_growth WHERE content_id=1 ORDER BY observed_at").fetchall()
    assert rows[0]["d_plays"] is None and rows[1]["d_plays"] == 200 and round(rows[1]["d_seconds"]) == 86400


def test_median_ignores_null_plays(seeded):
    r = seeded.execute("SELECT example_count, median_plays FROM v_trend_popularity WHERE trend_id=1").fetchone()
    assert r["example_count"] == 3 and r["median_plays"] == 175.0  # median of 300, 50


def test_post_vs_trend_ratio(seeded):
    r = seeded.execute("SELECT relation, ratio FROM v_post_vs_trend WHERE content_id=9").fetchone()
    assert r["relation"] == "ours" and abs(r["ratio"] - 400 / 175) < 1e-9


def test_current_steps_excludes_superseded(seeded):
    ids = [r["id"] for r in seeded.execute("SELECT id FROM v_current_steps WHERE stage='discover'")]
    assert ids == [2]


def test_run_cost_sums_attempts(seeded):
    r = seeded.execute("SELECT attempts, cost_usd FROM v_run_cost WHERE run_id=1 AND stage='discover'").fetchone()
    assert r["attempts"] == 2 and abs(r["cost_usd"] - 0.5) < 1e-9


def test_near_duplicates(conn):
    for sha in ("a" * 64, "b" * 64):
        conn.execute("INSERT INTO blob VALUES (?,1,'video/mp4','t')", (sha,))
        conn.execute("INSERT INTO media(blob_sha256,kind) VALUES (?,'video')", (sha,))
    conn.execute("INSERT INTO media_phash VALUES (?,0,'ffffffffffffffff')", ("a" * 64,))
    conn.execute("INSERT INTO media_phash VALUES (?,0,'fffffffffffffff0')", ("b" * 64,))
    r = conn.execute("SELECT distance FROM v_near_duplicates").fetchone()
    assert r["distance"] == 4


def test_median_odd_count(seeded):
    seeded.executescript("INSERT INTO trend VALUES (2,'t2',NULL,NULL,NULL,2); INSERT INTO trend_example VALUES (2,1),(2,2),(2,9);")
    r = seeded.execute("SELECT median_plays FROM v_trend_popularity WHERE trend_id=2").fetchone()
    assert r["median_plays"] == 300.0


def test_project_overview(seeded):
    seeded.executescript("""
      INSERT INTO project VALUES (2,'q','scout','open',1,'now');
      INSERT INTO project_reference VALUES (1,1,'inspiration'),(1,2,'style_ref');
    """)
    r = seeded.execute("SELECT * FROM v_project_overview WHERE project_id=1").fetchone()
    assert r["primary_trend"] == "t1" and r["reference_count"] == 2
    assert r["output_count"] == 1 and r["output_plays"] == 400
    e = seeded.execute("SELECT * FROM v_project_overview WHERE project_id=2").fetchone()
    assert e["primary_trend"] is None and e["output_count"] == 0 and e["output_plays"] is None


def test_shot_lineage(seeded):
    seeded.executescript("""
      INSERT INTO web_source(id,url,retrieved_at) VALUES (1,'http://src','now');
      INSERT INTO shotlist VALUES (1,1,1,2);
      INSERT INTO shot(shotlist_id,ord,duration_ms,framing,action) VALUES (1,1,1000,'f','a'),(1,2,1000,'f','a');
      INSERT INTO shot_citation(shotlist_id,shot_ord,content_id) VALUES (1,1,1);
      INSERT INTO shot_citation(shotlist_id,shot_ord,web_source_id) VALUES (1,2,1);
    """)
    rows = seeded.execute("SELECT * FROM v_shot_lineage ORDER BY shot_ord").fetchall()
    assert rows[0]["content_url"] == "ua" and rows[0]["source_url"] is None
    assert rows[1]["source_url"] == "http://src" and rows[1]["content_url"] is None
    assert all(r["step_id"] == 2 and r["run_id"] == 1 and r["version"] == 1 for r in rows)


def test_media_style(seeded):
    for sha in ("c" * 64, "d" * 64):
        seeded.execute("INSERT INTO blob VALUES (?,1,'video/mp4','t')", (sha,))
        seeded.execute("INSERT INTO media(blob_sha256,kind,duration_ms) VALUES (?,'video',3000)", (sha,))
        seeded.execute("INSERT INTO media_analysis(media_sha256,analyzer_version,step_id,first_word_ms,language) VALUES (?,1,1,200,'en')", (sha,))
    for i in range(3):
        seeded.execute("INSERT INTO media_shot(media_sha256,analyzer_version,ord,start_ms,end_ms) VALUES (?,1,?,?,?)",
                       ("c" * 64, i + 1, i * 1000, (i + 1) * 1000))
    r = seeded.execute("SELECT * FROM v_media_style WHERE media_sha256=?", ("c" * 64,)).fetchone()
    assert r["shot_count"] == 3 and abs(r["cuts_per_sec"] - 2 / 3) < 1e-3 and r["first_word_ms"] == 200
    z = seeded.execute("SELECT * FROM v_media_style WHERE media_sha256=?", ("d" * 64,)).fetchone()
    assert z["shot_count"] == 0 and z["cuts_per_sec"] is None
