import sqlite3
import pytest
from wonhee.db import queries, repo

KEY = "a" * 64
SHA1, SHA2 = "1" * 64, "2" * 64


@pytest.fixture
def step(conn):
    conn.execute("INSERT INTO project(id,title,kind,created_at) VALUES (1,'p','chain','now')")
    run = repo.start_run(conn, 1, 1.0)
    return repo.start_step(conn, run, "n1", {}, None)


def test_step_outputs_round_trip(conn, step):
    repo.set_step_outputs(conn, step, {"video": ("video", SHA1), "c": ("content", "content:3")})
    rows = queries.step_outputs(conn, step)
    assert set(rows) == {"video", "c"}
    assert rows["video"]["type"] == "video" and rows["video"]["ref"] == SHA1
    assert rows["c"]["port"] == "c"
    assert queries.step_outputs(conn, step + 99) == {}


def test_bad_type_rejected(conn, step):
    with pytest.raises(sqlite3.IntegrityError):
        repo.set_step_outputs(conn, step, {"x": ("audio", SHA1)})
    with pytest.raises(sqlite3.IntegrityError):
        repo.put_cache(conn, KEY, "n", 1, step, {"x": ("audio", SHA1)})


def test_cache_round_trip_and_replace(conn, step):
    assert queries.cache_get(conn, KEY) is None
    repo.put_cache(conn, KEY, "n", 1, step, {"a": ("video", SHA1), "b": ("text", SHA2)})
    assert queries.cache_get(conn, KEY) == {"a": ("video", SHA1), "b": ("text", SHA2)}
    repo.put_cache(conn, KEY, "n", 1, step, {"a": ("video", SHA2), "b": ("text", SHA2)})
    assert queries.cache_get(conn, KEY)["a"] == ("video", SHA2)


def test_bad_cache_key_rejected(conn, step):
    with pytest.raises(sqlite3.IntegrityError):
        repo.put_cache(conn, "short", "n", 1, step, {"a": ("video", SHA1)})
