-- Warm workers: long-lived processes that keep a model loaded and run chain runs from a queue (wonhee/worker.py).
-- Times are unix seconds (REAL), so staleness is a plain comparison.
CREATE TABLE job (
  id           INTEGER PRIMARY KEY,
  run_id       INTEGER NOT NULL REFERENCES run(id),
  lane         TEXT NOT NULL,
  status       TEXT NOT NULL CHECK (status IN ('queued','running','done','failed')),
  worker       TEXT,
  attempts     INTEGER NOT NULL DEFAULT 0,
  created_at   REAL NOT NULL,
  claimed_at   REAL,
  heartbeat_at REAL,
  finished_at  REAL,
  error        TEXT
) STRICT;
CREATE INDEX job_queue ON job(lane, status, id);

CREATE TABLE worker (
  name         TEXT PRIMARY KEY,
  lane         TEXT NOT NULL,
  pid          INTEGER NOT NULL,
  models       TEXT NOT NULL DEFAULT '[]',  -- JSON list of preloaded models
  started_at   REAL NOT NULL,
  loaded_at    REAL,
  heartbeat_at REAL NOT NULL,
  jobs_done    INTEGER NOT NULL DEFAULT 0,
  current_job  INTEGER
) STRICT;

INSERT INTO flag_code(code, meaning) VALUES
  ('WORKER_LOST', 'the worker running this step stopped heartbeating, so the step was marked failed and its job requeued');
