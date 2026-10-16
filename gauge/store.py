"""SQLite ledger: run, prediction, sample, summary, evaluation."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
  id INTEGER PRIMARY KEY,
  workload TEXT NOT NULL,
  family TEXT NOT NULL,
  spec_hash TEXT NOT NULL,
  config_hash TEXT NOT NULL,          -- params minus the size axis: what history groups by
  spec_json TEXT NOT NULL,
  features_json TEXT NOT NULL,        -- known before launch only
  size REAL,                          -- the workload's driving size variable
  image TEXT, image_digest TEXT,
  execution_mode TEXT NOT NULL,
  warmness TEXT NOT NULL,
  corun_group TEXT,                   -- set when launched alongside other runs on purpose
  provenance_json TEXT NOT NULL DEFAULT '{}',
  host_json TEXT,                     -- hardware/runtime fingerprint snapshot
  initial_json TEXT,                  -- free VRAM, other GPU pids, RAM, swap, temp at start
  container_id TEXT,
  prediction_committed REAL,
  container_start_requested REAL,
  container_started REAL,
  workload_finished REAL,
  container_finished REAL,
  marks_json TEXT NOT NULL DEFAULT '[]',
  outcome_json TEXT,                  -- workload-reported facts (token counts etc.): outcomes, never features
  exit_code INTEGER,
  termination_reason TEXT,
  docker_oom_killed INTEGER,
  failure_class TEXT,
  failure_source TEXT,
  status TEXT NOT NULL DEFAULT 'pending'  -- pending|running|done|error
);
CREATE TABLE IF NOT EXISTS prediction (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES run(id),
  created_at REAL NOT NULL,
  predictor TEXT NOT NULL,
  predictor_version INTEGER NOT NULL,
  status TEXT NOT NULL,
  runtime_estimate_s REAL, runtime_upper_s REAL,
  cpu_time_estimate_s REAL,
  ram_peak_upper_bytes REAL, vram_peak_upper_bytes REAL,
  evidence_json TEXT NOT NULL,
  history_cutoff REAL,
  note TEXT
);
CREATE TRIGGER IF NOT EXISTS prediction_immutable BEFORE UPDATE ON prediction
BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;
CREATE TABLE IF NOT EXISTS sample (
  run_id INTEGER NOT NULL REFERENCES run(id),
  t REAL NOT NULL,                    -- seconds since container_started
  cpu_usage_usec INTEGER, mem_current INTEGER, mem_anon INTEGER, mem_file INTEGER, swap_current INTEGER,
  io_rbytes INTEGER, io_wbytes INTEGER,
  vram_proc_bytes INTEGER,            -- scope process_sum: NVML process memory mapped to this cgroup
  dev_util REAL, dev_mem_util REAL, dev_mem_used INTEGER, dev_power_w REAL, dev_sm_mhz REAL,
  dev_temp_c REAL, dev_throttle INTEGER  -- scope device
);
CREATE INDEX IF NOT EXISTS sample_run ON sample(run_id, t);
CREATE TABLE IF NOT EXISTS summary (
  run_id INTEGER PRIMARY KEY REFERENCES run(id),
  json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluation (
  prediction_id INTEGER NOT NULL REFERENCES prediction(id),
  run_id INTEGER NOT NULL,
  target TEXT NOT NULL,
  actual REAL, estimate REAL, upper REAL,
  signed_error REAL, abs_error REAL, ratio REAL,
  underprediction REAL, excess_reservation REAL, covered INTEGER,
  censored INTEGER NOT NULL,
  PRIMARY KEY (prediction_id, target)
);
"""

SAMPLE_COLS = ("t", "cpu_usage_usec", "mem_current", "mem_anon", "mem_file", "swap_current", "io_rbytes",
               "io_wbytes", "vram_proc_bytes", "dev_util", "dev_mem_util", "dev_mem_used", "dev_power_w",
               "dev_sm_mhz", "dev_temp_c", "dev_throttle")

DEFAULT_DB = Path("data/gauge.sqlite")


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def insert_run(conn, **fields) -> int:
    fields = {k: (json.dumps(v) if k.endswith("_json") and not isinstance(v, str) else v) for k, v in fields.items()}
    cols = ",".join(fields)
    cur = conn.execute(f"INSERT INTO run ({cols}) VALUES ({','.join('?' * len(fields))})", tuple(fields.values()))
    return cur.lastrowid


def update_run(conn, run_id: int, **fields) -> None:
    fields = {k: (json.dumps(v) if k.endswith("_json") and not isinstance(v, str) else v) for k, v in fields.items()}
    sets = ",".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE run SET {sets} WHERE id=?", (*fields.values(), run_id))


def insert_prediction(conn, run_id: int, pred) -> int:
    cur = conn.execute(
        "INSERT INTO prediction (run_id, created_at, predictor, predictor_version, status, runtime_estimate_s,"
        " runtime_upper_s, cpu_time_estimate_s, ram_peak_upper_bytes, vram_peak_upper_bytes, evidence_json,"
        " history_cutoff, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, time.time(), pred.predictor, pred.predictor_version, pred.status, pred.runtime_estimate_s,
         pred.runtime_upper_s, pred.cpu_time_estimate_s, pred.ram_peak_upper_bytes, pred.vram_peak_upper_bytes,
         json.dumps(pred.evidence_run_ids), pred.history_cutoff, pred.note))
    return cur.lastrowid


def insert_samples(conn, run_id: int, rows: list[dict]) -> None:
    if not rows:
        return
    conn.executemany(
        f"INSERT INTO sample (run_id,{','.join(SAMPLE_COLS)}) VALUES (?{',?' * len(SAMPLE_COLS)})",
        [(run_id, *(r.get(c) for c in SAMPLE_COLS)) for r in rows])


def put_summary(conn, run_id: int, summary: dict) -> None:
    conn.execute("INSERT OR REPLACE INTO summary (run_id, json) VALUES (?,?)", (run_id, json.dumps(summary)))


def summary(conn, run_id: int) -> dict | None:
    row = conn.execute("SELECT json FROM summary WHERE run_id=?", (run_id,)).fetchone()
    return json.loads(row["json"]) if row else None


def history(conn, family: str, config_hash: str, before: float, workload: str | None = None) -> list[dict]:
    """Finished isolated runs of one family+config, completed before `before`, with their summaries."""
    q = ("SELECT r.id, r.size, r.spec_json, r.failure_class, r.container_finished, s.json AS summary FROM run r"
         " JOIN summary s ON s.run_id = r.id WHERE r.family=? AND r.config_hash=? AND r.status='done'"
         " AND r.execution_mode='isolated' AND r.corun_group IS NULL AND r.container_finished < ?")
    args: list = [family, config_hash, before]
    if workload:
        q += " AND r.workload=?"
        args.append(workload)
    out = []
    for r in conn.execute(q + " ORDER BY r.id", args):
        d = dict(r)
        d["summary"] = json.loads(d["summary"])
        d["params"] = json.loads(d.pop("spec_json"))["params"]
        out.append(d)
    return out
