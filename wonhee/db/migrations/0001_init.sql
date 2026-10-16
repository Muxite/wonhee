-- ===== media =====
CREATE TABLE blob (
  sha256    TEXT PRIMARY KEY CHECK (length(sha256) = 64),
  bytes     INTEGER NOT NULL CHECK (bytes >= 0),
  mime      TEXT NOT NULL,
  stored_at TEXT NOT NULL
) STRICT;

CREATE TABLE media (
  blob_sha256 TEXT PRIMARY KEY REFERENCES blob(sha256),
  kind        TEXT NOT NULL CHECK (kind IN ('video','audio','image','transcript')),
  duration_ms INTEGER,
  width       INTEGER,
  height      INTEGER,
  fps         REAL
) STRICT;

CREATE TABLE media_phash (
  blob_sha256 TEXT NOT NULL REFERENCES media(blob_sha256),
  frame_ms    INTEGER NOT NULL CHECK (frame_ms >= 0),
  phash       TEXT NOT NULL CHECK (length(phash) = 16),
  PRIMARY KEY (blob_sha256, frame_ms)
) STRICT;

-- ===== world =====
CREATE TABLE account (
  id          INTEGER PRIMARY KEY,
  platform    TEXT NOT NULL CHECK (platform IN ('instagram','tiktok','youtube','x')),
  external_id TEXT NOT NULL,
  handle      TEXT NOT NULL,
  relation    TEXT NOT NULL DEFAULT 'external' CHECK (relation IN ('ours','friend','external')),
  UNIQUE (platform, external_id),
  UNIQUE (id, platform)
) STRICT;

CREATE TABLE audio_track (
  id          INTEGER PRIMARY KEY,
  platform    TEXT NOT NULL,
  external_id TEXT NOT NULL,
  title       TEXT,
  artist      TEXT,
  UNIQUE (platform, external_id)
) STRICT;

CREATE TABLE content_item (
  id           INTEGER PRIMARY KEY,
  platform     TEXT NOT NULL,
  external_id  TEXT NOT NULL,
  url          TEXT NOT NULL,
  account_id   INTEGER NOT NULL,
  posted_at    TEXT,
  caption      TEXT,
  audio_id     INTEGER REFERENCES audio_track(id),
  media_sha256 TEXT REFERENCES media(blob_sha256),
  UNIQUE (platform, external_id),
  FOREIGN KEY (account_id, platform) REFERENCES account(id, platform)
) STRICT;

CREATE TABLE hashtag (
  id  INTEGER PRIMARY KEY,
  tag TEXT NOT NULL UNIQUE CHECK (tag = lower(tag) AND substr(tag, 1, 1) <> '#' AND length(tag) > 0)
) STRICT;

CREATE TABLE content_hashtag (
  content_id INTEGER NOT NULL REFERENCES content_item(id),
  hashtag_id INTEGER NOT NULL REFERENCES hashtag(id),
  PRIMARY KEY (content_id, hashtag_id)
) STRICT;

CREATE TABLE metric_snapshot (
  content_id  INTEGER NOT NULL REFERENCES content_item(id),
  observed_at TEXT NOT NULL,
  plays       INTEGER CHECK (plays IS NULL OR plays >= 0),
  likes       INTEGER CHECK (likes IS NULL OR likes >= 0),
  comments    INTEGER CHECK (comments IS NULL OR comments >= 0),
  shares      INTEGER CHECK (shares IS NULL OR shares >= 0),
  step_id     INTEGER NOT NULL REFERENCES step(id),
  PRIMARY KEY (content_id, observed_at)
) STRICT;

CREATE TABLE web_source (
  id              INTEGER PRIMARY KEY,
  url             TEXT NOT NULL UNIQUE,
  title           TEXT,
  retrieved_at    TEXT NOT NULL,
  snapshot_sha256 TEXT REFERENCES blob(sha256)
) STRICT;

-- ===== trends =====
CREATE TABLE trend (
  id              INTEGER PRIMARY KEY,
  name            TEXT NOT NULL,
  summary         TEXT,
  window_start    TEXT,
  window_end      TEXT,
  created_by_step INTEGER NOT NULL REFERENCES step(id)
) STRICT;

CREATE TABLE trend_beat (
  trend_id    INTEGER NOT NULL REFERENCES trend(id),
  ord         INTEGER NOT NULL,
  role        TEXT NOT NULL CHECK (role IN ('hook','setup','turn','payoff','outro')),
  description TEXT NOT NULL,
  PRIMARY KEY (trend_id, ord)
) STRICT;

CREATE TABLE trend_rule (
  trend_id INTEGER NOT NULL REFERENCES trend(id),
  ord      INTEGER NOT NULL,
  kind     TEXT NOT NULL CHECK (kind IN ('invariant','variation')),
  text     TEXT NOT NULL,
  PRIMARY KEY (trend_id, ord)
) STRICT;

CREATE TABLE trend_example (
  trend_id   INTEGER NOT NULL REFERENCES trend(id),
  content_id INTEGER NOT NULL REFERENCES content_item(id),
  PRIMARY KEY (trend_id, content_id)
) STRICT;

CREATE TABLE trend_source (
  trend_id      INTEGER NOT NULL REFERENCES trend(id),
  web_source_id INTEGER NOT NULL REFERENCES web_source(id),
  PRIMARY KEY (trend_id, web_source_id)
) STRICT;

-- ===== our work =====
CREATE TABLE intent (
  id             INTEGER PRIMARY KEY,
  raw_text       TEXT NOT NULL,
  schema_name    TEXT NOT NULL,
  schema_version INTEGER NOT NULL,
  compiled_json  TEXT NOT NULL CHECK (json_valid(compiled_json)),
  created_at     TEXT NOT NULL,
  confirmed_at   TEXT
) STRICT;

CREATE TABLE intent_turn (
  intent_id INTEGER NOT NULL REFERENCES intent(id),
  ord       INTEGER NOT NULL,
  role      TEXT NOT NULL CHECK (role IN ('user','compiler')),
  text      TEXT NOT NULL,
  gaps_json TEXT CHECK (gaps_json IS NULL OR json_valid(gaps_json)),
  PRIMARY KEY (intent_id, ord)
) STRICT;

CREATE TABLE project (
  id         INTEGER PRIMARY KEY,
  title      TEXT NOT NULL,
  kind       TEXT NOT NULL CHECK (kind IN ('scout','skit','edit')),
  status     TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','filming','editing','posted','dropped')),
  intent_id  INTEGER REFERENCES intent(id),
  created_at TEXT NOT NULL
) STRICT;

CREATE TABLE project_trend (
  project_id INTEGER NOT NULL REFERENCES project(id),
  trend_id   INTEGER NOT NULL REFERENCES trend(id),
  role       TEXT NOT NULL CHECK (role IN ('primary','secondary')),
  PRIMARY KEY (project_id, trend_id)
) STRICT;
CREATE UNIQUE INDEX one_primary_trend ON project_trend(project_id) WHERE role = 'primary';

CREATE TABLE project_reference (
  project_id INTEGER NOT NULL REFERENCES project(id),
  content_id INTEGER NOT NULL REFERENCES content_item(id),
  role       TEXT NOT NULL CHECK (role IN ('inspiration','counterexample','style_ref')),
  PRIMARY KEY (project_id, content_id)
) STRICT;

CREATE TABLE project_output (
  project_id INTEGER NOT NULL REFERENCES project(id),
  content_id INTEGER NOT NULL REFERENCES content_item(id),
  PRIMARY KEY (project_id, content_id)
) STRICT;

CREATE TABLE asset (
  id              INTEGER PRIMARY KEY,
  project_id      INTEGER NOT NULL REFERENCES project(id),
  media_sha256    TEXT NOT NULL REFERENCES media(blob_sha256),
  role            TEXT NOT NULL CHECK (role IN ('raw_take','edit','prop_capture','thumbnail')),
  created_by_step INTEGER REFERENCES step(id)
) STRICT;

CREATE TABLE shotlist (
  id              INTEGER PRIMARY KEY,
  project_id      INTEGER NOT NULL REFERENCES project(id),
  version         INTEGER NOT NULL CHECK (version >= 1),
  created_by_step INTEGER NOT NULL REFERENCES step(id),
  UNIQUE (project_id, version)
) STRICT;

CREATE TABLE shot (
  shotlist_id    INTEGER NOT NULL REFERENCES shotlist(id),
  ord            INTEGER NOT NULL CHECK (ord >= 1),
  duration_ms    INTEGER NOT NULL CHECK (duration_ms > 0),
  framing        TEXT NOT NULL,
  action         TEXT NOT NULL,
  on_screen_text TEXT,
  audio_cue      TEXT,
  PRIMARY KEY (shotlist_id, ord)
) STRICT;

CREATE TABLE shot_citation (
  id            INTEGER PRIMARY KEY,
  shotlist_id   INTEGER NOT NULL,
  shot_ord      INTEGER NOT NULL,
  content_id    INTEGER REFERENCES content_item(id),
  web_source_id INTEGER REFERENCES web_source(id),
  FOREIGN KEY (shotlist_id, shot_ord) REFERENCES shot(shotlist_id, ord),
  CHECK ((content_id IS NULL) <> (web_source_id IS NULL))
) STRICT;
CREATE UNIQUE INDEX shot_citation_content ON shot_citation(shotlist_id, shot_ord, content_id) WHERE content_id IS NOT NULL;
CREATE UNIQUE INDEX shot_citation_source ON shot_citation(shotlist_id, shot_ord, web_source_id) WHERE web_source_id IS NOT NULL;

-- ===== provenance =====
CREATE TABLE run (
  id           INTEGER PRIMARY KEY,
  project_id   INTEGER NOT NULL REFERENCES project(id),
  orchestrator TEXT NOT NULL CHECK (orchestrator IN ('fixed','agent')),
  status       TEXT NOT NULL CHECK (status IN ('running','done','weak','failed','stopped')),
  cost_cap_usd REAL NOT NULL CHECK (cost_cap_usd >= 0),
  started_at   TEXT NOT NULL,
  ended_at     TEXT
) STRICT;

CREATE TABLE step (
  id              INTEGER PRIMARY KEY,
  run_id          INTEGER NOT NULL REFERENCES run(id),
  stage           TEXT NOT NULL,
  attempt         INTEGER NOT NULL CHECK (attempt >= 1),
  params_json     TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(params_json)),
  status          TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','ok','weak','failed')),
  supersedes_step INTEGER REFERENCES step(id),
  started_at      TEXT NOT NULL,
  ended_at        TEXT,
  cost_usd        REAL NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
  UNIQUE (run_id, stage, attempt)
) STRICT;
CREATE INDEX step_supersedes ON step(supersedes_step);

CREATE TABLE step_input (
  step_id       INTEGER NOT NULL REFERENCES step(id),
  input_step_id INTEGER NOT NULL REFERENCES step(id),
  PRIMARY KEY (step_id, input_step_id)
) STRICT;

CREATE TABLE flag_code (
  code    TEXT PRIMARY KEY,
  meaning TEXT NOT NULL
) STRICT;

CREATE TABLE step_flag (
  step_id INTEGER NOT NULL REFERENCES step(id),
  code    TEXT NOT NULL REFERENCES flag_code(code),
  detail  TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (step_id, code)
) STRICT;

CREATE TABLE step_artifact (
  step_id     INTEGER NOT NULL REFERENCES step(id),
  blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
  role        TEXT NOT NULL CHECK (role IN ('api_response','llm_prompt','llm_reply','frame','error','search_result')),
  PRIMARY KEY (step_id, blob_sha256, role)
) STRICT;

CREATE TABLE llm_call (
  id          INTEGER PRIMARY KEY,
  step_id     INTEGER REFERENCES step(id),
  intent_id   INTEGER REFERENCES intent(id),
  model       TEXT NOT NULL,
  schema_name TEXT NOT NULL,
  tokens_in   INTEGER NOT NULL,
  tokens_out  INTEGER NOT NULL,
  cost_usd    REAL NOT NULL,
  valid       INTEGER NOT NULL CHECK (valid IN (0,1)),
  created_at  TEXT NOT NULL,
  CHECK ((step_id IS NULL) <> (intent_id IS NULL))
) STRICT;

-- ===== analysis (spec adjustment 2) =====
CREATE TABLE media_analysis (
  media_sha256      TEXT NOT NULL REFERENCES media(blob_sha256),
  analyzer_version  INTEGER NOT NULL,
  step_id           INTEGER NOT NULL REFERENCES step(id),
  first_word_ms     INTEGER,
  language          TEXT,
  transcript_sha256 TEXT REFERENCES blob(sha256),
  PRIMARY KEY (media_sha256, analyzer_version)
) STRICT;

CREATE TABLE media_shot (
  media_sha256     TEXT NOT NULL,
  analyzer_version INTEGER NOT NULL,
  ord              INTEGER NOT NULL CHECK (ord >= 1),
  start_ms         INTEGER NOT NULL CHECK (start_ms >= 0),
  end_ms           INTEGER NOT NULL,
  keyframe_sha256  TEXT REFERENCES media(blob_sha256),
  on_screen_text   TEXT,
  visual           TEXT,
  PRIMARY KEY (media_sha256, analyzer_version, ord),
  FOREIGN KEY (media_sha256, analyzer_version) REFERENCES media_analysis(media_sha256, analyzer_version),
  CHECK (end_ms > start_ms)
) STRICT;

INSERT INTO flag_code(code, meaning) VALUES
  ('LOW_SAMPLE',      'fewer examples than wanted; conclusions are thin'),
  ('WINDOW_WIDENED',  'time window was widened to find enough examples'),
  ('SCHEMA_RETRY',    'model reply failed validation once and was retried'),
  ('SCHEMA_FAIL',     'model reply failed validation twice'),
  ('UNCITED_CLAIM',   'an output item has no supporting source'),
  ('COST_CAP',        'run stopped at its cost cap'),
  ('NO_CANDIDATES',   'no trend candidates or reference content found'),
  ('DOWNLOAD_FAILED', 'one or more videos could not be downloaded'),
  ('LENGTH_OVER',     'shot list is longer than the requested max length'),
  ('EXCEPTION',       'unexpected error; traceback saved as an error artifact'),
  ('MISSING_KEY',     'a required API key is not configured');
