-- wonhee:foreign-keys-off
-- Rebuild project to add kind 'chain'; add node output + result cache tables.
PRAGMA legacy_alter_table = ON;
CREATE TABLE project_new (
  id         INTEGER PRIMARY KEY,
  title      TEXT NOT NULL,
  kind       TEXT NOT NULL CHECK (kind IN ('scout','skit','edit','library','style','chain')),
  status     TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','filming','editing','posted','dropped')),
  intent_id  INTEGER REFERENCES intent(id),
  created_at TEXT NOT NULL
) STRICT;
INSERT INTO project_new(id, title, kind, status, intent_id, created_at)
  SELECT id, title, kind, status, intent_id, created_at FROM project;
DROP TABLE project;
ALTER TABLE project_new RENAME TO project;
PRAGMA legacy_alter_table = OFF;

CREATE TABLE step_output (
  step_id INTEGER NOT NULL REFERENCES step(id), port TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('content','video','image','subtitles','text','file')),
  ref TEXT NOT NULL, PRIMARY KEY (step_id, port)) STRICT;
CREATE TABLE node_cache (
  key_sha256 TEXT NOT NULL CHECK (length(key_sha256)=64), port TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('content','video','image','subtitles','text','file')),
  ref TEXT NOT NULL, node TEXT NOT NULL, node_version INTEGER NOT NULL,
  created_by_step INTEGER NOT NULL REFERENCES step(id), PRIMARY KEY (key_sha256, port)) STRICT;
INSERT INTO flag_code(code, meaning) VALUES
  ('CACHED', 'result reused from the node cache; no work was done'),
  ('TRANSLATION_GAP', 'some lines came back untranslated and kept their original text'),
  ('NO_TEXT', 'nothing to work on (no speech or no on-screen text); output is empty');
