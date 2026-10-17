-- wonhee:foreign-keys-off
-- Add the 'images' port type (an ordered image set / carousel manifest): rebuild the two tables whose CHECK lists types.
PRAGMA legacy_alter_table = ON;
CREATE TABLE step_output_new (
  step_id INTEGER NOT NULL REFERENCES step(id), port TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('content','video','image','images','subtitles','text','file')),
  ref TEXT NOT NULL, PRIMARY KEY (step_id, port)) STRICT;
INSERT INTO step_output_new SELECT step_id, port, type, ref FROM step_output;
DROP TABLE step_output;
ALTER TABLE step_output_new RENAME TO step_output;
CREATE TABLE node_cache_new (
  key_sha256 TEXT NOT NULL CHECK (length(key_sha256)=64), port TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('content','video','image','images','subtitles','text','file')),
  ref TEXT NOT NULL, node TEXT NOT NULL, node_version INTEGER NOT NULL,
  created_by_step INTEGER NOT NULL REFERENCES step(id), PRIMARY KEY (key_sha256, port)) STRICT;
INSERT INTO node_cache_new SELECT key_sha256, port, type, ref, node, node_version, created_by_step FROM node_cache;
DROP TABLE node_cache;
ALTER TABLE node_cache_new RENAME TO node_cache;
PRAGMA legacy_alter_table = OFF;
INSERT INTO flag_code(code, meaning) VALUES ('TOO_MANY_SLIDES', 'the carousel has more slides than Instagram accepts in one post');
