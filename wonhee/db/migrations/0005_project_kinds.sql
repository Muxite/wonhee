-- wonhee:foreign-keys-off
-- Rebuild project to widen kind: 'library' (grabbed reels: ours, friends', anyone's) and 'style' (style rips).
-- SQLite cannot alter a CHECK in place; this is the documented create-copy-drop-rename procedure.
PRAGMA legacy_alter_table = ON;
CREATE TABLE project_new (
  id         INTEGER PRIMARY KEY,
  title      TEXT NOT NULL,
  kind       TEXT NOT NULL CHECK (kind IN ('scout','skit','edit','library','style')),
  status     TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','filming','editing','posted','dropped')),
  intent_id  INTEGER REFERENCES intent(id),
  created_at TEXT NOT NULL
) STRICT;
INSERT INTO project_new(id, title, kind, status, intent_id, created_at)
  SELECT id, title, kind, status, intent_id, created_at FROM project;
DROP TABLE project;
ALTER TABLE project_new RENAME TO project;
PRAGMA legacy_alter_table = OFF;
