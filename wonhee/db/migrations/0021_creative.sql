-- Creative projects span many execution projects; memory and transport receipts survive conversation compaction.
CREATE TABLE creative_project (
 id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'active',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
) STRICT;
CREATE TABLE creative_route (
 route_key TEXT PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES creative_project(id), updated_at TEXT NOT NULL
) STRICT;
CREATE TABLE creative_brief (
 id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES creative_project(id), revision INTEGER NOT NULL,
 content TEXT NOT NULL, source_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
 UNIQUE(project_id,revision)
) STRICT;
CREATE TABLE creative_intake (
 id INTEGER PRIMARY KEY, project_id INTEGER REFERENCES creative_project(id), source_key TEXT NOT NULL UNIQUE,
 source TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', role TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('received','downloading','stored','failed')),
 ref TEXT REFERENCES blob(sha256), filename TEXT, mime TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
) STRICT;
CREATE TABLE creative_memory (
 id INTEGER PRIMARY KEY, project_id INTEGER REFERENCES creative_project(id), kind TEXT NOT NULL,
 content TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('confirmed','proposed','superseded','forgotten')),
 source_json TEXT NOT NULL DEFAULT '{}', supersedes INTEGER REFERENCES creative_memory(id),
 indexed INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
) STRICT;
CREATE VIRTUAL TABLE creative_memory_fts USING fts5(content, content='creative_memory', content_rowid='id');
CREATE TRIGGER creative_memory_insert AFTER INSERT ON creative_memory BEGIN
 INSERT INTO creative_memory_fts(rowid,content) VALUES(new.id,new.content);
END;
CREATE TRIGGER creative_memory_update AFTER UPDATE OF content ON creative_memory BEGIN
 INSERT INTO creative_memory_fts(creative_memory_fts,rowid,content) VALUES('delete',old.id,old.content);
 INSERT INTO creative_memory_fts(rowid,content) VALUES(new.id,new.content);
END;
CREATE TABLE creative_idea (
 id INTEGER PRIMARY KEY, project_id INTEGER REFERENCES creative_project(id), content TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','accepted','producing','done','rejected')),
 source_json TEXT NOT NULL DEFAULT '{}', memory_id INTEGER REFERENCES creative_memory(id),
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
) STRICT;
CREATE TABLE creative_submission (
 id INTEGER PRIMARY KEY, request_key TEXT NOT NULL UNIQUE, project_id INTEGER NOT NULL REFERENCES creative_project(id),
 asset_id INTEGER NOT NULL REFERENCES creative_intake(id), style TEXT NOT NULL, params_json TEXT NOT NULL,
 route_json TEXT NOT NULL DEFAULT '{}', brief_revision INTEGER NOT NULL,
 status TEXT NOT NULL DEFAULT 'queued' CHECK(status IN ('queued','running','done','failed')),
 phase TEXT NOT NULL DEFAULT 'speech', result_ref TEXT REFERENCES blob(sha256), error TEXT,
 heartbeat_at REAL, attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
) STRICT;
CREATE TABLE creative_run (
 submission_id INTEGER NOT NULL REFERENCES creative_submission(id), phase TEXT NOT NULL,
 run_id INTEGER NOT NULL REFERENCES run(id), PRIMARY KEY(submission_id,phase)
) STRICT;
CREATE TABLE creative_delivery (
 id INTEGER PRIMARY KEY, submission_id INTEGER REFERENCES creative_submission(id), project_id INTEGER REFERENCES creative_project(id),
 ref TEXT NOT NULL REFERENCES blob(sha256), filename TEXT NOT NULL, mime TEXT NOT NULL,
 token TEXT NOT NULL UNIQUE, route_json TEXT NOT NULL DEFAULT '{}', content TEXT NOT NULL DEFAULT '',
 status TEXT NOT NULL DEFAULT 'queued' CHECK(status IN ('queued','sending','sent','failed','uncertain')),
 message_ids_json TEXT NOT NULL DEFAULT '[]', error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(submission_id,ref)
) STRICT;
CREATE TABLE creative_maintenance (
 name TEXT PRIMARY KEY, error TEXT, updated_at TEXT NOT NULL
) STRICT;
CREATE INDEX creative_intake_project ON creative_intake(project_id,id);
CREATE INDEX creative_memory_scope ON creative_memory(project_id,status,kind,id);
CREATE INDEX creative_submission_queue ON creative_submission(status,id);
CREATE INDEX creative_delivery_queue ON creative_delivery(status,id);
