-- Correct the new external-output receipt to match the store's STRICT-table invariant.
-- 0023 can already be applied by live open_db callers, so preserve it and rebuild additively.
ALTER TABLE stagecraft_delivery RENAME TO stagecraft_delivery_legacy;
CREATE TABLE stagecraft_delivery (
    job_id TEXT PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES creative_project(id),
    request_key TEXT NOT NULL,
    ref TEXT NOT NULL REFERENCES blob(sha256),
    route_json TEXT NOT NULL,
    delivery_id INTEGER NOT NULL REFERENCES creative_delivery(id),
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;
INSERT INTO stagecraft_delivery SELECT * FROM stagecraft_delivery_legacy;
DROP TABLE stagecraft_delivery_legacy;
