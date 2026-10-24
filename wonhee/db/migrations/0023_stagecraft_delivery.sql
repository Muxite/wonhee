CREATE TABLE stagecraft_delivery (
    job_id TEXT PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES creative_project(id),
    request_key TEXT NOT NULL,
    ref TEXT NOT NULL REFERENCES blob(sha256),
    route_json TEXT NOT NULL,
    delivery_id INTEGER NOT NULL REFERENCES creative_delivery(id),
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
