-- Post projects: a creative project with a format and named slots that fill in any order, so a post can sit half-filled.
CREATE TABLE post_meta (
 project_id INTEGER PRIMARY KEY REFERENCES creative_project(id), format TEXT NOT NULL, goal TEXT NOT NULL
) STRICT;
CREATE TABLE post_slot (
 project_id INTEGER NOT NULL REFERENCES creative_project(id), name TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'empty' CHECK(state IN ('empty','have','rejected')),
 ref TEXT, text TEXT, source TEXT, reason TEXT, updated_at TEXT NOT NULL,
 PRIMARY KEY(project_id,name)
) STRICT;
