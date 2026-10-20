-- GitHub slop machine (docs/specs/2026-10-20-github-slop-design.md): repos already picked, and its flag codes.
CREATE TABLE slop_seen (
  repo       TEXT PRIMARY KEY,
  first_seen TEXT NOT NULL,
  run_id     INTEGER
) STRICT;
INSERT INTO flag_code(code, meaning) VALUES
  ('SLOP_NO_GRAPH', 'a slop card set has a text slide 2: the export browser is missing or no imports resolved'),
  ('SLOP_RATE_LIMIT', 'the GitHub API refused with a rate limit; set GITHUB_TOKEN in the project env for more'),
  ('SLOP_NO_REPO', 'no unseen repo passed the filters after several searches'),
  ('SLOP_CLONE_FAILED', 'the shallow clone of a picked repo failed or timed out');
