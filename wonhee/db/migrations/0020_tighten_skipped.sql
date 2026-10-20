-- compose_video tighten (docs/specs/2026-10-20-reel-styles-design.md §3): a take with nothing to find pauses in.
INSERT INTO flag_code(code, meaning) VALUES
  ('TIGHTEN_SKIPPED', 'tighten left the take untrimmed: it has no word timings and no audio stream to detect pauses in');
