-- compose_video (docs/specs/2026-10-20-reel-styles-design.md §3): pause trimming reports what it cut.
INSERT INTO flag_code(code, meaning) VALUES
  ('PAUSES_TRIMMED', 'compose_video cut pauses out of the take (tighten); the detail holds the seconds removed and the new length');
