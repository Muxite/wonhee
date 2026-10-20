-- alignment check 2026-10-20: the caption-timing flag takes the reel-styles name, and focus: subject gets a flag.
-- 0017 was already applied, so it is not edited: the old code is replaced here when nothing references it.
INSERT INTO flag_code(code, meaning) VALUES
  ('WORDS_ESTIMATED', 'caption word times were estimated: subtitles have line timing only, so each line''s time was split evenly across its words'),
  ('FOCUS_NO_SUBJECT', 'a compose photo layer asked for focus: subject but the mask found no subject, so it was cropped to the centre');
DELETE FROM flag_code WHERE code = 'CAPTIONS_ESTIMATED_TIMING'
  AND NOT EXISTS (SELECT 1 FROM step_flag WHERE code = 'CAPTIONS_ESTIMATED_TIMING');
