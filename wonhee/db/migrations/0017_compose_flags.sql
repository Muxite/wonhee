-- compose / frame_video / grid_plan (docs/specs/2026-10-20-meme-and-popart-formats-design.md): their flag codes.
INSERT INTO flag_code(code, meaning) VALUES
  ('LAYOUT_ADJUSTED', 'a text or sticker layer sat in a reel UI safe zone (caption block or button strip) and was moved inside'),
  ('CAPTIONS_ESTIMATED_TIMING', 'subtitles have line timing only, so each line''s time was split evenly across its words'),
  ('GRID_PLANNED', 'grid_plan assigned a template, palette and pattern to each tile; the detail holds the plan');
