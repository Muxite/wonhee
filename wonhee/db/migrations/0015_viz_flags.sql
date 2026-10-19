-- viz render nodes (moved in from vizkit) and publish_muksite.
INSERT INTO flag_code(code, meaning) VALUES
  ('VIZ_SOURCE', 'what a viz render looked at: path, resolved commit and file counts, so the picture can be reproduced'),
  ('VIZ_EMPTY', 'the viz scan found no files to draw (wrong path, or include/exclude/focus matched nothing)'),
  ('VIZ_NO_EDGES', 'the dependency graph resolved no imports, so it is a cloud of unconnected files'),
  ('EXPORT_UNAVAILABLE', 'a browser render could not run: Playwright or its Chromium is missing'),
  ('WAVESIM_FAILED', 'the WaveSim docker render failed or produced no image'),
  ('PUBLISHED', 'an image was uploaded to a public site; the detail holds the public URL'),
  ('PUBLISH_FAILED', 'an upload to a public site was refused or failed');
