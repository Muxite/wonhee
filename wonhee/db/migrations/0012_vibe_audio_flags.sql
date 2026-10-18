INSERT INTO flag_code(code, meaning) VALUES
  ('DENOISE_FALLBACK', 'the RNNoise model could not be fetched, so audio was denoised with afftdn instead'),
  ('REVIEW_CLAMPED', 'a vision reviewer chose an edit knob outside its range, so the value was clamped');
