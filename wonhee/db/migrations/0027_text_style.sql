-- The on-photo text style of a reference image (colour pair, relative size, alignment, box, dimming, block position),
-- as a TextSpec-shaped JSON profile. Keyed by image, like image_analysis, with its own analyzer version.
CREATE TABLE text_style (
  media_sha256     TEXT NOT NULL REFERENCES media(blob_sha256),
  analyzer_version INTEGER NOT NULL,
  profile_json     TEXT NOT NULL,
  PRIMARY KEY (media_sha256, analyzer_version)
) STRICT;

INSERT INTO flag_code(code, meaning) VALUES
  ('NO_TEXT_FOUND', 'no text block found on the image; the style profile is empty'),
  ('TEXT_STYLE_UNSURE', 'text block found but a measurement is a weak guess; look at the profile');
