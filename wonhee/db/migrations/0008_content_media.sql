-- Every file a post carries, in order: one video for a reel, N images (or mixed) for a carousel.
-- content_item.media_sha256 stays "the main video" so video analysis and the scout path are unchanged.
CREATE TABLE content_media (
  content_id   INTEGER NOT NULL REFERENCES content_item(id),
  ord          INTEGER NOT NULL CHECK (ord >= 0),
  media_sha256 TEXT NOT NULL REFERENCES media(blob_sha256),
  kind         TEXT NOT NULL CHECK (kind IN ('video', 'image')),
  PRIMARY KEY (content_id, ord)
) STRICT;

-- What is on a still image (on-screen text, short visual description); keyed by image, so it caches across posts.
CREATE TABLE image_analysis (
  media_sha256     TEXT NOT NULL REFERENCES media(blob_sha256),
  analyzer_version INTEGER NOT NULL,
  step_id          INTEGER NOT NULL REFERENCES step(id),
  on_screen_text   TEXT,
  visual           TEXT,
  PRIMARY KEY (media_sha256, analyzer_version)
) STRICT;

INSERT INTO flag_code(code, meaning) VALUES ('IMAGE_FAILED', 'one image of a photo or carousel post could not be downloaded');
