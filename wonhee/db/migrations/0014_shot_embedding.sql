-- shot_index: one SigLIP2 vector per shot keyframe, so shots can be searched by description across all footage.
-- vec is float16 little-endian, L2-normalized (a dot product is the cosine). Rows are keyed by the model, so a new
-- model indexes beside the old one instead of mixing vector spaces.
CREATE TABLE shot_embedding (
  media_sha256    TEXT NOT NULL,
  model           TEXT NOT NULL,
  ord             INTEGER NOT NULL,
  start_ms        INTEGER NOT NULL,
  end_ms          INTEGER NOT NULL,
  keyframe_sha256 TEXT NOT NULL,
  tags            TEXT NOT NULL DEFAULT '[]',  -- JSON [[tag, score], ...], best first
  vec             BLOB NOT NULL,
  PRIMARY KEY (media_sha256, model, ord)
) STRICT;
CREATE INDEX shot_embedding_model ON shot_embedding(model);
