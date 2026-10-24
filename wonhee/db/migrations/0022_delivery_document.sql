-- The same original bytes may be sent as a document instead of a Telegram playable preview.
ALTER TABLE creative_delivery ADD COLUMN as_document INTEGER NOT NULL DEFAULT 0 CHECK(as_document IN (0,1));
