-- Gap 1: the verbatim chat index was only ONE of four memory layers, and the
-- other three were never searchable. Measured 2026-08-29: 108,570 chat messages
-- were indexed while 759 dated ledger entries - the layer that actually records
-- what SUPERSEDED what - were not. A /recall for the ERF hero decision returned
-- the conversation and missed the logged decision, because the logged decision
-- was not in the index at all.
--
-- `source` marks which layer a row came from, so results can be attributed and
-- filtered. Defaulting to 'chat' leaves all 108,570 existing rows correct
-- without a rewrite.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'chat';
CREATE INDEX IF NOT EXISTS idx_messages_source ON messages (source, created_at DESC);

-- A ledger entry is neither a user turn nor an assistant turn. Reusing either
-- label would corrupt the --role filter that /recall offers over real
-- conversation, so the constraint widens to admit 'note' instead.
ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_role_check;
ALTER TABLE messages ADD CONSTRAINT messages_role_check
  CHECK (role IN ('user', 'assistant', 'note'));
