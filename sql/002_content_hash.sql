-- Idempotent ingestion.
--
-- backfill.py deduped in the application: it SELECTed every existing
-- (session_id, role, LEFT(content,200)) into a Python set and filtered against
-- it. That has two problems at history scale. It pulls the whole table into
-- memory on every run, and it is only advisory - two ingesters running at once,
-- or a crash midway, can still write duplicates because nothing in the database
-- forbids them.
--
-- A generated column plus a unique index moves the guarantee into Postgres, so
-- ON CONFLICT DO NOTHING makes re-running the backfill free and safe. Resuming
-- an interrupted run is then just running it again.
--
-- created_at is part of the key on purpose. The same sentence typed on two
-- different days is two real messages, and collapsing them would silently lose
-- the second - which matters most for the short ones ("yes", "do it", "ship it")
-- that carry the decision.
--
-- created_at is pinned to UTC inside the hash because extract(epoch from
-- timestamptz) reads the session TimeZone and is therefore not immutable, which
-- a generated column rejects outright. Without the pin this migration fails with
-- "generation expression is not immutable" - and had it been allowed, the same
-- row would hash differently for a client in Perth and a client in UTC.

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS content_hash TEXT
    GENERATED ALWAYS AS (
        encode(sha256(
            (session_id || '|' || role || '|' ||
             extract(epoch from (created_at AT TIME ZONE 'UTC'))::text ||
             '|' || content)::bytea
        ), 'hex')
    ) STORED;

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_content_hash
    ON messages (content_hash);
