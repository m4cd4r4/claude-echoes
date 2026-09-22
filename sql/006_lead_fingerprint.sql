-- Leading-text fingerprints of user turns, for ECHOES_AUTOMATION_DEMOTE.
--
-- A user-role row whose opening text is shared by many sessions is a scripted
-- prompt (a headless job, an injected IDE notice, a pasted template), not
-- something the person asked. Finding those prints at query time means reading
-- the head of every user row, and the heads of long rows sit in TOAST: measured
-- 2026-09-22 at ~15 s over ~60k user rows. Stored once per row, with an index,
-- the grouping is an index scan.
--
-- A separate table on purpose, like 005: a column on `messages` has to be
-- backfilled with an UPDATE, and every updated row re-enters every index on
-- the table, the HNSW graph included. Measured 2026-09-22: that backfill was
-- still running after 10 minutes and was cancelled. Rows here are small and
-- the table is dropped with DROP TABLE message_fp plus the flag off.
--
-- The expression mirrors server/intent.py lead_fingerprint(): first 160 chars,
-- digits masked, whitespace collapsed, trimmed, lowercased, first 120 chars.
-- Keep the two in step.
--
-- Optional: without this file /search computes the prints inline.
-- Run with: docker exec -i echoes-postgres psql -U echoes -d echoes < sql/006_lead_fingerprint.sql
CREATE OR REPLACE FUNCTION echoes_lead_fp(t text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT lower(left(btrim(regexp_replace(regexp_replace(
               left(t, 160), '[0-9]', '#', 'g'), '[[:space:]]+', ' ', 'g')), 120))
$$;

CREATE TABLE IF NOT EXISTS message_fp (
    message_id  BIGINT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    session_id  VARCHAR(255) NOT NULL,
    fp          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_message_fp ON message_fp (fp, session_id);

-- Kept on write, so every ingest path (the hook, the backfill scripts) agrees.
CREATE OR REPLACE FUNCTION echoes_add_message_fp() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.role = 'user' THEN
        INSERT INTO message_fp (message_id, session_id, fp)
        VALUES (NEW.id, NEW.session_id, echoes_lead_fp(NEW.content))
        ON CONFLICT (message_id) DO NOTHING;
    END IF;
    RETURN NULL;
END $$;

DROP TRIGGER IF EXISTS trg_messages_fp ON messages;
CREATE TRIGGER trg_messages_fp
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION echoes_add_message_fp();

INSERT INTO message_fp (message_id, session_id, fp)
SELECT id, session_id, echoes_lead_fp(content)
  FROM messages WHERE role = 'user'
ON CONFLICT (message_id) DO NOTHING;

ANALYZE message_fp;
