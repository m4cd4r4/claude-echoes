-- Two failed attempts are recorded here because both look correct and both are
-- rejected only at CREATE time, which is easy to rediscover.
--
-- 1. sha256((... )::bytea)
--    text::bytea performs ESCAPE PROCESSING, so any message containing a
--    literal \x or \001 sequence aborts the INSERT with
--        invalid input syntax for type bytea
--    Message content is arbitrary user text - code, regexes, Windows paths - so
--    this is a matter of when, not if. It surfaced on the first real batch.
--
-- 2. sha256(convert_to(..., 'UTF8'))
--    convert_to is the escape-free conversion, but it is STABLE rather than
--    IMMUTABLE (it reads the database encoding), and a generated column accepts
--    immutable expressions only:
--        generation expression is not immutable
--
-- md5(text) takes text directly, is immutable, and needs no extension. This
-- column exists to deduplicate an import, not to authenticate anything, so a
-- 128-bit digest is the right tool: at 108k rows the collision probability is
-- around 1e-28. If that ever stops being true, pgcrypto's digest(text,'sha256')
-- is immutable and is the drop-in replacement.
DROP INDEX IF EXISTS idx_messages_content_hash;
ALTER TABLE messages DROP COLUMN IF EXISTS content_hash;

ALTER TABLE messages
    ADD COLUMN content_hash TEXT
    GENERATED ALWAYS AS (
        md5(session_id || '|' || role || '|' ||
            extract(epoch from (created_at AT TIME ZONE 'UTC'))::text ||
            '|' || content)
    ) STORED;

CREATE UNIQUE INDEX idx_messages_content_hash ON messages (content_hash);
