-- Index-time chunks for long messages.
--
-- A long assistant wrap-up states its facts thousands of characters in. One
-- embedding of the whole message is dominated by its opening, and the judge
-- reads only its head, so a fact at char 4000 is invisible to both. Messages
-- over 1500 chars are split into overlapping windows here, each embedded and
-- full-text indexed on its own; /search (ECHOES_CHUNK_SEARCH=1) ranks chunks
-- and collapses them back to the parent message.
--
-- A separate table on purpose: `messages` and its indexes are untouched, and
-- the whole feature is removed with DROP TABLE message_chunks plus the flags off.
--
-- Run with: docker exec echoes-postgres psql -U echoes -d echoes -f /sql/005_chunks.sql
CREATE TABLE IF NOT EXISTS message_chunks (
    id          BIGSERIAL PRIMARY KEY,
    message_id  BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    ord         INT    NOT NULL,
    start_char  INT    NOT NULL,
    content     TEXT   NOT NULL,
    embedding   vector(768),
    UNIQUE (message_id, ord)
);

CREATE INDEX IF NOT EXISTS idx_chunks_message
    ON message_chunks (message_id);

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON message_chunks USING gin (to_tsvector('english', content));

-- Same opclass and default params as idx_messages_embedding, so the two arms
-- of the vector search are searched the same way.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON message_chunks USING hnsw (embedding vector_cosine_ops);
