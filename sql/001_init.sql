-- claude-echoes schema
-- Run with: docker exec echoes-postgres psql -U echoes -d echoes -f /sql/001_init.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS messages (
    id           BIGSERIAL PRIMARY KEY,
    session_id   VARCHAR(255) NOT NULL,
    project      VARCHAR(255) NOT NULL DEFAULT 'unknown',
    machine      VARCHAR(255),
    role         VARCHAR(20)  NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT         NOT NULL,
    model        VARCHAR(100),
    embedding    vector(768),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_project
    ON messages (project, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_messages_fts
    ON messages USING gin (to_tsvector('english', content));

-- HNSW index for cosine distance semantic search.
-- Only created if not already present so migrations are safe to re-run.
CREATE INDEX IF NOT EXISTS idx_messages_embedding
    ON messages USING hnsw (embedding vector_cosine_ops);
