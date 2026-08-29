"""
claude-echoes server

Minimal FastAPI service that:
  - accepts message writes from the Claude Code hook
  - embeds them via local Ollama (nomic-embed-text)
  - stores them in Postgres with pgvector
  - exposes a semantic search endpoint

Everything is inline in one file on purpose. If you can't read it end to end
in 5 minutes, something has gone wrong.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
import asyncpg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# --- config ---------------------------------------------------------------

DB_DSN       = os.environ.get("ECHOES_DB_DSN",
                              "postgresql://echoes:echoes@postgres:5432/echoes")
OLLAMA_URL   = os.environ.get("ECHOES_OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("ECHOES_OLLAMA_MODEL", "nomic-embed-text")
EMBED_TIMEOUT_S = float(os.environ.get("ECHOES_EMBED_TIMEOUT", "4"))
MAX_CONTENT_CHARS = 8000   # nomic context window is 8192 tokens; clip safely

# --- models ---------------------------------------------------------------

class MessageIn(BaseModel):
    session_id: str
    project: str = "unknown"
    machine: Optional[str] = None
    role: str                    # "user" | "assistant"
    content: str
    model: Optional[str] = None

class SearchHit(BaseModel):
    id: int
    session_id: str
    project: str
    role: str
    content: str
    model: Optional[str]
    created_at: str
    similarity: float

# --- app lifecycle --------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=5)
    app.state.http = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=EMBED_TIMEOUT_S)
    )
    yield
    await app.state.http.close()
    await app.state.pool.close()

app = FastAPI(title="claude-echoes", lifespan=lifespan)

# --- helpers --------------------------------------------------------------

async def embed_text(http: aiohttp.ClientSession, text: str) -> Optional[str]:
    """
    Get a 768-dim nomic embedding as a pgvector text literal.
    Returns None on failure so callers can still store the message.
    """
    try:
        async with http.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": (text or "")[:MAX_CONTENT_CHARS]},
        ) as r:
            if r.status != 200:
                return None
            data = await r.json()
            vec = data.get("embedding")
            if not vec:
                return None
            return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
    except Exception:
        return None

# --- endpoints ------------------------------------------------------------

@app.get("/health")
async def health():
    async with app.state.pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    return {"ok": True, "db": "up", "model": OLLAMA_MODEL}

@app.post("/message")
async def write_message(msg: MessageIn):
    if msg.role not in ("user", "assistant"):
        raise HTTPException(400, "role must be 'user' or 'assistant'")
    if not msg.content.strip():
        return {"skipped": True, "reason": "empty"}

    emb = await embed_text(app.state.http, msg.content)

    async with app.state.pool.acquire() as conn:
        if emb is not None:
            row = await conn.fetchrow(
                """
                INSERT INTO messages (session_id, project, machine, role, content, model, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
                RETURNING id, created_at
                """,
                msg.session_id, msg.project, msg.machine, msg.role,
                msg.content, msg.model, emb,
            )
        else:
            row = await conn.fetchrow(
                """
                INSERT INTO messages (session_id, project, machine, role, content, model)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, created_at
                """,
                msg.session_id, msg.project, msg.machine, msg.role,
                msg.content, msg.model,
            )

    return {
        "id": row["id"],
        "embedded": emb is not None,
        "created_at": row["created_at"].isoformat(),
    }

@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    project: Optional[str] = None,
    role: Optional[str] = None,
    days: Optional[int] = Query(None, ge=1, le=3650),
    hybrid: bool = Query(True),
    candidates: int = Query(60, ge=10, le=500),
):
    """Hybrid retrieval: dense vector + lexical BM25-ish, fused with RRF.

    This endpoint used to run pure cosine, while the benchmark harness that
    produced the README's headline number ran vector + BM25 + RRF. That gap
    mattered: the shipped /recall was materially weaker than the configuration
    the project is measured on, and nothing said so.

    The lexical half is Postgres full-text ranking, not true BM25. It is close
    enough in behaviour for the job it does here - catching the exact-token
    matches that embeddings miss, like an error string, a flag, or a commit sha -
    and it costs nothing extra, because sql/001_init.sql has always created
    idx_messages_fts on to_tsvector('english', content) and nothing has ever
    queried it. The expression below is written to match that index exactly, so
    it is used rather than rebuilt per query.

    RRF (Cormack et al.) fuses by RANK, not by score, which is the point: cosine
    similarity and ts_rank are not on comparable scales, so any weighted sum of
    the two raw scores is arbitrary. k=60 is the constant from the paper.

    hybrid=false restores the old pure-vector path, so the two are comparable on
    the same data without redeploying.
    """
    qvec = await embed_text(app.state.http, q)
    if qvec is None:
        raise HTTPException(503, "embedding service unavailable")

    if role and role not in ("user", "assistant"):
        raise HTTPException(400, "role must be 'user' or 'assistant'")

    # Filters are shared by both arms, so a project/role/days filter cannot
    # produce a hit from one arm that the other was never allowed to see.
    #
    # The placeholder numbering has to follow the MODE, not a fixed offset: the
    # vector-only SQL never mentions the query text, and asyncpg rejects a
    # parameter that appears in the argument list but not in the statement. A
    # fixed "start filters at $3" produced a 500 on every hybrid=false call
    # while hybrid=true worked, which is exactly the kind of break that hides
    # behind a default value.
    params: list = [qvec] if not hybrid else [qvec, q]
    idx = len(params) + 1

    conds = []
    if project:
        conds.append(f"project = ${idx}")
        params.append(project); idx += 1
    if role:
        conds.append(f"role = ${idx}")
        params.append(role); idx += 1
    if days:
        conds.append(f"created_at > NOW() - INTERVAL '{int(days)} days'")

    where_extra = (" AND " + " AND ".join(conds)) if conds else ""

    if not hybrid:
        sql = f"""
            SELECT id, session_id, project, role, content, model, created_at,
                   1 - (embedding <=> $1::vector) AS score
            FROM messages
            WHERE embedding IS NOT NULL{where_extra}
            ORDER BY embedding <=> $1::vector
            LIMIT {int(limit)}
        """
    else:
        sql = f"""
            WITH vec AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rnk
                FROM messages
                WHERE embedding IS NOT NULL{where_extra}
                ORDER BY embedding <=> $1::vector
                LIMIT {int(candidates)}
            ),
            lex AS (
                SELECT id, ROW_NUMBER() OVER (
                           ORDER BY ts_rank(to_tsvector('english', content),
                                            websearch_to_tsquery('english', $2)) DESC
                       ) AS rnk
                FROM messages
                WHERE to_tsvector('english', content)
                      @@ websearch_to_tsquery('english', $2){where_extra}
                ORDER BY ts_rank(to_tsvector('english', content),
                                 websearch_to_tsquery('english', $2)) DESC
                LIMIT {int(candidates)}
            ),
            fused AS (
                SELECT COALESCE(v.id, l.id) AS id,
                       COALESCE(1.0 / (60 + v.rnk), 0)
                     + COALESCE(1.0 / (60 + l.rnk), 0) AS score
                FROM vec v
                FULL OUTER JOIN lex l ON v.id = l.id
            )
            SELECT m.id, m.session_id, m.project, m.role, m.content, m.model,
                   m.created_at, f.score
            FROM fused f
            JOIN messages m ON m.id = f.id
            ORDER BY f.score DESC, m.created_at DESC
            LIMIT {int(limit)}
        """

    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    return {
        "query": q,
        "mode": "hybrid" if hybrid else "vector",
        "count": len(rows),
        "results": [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "project": r["project"],
                "role": r["role"],
                "content": r["content"],
                "model": r["model"],
                "created_at": r["created_at"].isoformat(),
                "similarity": round(float(r["score"]), 4),
            }
            for r in rows
        ],
    }

@app.get("/session/{session_id}")
async def get_session(session_id: str, limit: int = 500):
    """Return the full message list for a session in chronological order."""
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, role, content, model, created_at
            FROM messages
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2
            """,
            session_id, limit,
        )
    return {
        "session_id": session_id,
        "count": len(rows),
        "messages": [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "model": r["model"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ],
    }
