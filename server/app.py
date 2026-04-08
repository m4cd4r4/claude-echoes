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
):
    qvec = await embed_text(app.state.http, q)
    if qvec is None:
        raise HTTPException(503, "embedding service unavailable")

    conds = ["embedding IS NOT NULL"]
    params: list = [qvec]
    idx = 2

    if project:
        conds.append(f"project = ${idx}")
        params.append(project); idx += 1
    if role:
        if role not in ("user", "assistant"):
            raise HTTPException(400, "role must be 'user' or 'assistant'")
        conds.append(f"role = ${idx}")
        params.append(role); idx += 1
    if days:
        conds.append(f"created_at > NOW() - INTERVAL '{int(days)} days'")

    sql = f"""
        SELECT id, session_id, project, role, content, model, created_at,
               1 - (embedding <=> $1::vector) AS similarity
        FROM messages
        WHERE {" AND ".join(conds)}
        ORDER BY embedding <=> $1::vector
        LIMIT {int(limit)}
    """
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    return {
        "query": q,
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
                "similarity": round(float(r["similarity"]), 4),
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
