"""
claude-echoes server

Minimal FastAPI service that:
  - accepts message writes from the Claude Code hook
  - embeds them via local Ollama (nomic-embed-text)
  - stores them in Postgres with pgvector
  - exposes a semantic search endpoint with optional BM25+pgvector RRF hybrid

Everything is inline in one file on purpose. If you can't read it end to end
in 5 minutes, something has gone wrong.

Hybrid search (added):
  GET /search?q=...&hybrid=true   — RRF fusion of pgvector cosine + Postgres
                                    full-text search (GIN tsvector index)
  GET /search/hybrid?q=...        — same, explicit endpoint

The GIN index (idx_messages_fts) already exists in the schema; this wires it.
RRF formula: score(doc) = Σ 1/(k + rank_i), k=60 (Cormack et al. 2009).
Benchmark evidence: BM25 hybrid gives +7–15 points on temporal and rare-term
queries over pure cosine (see benchmarks/README.md for full breakdown).
"""
from __future__ import annotations

import os
import re
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

# RRF constant from Cormack et al. 2009. Higher k = less rank-sensitivity.
_RRF_K = 60
_WIDE_K = 30     # candidates per leg before fusion

def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 query construction. Mirrors benchmarks/run_longmemeval.py."""
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if 2 <= len(t) <= 30]


def _rrf_merge(
    vec_rows: list[dict],
    fts_rows: list[dict],
    k: int = _RRF_K,
    limit: int = 10,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over two ranked lists.

    score(doc) = Σ_i  1 / (k + rank_i)

    Both lists are dicts with an 'id' key. The function preserves all fields
    from whichever list contributed the higher-ranked entry for each doc.
    Returns top-limit docs sorted by fused score descending.
    """
    scores: dict[int, dict] = {}
    for rank, row in enumerate(vec_rows):
        doc_id = row["id"]
        scores.setdefault(doc_id, {"row": row, "rrf": 0.0})
        scores[doc_id]["rrf"] += 1.0 / (k + rank + 1)
    for rank, row in enumerate(fts_rows):
        doc_id = row["id"]
        scores.setdefault(doc_id, {"row": row, "rrf": 0.0})
        scores[doc_id]["rrf"] += 1.0 / (k + rank + 1)
    ranked = sorted(scores.values(), key=lambda v: -v["rrf"])
    return [v["row"] for v in ranked[:limit]]


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

async def _search_vec(
    conn,
    qvec: str,
    project: Optional[str],
    role: Optional[str],
    days: Optional[int],
    wide_k: int,
) -> list[dict]:
    """pgvector ANN leg: top wide_k by cosine distance."""
    conds = ["embedding IS NOT NULL"]
    params: list = [qvec]
    idx = 2

    if project:
        conds.append(f"project = ${idx}"); params.append(project); idx += 1
    if role:
        conds.append(f"role = ${idx}"); params.append(role); idx += 1
    if days:
        conds.append(f"created_at > NOW() - INTERVAL '{int(days)} days'")

    sql = f"""
        SELECT id, session_id, project, role, content, model, created_at,
               1 - (embedding <=> $1::vector) AS similarity
        FROM messages
        WHERE {" AND ".join(conds)}
        ORDER BY embedding <=> $1::vector
        LIMIT {wide_k}
    """
    rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


async def _search_fts(
    conn,
    query_tokens: list[str],
    project: Optional[str],
    role: Optional[str],
    days: Optional[int],
    wide_k: int,
) -> list[dict]:
    """
    Full-text search leg using the GIN tsvector index (idx_messages_fts).

    Constructs a plainto_tsquery from the tokenized query terms. The GIN
    index already exists in the schema; this is the first server endpoint
    to use it for retrieval.
    """
    if not query_tokens:
        return []

    # Build tsquery: "term1 & term2 & ..."
    tsquery = " & ".join(query_tokens)

    conds = ["to_tsvector('english', content) @@ plainto_tsquery('english', $1)"]
    params: list = [" ".join(query_tokens)]
    idx = 2

    if project:
        conds.append(f"project = ${idx}"); params.append(project); idx += 1
    if role:
        conds.append(f"role = ${idx}"); params.append(role); idx += 1
    if days:
        conds.append(f"created_at > NOW() - INTERVAL '{int(days)} days'")

    sql = f"""
        SELECT id, session_id, project, role, content, model, created_at,
               ts_rank_cd(to_tsvector('english', content),
                          plainto_tsquery('english', $1)) AS fts_rank
        FROM messages
        WHERE {" AND ".join(conds)}
        ORDER BY fts_rank DESC
        LIMIT {wide_k}
    """
    rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


def _format_results(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "session_id": r["session_id"],
            "project": r["project"],
            "role": r["role"],
            "content": r["content"],
            "model": r["model"],
            "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
            "similarity": round(float(r.get("similarity", r.get("fts_rank", 0.0))), 4),
        })
    return out


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    project: Optional[str] = None,
    role: Optional[str] = None,
    days: Optional[int] = Query(None, ge=1, le=3650),
    hybrid: bool = Query(False, description="Use BM25+pgvector RRF hybrid search"),
):
    """
    Semantic search over stored Claude Code messages.

    By default, uses pure pgvector cosine similarity. Pass `hybrid=true` to
    enable RRF fusion with the Postgres GIN full-text index — recommended for
    queries with specific rare terms, names, or temporal questions.
    """
    if hybrid:
        return await _search_hybrid_impl(q=q, limit=limit, project=project,
                                         role=role, days=days)

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
        "hybrid": False,
        "count": len(rows),
        "results": _format_results([dict(r) for r in rows]),
    }


async def _search_hybrid_impl(
    q: str,
    limit: int,
    project: Optional[str],
    role: Optional[str],
    days: Optional[int],
    wide_k: int = _WIDE_K,
) -> dict:
    """
    Hybrid search implementation: pgvector cosine + GIN full-text RRF fusion.

    Retrieves top wide_k candidates from each leg independently, then fuses
    them with Reciprocal Rank Fusion (k=60). Both legs run in the same DB
    connection; embed call is the only async I/O overhead beyond vanilla search.

    Falls back to pure pgvector if the FTS leg returns no results (e.g. the
    query contains no indexable terms).

    Evidence for why this matters:
      - BM25 hybrid gave +7-15 points on LongMemEval temporal-reasoning and
        single-session-user categories (see benchmarks/README.md).
      - The GIN index already exists (idx_messages_fts); this is the first
        server code that uses it.
    """
    if role and role not in ("user", "assistant"):
        raise HTTPException(400, "role must be 'user' or 'assistant'")

    qvec = await embed_text(app.state.http, q)
    if qvec is None:
        raise HTTPException(503, "embedding service unavailable")

    query_tokens = _tokenize(q)

    async with app.state.pool.acquire() as conn:
        vec_rows = await _search_vec(conn, qvec, project, role, days, wide_k)
        fts_rows = await _search_fts(conn, query_tokens, project, role, days, wide_k)

    if not fts_rows:
        # FTS returned nothing (no indexable terms) — degrade to pure vector
        fused = vec_rows[:limit]
    else:
        fused = _rrf_merge(vec_rows, fts_rows, k=_RRF_K, limit=limit)

    return {
        "query": q,
        "hybrid": True,
        "vec_hits": len(vec_rows),
        "fts_hits": len(fts_rows),
        "count": len(fused),
        "results": _format_results(fused),
    }


@app.get("/search/hybrid")
async def search_hybrid(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    project: Optional[str] = None,
    role: Optional[str] = None,
    days: Optional[int] = Query(None, ge=1, le=3650),
    wide_k: int = Query(_WIDE_K, ge=5, le=100,
                        description="Candidates per leg before RRF fusion"),
):
    """
    Hybrid BM25+pgvector RRF search endpoint.

    Explicitly separate from /search so it can be called directly without
    the ?hybrid=true parameter — useful for integrations and benchmarking
    that want to distinguish the two retrieval modes cleanly.

    Returns extra diagnostic fields (vec_hits, fts_hits) that /search omits.
    """
    return await _search_hybrid_impl(q=q, limit=limit, project=project,
                                      role=role, days=days, wide_k=wide_k)

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
