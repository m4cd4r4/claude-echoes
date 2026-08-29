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

import asyncio
import json
import os
import time
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

# --- re-ranker -------------------------------------------------------------
# Retrieval matches on shared vocabulary. When a question and its answer share
# none, both arms miss: "what did we decide about the ERF hero video" never
# reaches the message that says "triptych" and "IMG_5506", because the question
# contains neither word. No amount of lexical relaxation fixes that - the terms
# are simply absent.
#
# A re-ranker reads the question against each candidate and judges relevance
# directly, so it does not need shared words. This is the component the README's
# 86.4% LongMemEval figure was measured WITH (benchmarks/run_longmemeval.py) and
# that /search shipped WITHOUT.
#
# It runs locally on ollama, so the "no conversation data leaves the machine"
# property survives. One BATCHED call scores every candidate at once - scoring
# them individually would be N generations per query and is not worth it.
RERANK_MODEL   = os.environ.get("ECHOES_RERANK_MODEL", "qwen2.5:3b-instruct")
RERANK_ENABLED = os.environ.get("ECHOES_RERANK", "1") not in ("0", "false", "False")
RERANK_POOL    = 24    # candidates handed to the model
RERANK_SNIPPET = 420   # chars of each candidate the model sees
RERANK_TIMEOUT = 60

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
    # Warm both models in the background. A cold load of the re-ranker costs
    # ~60s, which is longer than its own timeout - so without this the FIRST
    # query of a fresh stack always reports rerank=TimeoutError and silently
    # falls back to RRF order. Measured 2026-08-29. Fire-and-forget: a warm-up
    # that blocks startup would make the server look hung instead of slow.
    async def _warm():
        try:
            await embed_text(app.state.http, "warmup")
        except Exception:
            pass
        if RERANK_ENABLED:
            try:
                async with app.state.http.post(
                    OLLAMA_URL + "/api/generate",
                    json={"model": RERANK_MODEL, "prompt": "Reply with [1]",
                          "stream": False, "keep_alive": -1,
                          "options": {"num_predict": 8}},
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as r:
                    await r.read()
            except Exception:
                pass

    app.state.warm = asyncio.create_task(_warm())

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
            # keep_alive is sent per request as well as set on the container,
            # so the model stays resident against a stock ollama the user
            # started themselves. See docker-compose.yml for the measurement.
            json={"model": OLLAMA_MODEL,
                  "prompt": (text or "")[:MAX_CONTENT_CHARS],
                  "keep_alive": -1},
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


# websearch_to_tsquery ANDs every content word, so a natural-language QUESTION
# ("how many certifications does AzurePrep have") narrows the conjunction until
# it matches almost nothing - measured 2026-08-29: 5 rows for the question form
# against 511 for the same query stripped to its content words. The hybrid then
# silently degraded to vector-only, which is weakest on exactly the "which
# number was it" lookups this tool exists for.
#
# So we keep AND first (it is precise when it fires) and fall back to OR when it
# returns almost nothing. ts_rank still rewards documents carrying MORE of the
# terms, so the OR arm degrades gracefully rather than flooding.
_STOP = {
    "a","an","and","are","as","at","be","but","by","did","do","does","for","from",
    "get","give","had","has","have","how","i","if","in","into","is","it","its","many",
    "me","much","my","of","on","or","our","should","so","tell","that","the","their",
    "them","then","there","these","this","to","was","we","were","what","when","where",
    "which","who","why","will","with","would","you","your","actually","really","just",
}
DF_CAP = 2000  # capped document frequency: only the ordering matters  # below this many AND hits, retry the lexical arm with OR


def content_words(q: str) -> list:
    """Query words with stopwords and question words dropped, deduped, and
    sanitised to what to_tsquery accepts."""
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z0-9_]+", q.lower()):
        if len(w) < 2 or w in _STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


async def pick_lex_query(conn, q: str, where_sql: str, where_args: list) -> Optional[str]:
    """Choose the lexical tsquery by PROGRESSIVE RELAXATION, in ONE statement.

    Measured on 108,570 messages, 2026-08-29:

    1. websearch_to_tsquery ANDs every content word, so the raw question
       "how many certifications and questions does AzurePrep have" matched
       5 rows and missed the answer. Stripped of stopwords it matched 511,
       answer included. Dropping question words is free precision.
    2. A pure OR fallback matched 29,324 rows and cost 29-46 seconds a query -
       ts_rank has no IDF term, so a wide OR buries the rare word. Rejected.
    3. Relaxing AND from all content words downward is precise AND cheap - but
       it MUST be one statement. Running the tiers as a LOOP of prepared
       probes was measured at 71-139 SECONDS: asyncpg prepares each probe, and
       after a few executions Postgres switches to a GENERIC plan that cannot
       use the tsquery's selectivity, falls back to a sequential scan, and
       recomputes to_tsvector over every row. The same probes under EXPLAIN
       ANALYZE took 2ms each, which is why the loop looked safe.

    Longest words first, as a cheap proxy for rare. ORDERING BY REAL DOCUMENT
    FREQUENCY WAS TRIED AND MEASURED WORSE: 5/7 against 6/7 on the graded set
    in scripts/eval_retrieval.mjs. Rarest-first builds an over-specific
    conjunction that matches SOMETHING early, and the loop stops at the first
    non-empty tier, so a spurious rare-term match beats the right answer. It
    fixed the one case it was designed for ("429") and broke a different one.
    Seven cases is too small to tune on, so the simpler rule stays.
    """
    words = content_words(q)
    if not words:
        return None
    ranked = sorted(words, key=len, reverse=True)
    tiers = [" & ".join(ranked[:k]) for k in range(len(ranked), 1, -1)]
    sql = (
        "SELECT t.q FROM unnest($1::text[]) WITH ORDINALITY AS t(q, ord) "
        "WHERE EXISTS (SELECT 1 FROM messages "
        "              WHERE to_tsvector('english', content) "
        f"                    @@ to_tsquery('english', t.q){where_sql}) "
        "ORDER BY t.ord LIMIT 1"
    )
    try:
        return await conn.fetchval(sql, tiers, *where_args)
    except Exception:
        return None


async def rerank(http, q: str, rows: list, want: int) -> tuple:
    """Reorder candidates by asking a small local model to judge relevance.

    ONE batched call: the model sees the question and every numbered snippet,
    and returns the indices it judges relevant, best first. Scoring candidates
    one at a time would be N generations per query for the same answer.

    Fails OPEN. A re-ranker that times out, returns nothing, or emits junk must
    leave the RRF order untouched rather than empty the result set - a silent
    quality regression is recoverable, a silent empty page is not. Every exit
    path returns rows plus a reason string, so /search can report which ranking
    the caller is actually looking at.
    """
    if not rows:
        return rows, "no candidates"
    lines = []
    for i, r in enumerate(rows):
        body = " ".join((r["content"] or "").split())[:RERANK_SNIPPET]
        lines.append("[%d] (%s, %s) %s" % (i, r["role"], r["created_at"].strftime("%Y-%m-%d"), body))
    prompt = (
        "You rank past chat messages by how well they ANSWER a question.\n"
        "A message can be highly relevant while sharing no words with the "
        "question - judge the subject matter, not the wording.\n\n"
        "QUESTION: " + q + "\n\nMESSAGES:\n" + "\n".join(lines) +
        "\n\nReturn ONLY a JSON array of the message numbers that help answer "
        "the question, best first, at most " + str(want) + ". No prose. Example: [3,0,7]"
    )
    try:
        async with http.post(
            OLLAMA_URL + "/api/generate",
            json={"model": RERANK_MODEL, "prompt": prompt, "stream": False,
                  "keep_alive": -1, "options": {"temperature": 0, "num_predict": 64}},
            timeout=aiohttp.ClientTimeout(total=RERANK_TIMEOUT),
        ) as r:
            if r.status != 200:
                return rows, "http %d" % r.status
            raw = (await r.json()).get("response", "")
    except Exception as e:
        return rows, type(e).__name__

    m = re.search(r"\[[^\]]*\]", raw)
    if not m:
        return rows, "no json array"
    try:
        order = [int(x) for x in json.loads(m.group(0))]
    except Exception:
        return rows, "unparseable"

    seen, picked = set(), []
    for i in order:
        if 0 <= i < len(rows) and i not in seen:
            seen.add(i)
            picked.append(rows[i])
    if not picked:
        # The model judged nothing relevant. That is a real answer for a
        # question the archive cannot serve, but it is indistinguishable from a
        # sulking 3B model, so keep RRF order and say so.
        return rows, "model returned empty"
    # Anything it did not name keeps its RRF order behind what it did.
    picked += [r for i, r in enumerate(rows) if i not in seen]
    return picked, "ok"


@app.get("/health")
async def health():
    """Exercise BOTH dependencies. A health check that cannot fail while the
    service is down is decoration - this one returned ok:true through an outage
    on 2026-08-29 because it only read a model name out of config."""
    db = "down"
    async with app.state.pool.acquire() as conn:
        if await conn.fetchval("SELECT 1") == 1:
            db = "up"
    embed_ok = await embed_text(app.state.http, "health") is not None
    ok = db == "up" and embed_ok
    body = {"ok": ok, "db": db, "embeddings": "up" if embed_ok else "down",
            "model": OLLAMA_MODEL}
    if not ok:
        raise HTTPException(503, body)
    return body

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
    rerank_: bool = Query(True, alias="rerank"),
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
    _t0 = time.perf_counter()
    qvec = await embed_text(app.state.http, q)
    _t_embed = time.perf_counter() - _t0
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
    # $2 is the chosen lexical tsquery, resolved below against the same filters
    # the main query uses - a probe that ignored the filters could pick a query
    # whose only matches are then filtered away, leaving the lexical arm empty.
    params: list = [qvec] if not hybrid else [qvec, ""]
    idx = len(params) + 1

    do_rerank = rerank_ and RERANK_ENABLED
    # Pull a deeper pool when re-ranking: the whole point is that the right
    # answer may sit below the RRF cut-off, so handing the model only the
    # top-`limit` rows would ask it to reorder a set the answer is not in.
    sql_limit = max(int(limit), RERANK_POOL) if do_rerank else int(limit)

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
            LIMIT {int(sql_limit)}
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
                                            to_tsquery('english', $2)) DESC
                       ) AS rnk
                FROM messages
                WHERE to_tsvector('english', content)
                      @@ to_tsquery('english', $2){where_extra}
                ORDER BY ts_rank(to_tsvector('english', content),
                                 to_tsquery('english', $2)) DESC
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
            LIMIT {int(sql_limit)}
        """

    async with app.state.pool.acquire() as conn:
        if hybrid:
            # Filter placeholders start at $3 in the main query; the probe only
            # binds them, so renumber them to $2.. for its own statement.
            probe_where = where_extra
            for n in range(idx - 1, 2, -1):
                probe_where = probe_where.replace(f"${n}", f"${n - 1}")
            lexq = await pick_lex_query(conn, q, probe_where, params[2:])
            if lexq is None:
                # No lexical arm at all - RRF degrades to the vector ranking,
                # which is the correct behaviour, not an error.
                lexq = "zzzz_no_lexical_match_zzzz"
            params[1] = lexq
        _t1 = time.perf_counter()
        rows = await conn.fetch(sql, *params)
        _t_sql = time.perf_counter() - _t1
        _t_probe = _t1 - _t0 - _t_embed

    rank_note = "off"
    _t_rank = 0.0
    if do_rerank:
        _t2 = time.perf_counter()
        rows, rank_note = await rerank(app.state.http, q, list(rows), int(limit))
        _t_rank = time.perf_counter() - _t2
    rows = rows[: int(limit)]

    return {
        "query": q,
        "mode": "hybrid" if hybrid else "vector",
        "lex_query": (params[1] if hybrid else None),
        "rerank": rank_note,
        "timings_ms": {"embed": round(_t_embed*1000), "probe": round(_t_probe*1000),
                       "sql": round(_t_sql*1000), "rerank": round(_t_rank*1000)},
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
