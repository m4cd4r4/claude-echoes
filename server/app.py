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
from datetime import datetime
from typing import Optional

import aiohttp
import asyncpg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from chunking import CHUNK_MIN_CHARS, CHUNK_SIZE, split_chunks
from intent import (build_project_index, collapse_near_duplicates,
                    lead_fingerprint, project_words, recency_intent, reorder,
                    resolve_project)

# --- config ---------------------------------------------------------------

DB_DSN       = os.environ.get("ECHOES_DB_DSN",
                              "postgresql://echoes:echoes@postgres:5432/echoes")
OLLAMA_URL   = os.environ.get("ECHOES_OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("ECHOES_OLLAMA_MODEL", "nomic-embed-text")
EMBED_TIMEOUT_S = float(os.environ.get("ECHOES_EMBED_TIMEOUT", "4"))
MAX_CONTENT_CHARS = 8000   # nomic context window is 8192 tokens; clip safely

# --- re-ranker -------------------------------------------------------------
# Retrieval matches on shared vocabulary. When a question and its answer share
# none, both arms miss: "what did we decide about the hero video" never
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
RERANK_MODEL   = os.environ.get("ECHOES_RERANK_MODEL", "qwen2.5:7b-instruct")
RERANK_ENABLED = os.environ.get("ECHOES_RERANK", "1") not in ("0", "false", "False")
RERANK_POOL    = 24    # candidates handed to the model
RERANK_SNIPPET = 420   # chars of each candidate the model sees
RERANK_TIMEOUT = 60
# Backend: "llm" = the qwen listwise call below; "ce" = a cross-encoder
# (bge-reranker-v2-m3 on TEI, docker-compose.rerank.yml) that scores each
# (question, snippet) pair. The cross-encoder is cheap per pair, so it reads a
# longer snippet than the LLM prompt can afford.
RERANK_BACKEND = os.environ.get("ECHOES_RERANK_BACKEND", "llm")
CE_URL         = os.environ.get("ECHOES_CE_URL", "http://reranker:80")
CE_SNIPPET     = int(os.environ.get("ECHOES_CE_SNIPPET", "1500"))
CE_TIMEOUT     = 20
CE_BATCH       = 32    # TEI default --max-client-batch-size

# Recency tie-break. See the sizing note in /search before changing these.
# HNSW recall. ef_search below the requested candidate count silently truncates
# the search; see the measurement in /search. 4x with a floor of 200 keeps recall
# high at the default candidates=60 without making a deep pool pathological.
EF_SEARCH_FACTOR = int(os.environ.get("ECHOES_EF_SEARCH_FACTOR", "4"))
EF_SEARCH_MIN    = int(os.environ.get("ECHOES_EF_SEARCH_MIN", "200"))

RECENCY_WEIGHT = float(os.environ.get("ECHOES_RECENCY_WEIGHT", "0.0008"))
RECENCY_HALFLIFE_DAYS = float(os.environ.get("ECHOES_RECENCY_HALFLIFE_DAYS", "45"))

# Abstention. Needs the same local model as the re-ranker, so it is GPU-gated
# alongside it and off in the base stack.
ABSTAIN_ENABLED = os.environ.get("ECHOES_ABSTAIN", "0") not in ("0", "", "false")
# The judge sees the CANDIDATE POOL, not the final page. Judging the top 5 asks
# it to rule on rows the re-ranker has not sorted yet: measured 2026-09-10, a
# correct answer sat at RRF rank 17, was invisible to a top-5 judge, and the
# query abstained on a question the corpus could answer. 12x500 keeps the prompt
# inside the 4096-token context that 420x24 would overflow.
ABSTAIN_ROWS = 12
ABSTAIN_SNIPPET = 500
ABSTAIN_TIMEOUT = 45

# Index-time chunks (sql/005_chunks.sql, server/chunking.py). Both default OFF
# so the flags can be measured apart: CHUNK_SEARCH changes which rows reach the
# pool; JUDGE_CHUNK changes which 500 chars of a long row the judge reads.
# Writes always chunk once the table exists, flags or not, so a later switch-on
# finds new messages already indexed.
CHUNK_SEARCH = os.environ.get("ECHOES_CHUNK_SEARCH", "0") not in ("0", "", "false")
JUDGE_CHUNK  = os.environ.get("ECHOES_JUDGE_CHUNK", "0") not in ("0", "", "false")
# RERANK_CHUNK: the re-ranker reads the matched chunk's best window instead of
# the row's head. Only changes anything alongside CHUNK_SEARCH.
RERANK_CHUNK = os.environ.get("ECHOES_RERANK_CHUNK", "0") not in ("0", "", "false")
# CHUNK_CAP: max parents per arm that may enter the pool through a chunk.
# 0 = no cap. Measured 2026-09-19 because chunk rows crowded out whole hits.
CHUNK_CAP    = int(os.environ.get("ECHOES_CHUNK_CAP", "0") or 0)
# PAIR_GATE: abstain when a question names a subject AND a topic, both known to
# the corpus, and no retrieved passage mentions both. The compositional-absence
# case the judge is worst at - see pair_uncovered().
PAIR_GATE    = os.environ.get("ECHOES_PAIR_GATE", "0") not in ("0", "", "false")

# Query-intent handling (server/intent.py). Each is its own flag so a
# regression can be pinned on one of them.
#
# RECENCY_INTENT: "when did I last...", "latest", "most recent" is answered by
# date, not relevance. With a resolved project it returns that project's newest
# sessions; without one it re-orders the relevant results newest first.
RECENCY_INTENT = os.environ.get("ECHOES_RECENCY_INTENT", "0") not in ("0", "", "false")
# PROJECT_RESOLVE: a project named in the question ("the billing service")
# pulls that project's best matches into the candidate pool, so the re-ranker
# can choose them, and a recency question about it is answered from it. A
# boost, not a filter - an explicit project= param still filters.
# PROJECT_LIFT additionally moves its rows up that many places AFTER the
# re-ranker. Measured 2026-09-22 on the graded set (108 cases): lift 10 lost one
# single_session case, lift 3 kept Recall@5 but cost MRR 0.601 -> 0.582, lift 0
# (pool injection only) held both. So it defaults to 0.
PROJECT_RESOLVE = os.environ.get("ECHOES_PROJECT_RESOLVE", "0") not in ("0", "", "false")
PROJECT_LIFT    = int(os.environ.get("ECHOES_PROJECT_LIFT", "0"))
PROJECT_POOL    = 10    # per arm, rows pulled from the resolved project
PROJECT_TTL_S   = float(os.environ.get("ECHOES_PROJECT_CACHE_TTL", "300"))
PROJECT_IGNORE  = [p.strip() for p in
                   os.environ.get("ECHOES_PROJECT_IGNORE", "").split(",") if p.strip()]
# DEDUPE: collapse results whose normalised leading text is identical, keeping
# the best-ranked one. Templated messages otherwise fill a page with one answer.
DEDUPE       = os.environ.get("ECHOES_DEDUPE", "0") not in ("0", "", "false")
DEDUPE_CHARS = int(os.environ.get("ECHOES_DEDUPE_CHARS", "200"))
# AUTOMATION_DEMOTE: a user-role message whose leading text is shared by
# AUTOMATION_MIN_SESSIONS or more sessions is a scripted prompt (a headless
# job, a pasted template), not something the person asked. Such rows go to the
# back of the page.
AUTOMATION_DEMOTE = os.environ.get("ECHOES_AUTOMATION_DEMOTE", "0") not in ("0", "", "false")
AUTOMATION_MIN_SESSIONS = int(os.environ.get("ECHOES_AUTOMATION_MIN_SESSIONS", "5"))
AUTOMATION_CHARS = 120
AUTOMATION_MIN_LEN = 40   # a shared "continue" or "yes" is not automation
AUTOMATION_TTL_S = 900
# Mirrors intent.lead_fingerprint() and sql/006 - keep the three in step.
# With sql/006 applied the prints live in message_fp; without it, they are
# computed inline (correct, but ~15 s over ~60k user rows, so it runs in the
# background cache refresh, never on the request path after the first load).
_FP_INLINE = (
    f"lower(left(btrim(regexp_replace(regexp_replace("
    f"left(content, {AUTOMATION_CHARS + 40}), '[0-9]', '#', 'g'), "
    f"'[[:space:]]+', ' ', 'g')), {AUTOMATION_CHARS}))")

def automation_sql(stored: bool) -> str:
    src = ("SELECT fp, session_id FROM message_fp" if stored else
           f"SELECT {_FP_INLINE} AS fp, session_id FROM messages WHERE role = 'user'")
    return f"""
        SELECT fp FROM ({src}) t
        WHERE length(fp) >= {AUTOMATION_MIN_LEN}
        GROUP BY fp HAVING count(DISTINCT session_id) >= {AUTOMATION_MIN_SESSIONS}
    """

# --- models ---------------------------------------------------------------

class MessageIn(BaseModel):
    session_id: str
    project: str = "unknown"
    machine: Optional[str] = None
    role: str                    # "user" | "assistant"
    content: str
    model: Optional[str] = None
    # When the turn actually happened. Omit and the column default now() applies,
    # which is only correct for a live write. A queued turn drained after an
    # outage, or an imported transcript, MUST send the real time or every
    # `days`-filtered search lies about it - and because content_hash is
    # generated from created_at, a wrong time also defeats the dedupe index.
    created_at: Optional[datetime] = None

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
    # A stack that never ran sql/005 keeps working exactly as before: no chunk
    # writes, and the search flags are ignored rather than turned into 500s.
    async with app.state.pool.acquire() as conn:
        app.state.chunks = bool(await conn.fetchval(
            "SELECT to_regclass('message_chunks') IS NOT NULL"))
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
        # Load the intent caches now rather than on the first query.
        if PROJECT_RESOLVE:
            await _projects_cache.get(app.state.pool)
        if AUTOMATION_DEMOTE:
            await _automated_cache.get(app.state.pool)
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
# The fix is NOT a pure OR fallback - that was measured at 29-46s a query,
# because ts_rank has no IDF term and a wide OR buries the rare word. Instead
# the AND is relaxed one term at a time, longest first, in a single statement.
_STOP = {
    "a","an","and","are","as","at","be","but","by","did","do","does","for","from",
    "get","give","had","has","have","how","i","if","in","into","is","it","its","many",
    "me","much","my","of","on","or","our","should","so","tell","that","the","their",
    "them","then","there","these","this","to","was","we","were","what","when","where",
    "which","who","why","will","with","would","you","your","actually","really","just",
    # Question-template verbs. Left in, they become REQUIRED terms in the AND:
    # "tailwind & config & server" matched 20 rows, "+ learn" matched 1.
    "know","knew","learn","learned","learnt","find","decide","decided",
    "happen","happened","about","go","went","can","could","may","might","must",
}


def lex_term(w: str) -> str:
    """One content word as a tsquery term, covering the english stemmer's
    y->i split.

    "certified" stems to certifi but "certification" to certif, so an exact
    AND on the question's stem dropped the only row that held the answer. A
    word with a y->i ending (-ied, -ies, -y) matches either stem.

    General prefix matching (every term as "word:*") was measured and rejected
    2026-09-19: it gained one single_session case but pushed multi_session
    gold down and lowered temporal MRR - broad prefixes are lexical noise.
    """
    base = None
    if len(w) > 5 and (w.endswith("ied") or w.endswith("ies")):
        base = w[:-3]
    elif len(w) > 5 and w.endswith("y"):
        base = w[:-1]
    return f"({base} | {w})" if base else w

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
    tiers = [" & ".join(lex_term(w) for w in ranked[:k])
             for k in range(len(ranked), 1, -1)]
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
    if RERANK_BACKEND == "ce":
        return await rerank_ce(http, q, rows)
    lines = []
    for i, r in enumerate(rows):
        # Window first, then whitespace-collapse: the chunk offset indexes the
        # raw content, so collapsing first would shift it.
        body = " ".join(chunk_window(r, q, RERANK_SNIPPET, RERANK_CHUNK).split())[:RERANK_SNIPPET]
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
        # judge that simply declined to answer, so keep RRF order and say so.
        return rows, "model returned empty"
    # Anything it did not name keeps its RRF order behind what it did.
    picked += [r for i, r in enumerate(rows) if i not in seen]
    return picked, "ok"


async def rerank_ce(http, q: str, rows: list) -> tuple:
    """Cross-encoder re-rank: score every row against the question, sort by
    score. Fails open to the incoming (RRF) order, same contract as rerank()."""
    texts = [" ".join(chunk_window(r, q, CE_SNIPPET, RERANK_CHUNK).split())
             for r in rows]
    # TEI refuses more than 32 texts per request (HTTP 413). Each pair is
    # scored independently, so batching changes nothing but the request count.
    scored = []
    try:
        for off in range(0, len(texts), CE_BATCH):
            async with http.post(
                CE_URL + "/rerank",
                json={"query": q, "texts": texts[off:off + CE_BATCH], "truncate": True},
                timeout=aiohttp.ClientTimeout(total=CE_TIMEOUT),
            ) as r:
                if r.status != 200:
                    return rows, "ce http %d" % r.status
                scored += [{**s, "index": s["index"] + off} for s in await r.json()]
    except Exception as e:
        return rows, "ce " + type(e).__name__
    try:
        order = [s["index"] for s in sorted(scored, key=lambda s: -s["score"])]
    except Exception:
        return rows, "ce unparseable"
    if sorted(order) != list(range(len(rows))):
        return rows, "ce bad indices"
    return [rows[i] for i in order], "ok"


def judge_snippet(r, q: str) -> str:
    """The ABSTAIN_SNIPPET chars of a row the judge reads.

    Default: the head. Measured 2026-09-19 (diag report 04): a long wrap-up
    states its fact at char 4000+, the gold row was in the judge's 12 and it
    still abstained, because none of the 12 heads showed the answer.

    With JUDGE_CHUNK and a row that search reached through a chunk, the judge
    reads the matched chunk instead: of its ABSTAIN_SNIPPET-sized windows, the
    one holding the most query words, earliest on a tie. The window stays
    inside CHUNK_SIZE of the chunk's start - a term window over the WHOLE
    message was measured in report 04 and flips verdicts on merely on-topic
    text.
    """
    return chunk_window(r, q, ABSTAIN_SNIPPET, JUDGE_CHUNK)


def chunk_window(r, q: str, size: int, enabled: bool) -> str:
    """`size` chars of a row: its head, or - when `enabled` and search reached
    the row through a chunk - the size-char window inside the matched chunk
    holding the most query words, earliest on a tie."""
    content = r["content"] or ""
    sc = r.get("matched_chunk") if enabled else None
    if sc is None:
        return content[:size]
    chunk = content[sc:sc + CHUNK_SIZE]
    if len(chunk) <= size:
        return chunk
    words = content_words(q)
    low = chunk.lower()
    best, best_hits = 0, -1
    for s in range(0, len(chunk) - size + 1, 50):
        win = low[s:s + size]
        hits = sum(1 for w in words if w in win)
        if hits > best_hits:
            best, best_hits = s, hits
    return chunk[best:best + size]


# Cap on the document-frequency count. Only the RELATIVE rarity of the query's
# words matters, so counting past this is work nobody reads.
PAIR_DF_CAP = 5000

def pair_terms(q: str) -> tuple:
    """Split a question's content words into a named component and a topic
    component. A capitalised token that is not the first word of the question
    is treated as naming something - a client, a product, a place."""
    toks = re.findall(r"[A-Za-z0-9_]+", q)
    named = {w.lower() for w in toks[1:] if w[:1].isupper() and w[1:2].islower()}
    ent, topic = [], []
    for w in content_words(q):
        (ent if w in named else topic).append(w)
    return ent, topic


async def pair_df(conn, words: list) -> dict:
    """Document frequency of each word, counted through the FTS index."""
    if not words:
        return {}
    rows = await conn.fetch(
        "SELECT w, (SELECT count(*)::int FROM ("
        "          SELECT 1 FROM messages"
        "           WHERE to_tsvector('english', content)"
        "                 @@ plainto_tsquery('english', w)"
        f"          LIMIT {PAIR_DF_CAP}) z) AS n"
        " FROM unnest($1::text[]) AS w", words)
    return {r["w"]: r["n"] for r in rows}


def pair_rarest(dfs: dict, words: list):
    """The rarest of `words` that the corpus actually knows, or None. A word the
    corpus has never seen carries no evidence either way, so it is skipped
    rather than treated as proof of absence."""
    known = [(w, dfs[w]) for w in words if dfs.get(w, 0) > 0]
    return min(known, key=lambda x: x[1])[0] if known else None


async def pair_uncovered(conn, q: str, rows) -> bool:
    """True when the question is compositional and the corpus does not join it up.

    WHY A SEPARATE GATE. judge_abstain() is deliberately lenient - it answers
    YES on anything bearing on the question at all, because its one unacceptable
    failure is inventing absence. That leniency is exactly what compositional
    absence exploits: for 'the webhook work for <client>', passages about
    webhooks ARE background, and passages mentioning the client ARE background,
    so the judge answers even when the two were never discussed together.
    Measured 2026-09-20 on the 15 generated compound cases: the judge alone
    scores 0.333.

    So decompose instead of persuading. Take the rarest named word and the
    rarest topic word the corpus knows, and require ONE passage to contain both.
    Neither component alone is evidence; their conjunction is.

    Estimated 2026-09-20 over the case set before building: fires on 13 compound
    cases for +8, +1 abstention, -2 single_session. The losses are real and are
    the price - a gold passage that refers to its subject by a synonym is not
    covered. Fails open on any DB error, same contract as the judge.
    """
    ent, topic = pair_terms(q)
    if not ent or not topic or not rows:
        return False
    try:
        dfs = await pair_df(conn, ent + topic)
    except Exception:
        return False
    a, b = pair_rarest(dfs, ent), pair_rarest(dfs, topic)
    if not a or not b:
        return False
    for r in rows[:ABSTAIN_ROWS]:
        s = (r["content"] or "").lower()
        if a in s and b in s:
            return False
    return True


class _Cached:
    """A value refreshed from the DB at most every `ttl` seconds.

    A stale value is served while ONE background task refreshes it, so a query
    never waits on the refresh after the first load. `load` is awaited with a
    pooled connection."""
    def __init__(self, ttl: float, load, empty):
        self.ttl, self.load, self.value = ttl, load, empty
        self.at, self.task, self.loaded = 0.0, None, False

    async def _refresh(self, pool):
        try:
            async with pool.acquire() as conn:
                self.value = await self.load(conn)
            self.loaded = True
        except Exception:
            pass   # keep serving the last good value
        finally:
            self.at = time.monotonic()

    async def get(self, pool):
        if not self.loaded:
            # First use: nothing to serve yet, so wait (once) for the load.
            if self.task is None or self.task.done():
                self.task = asyncio.create_task(self._refresh(pool))
            await self.task
        elif time.monotonic() - self.at > self.ttl and (
                self.task is None or self.task.done()):
            self.task = asyncio.create_task(self._refresh(pool))
        return self.value


async def _load_projects(conn) -> dict:
    # Busiest first, so a key shared by two spellings of one folder resolves
    # to the one that holds the history.
    rows = await conn.fetch(
        "SELECT project, count(*)::int AS n FROM messages "
        "GROUP BY project ORDER BY n DESC")
    return {"index": build_project_index([r["project"] for r in rows], PROJECT_IGNORE),
            "counts": {r["project"]: r["n"] for r in rows}}


# A project named after an ordinary word ("search", "watch") matches questions
# that are not about it at all. The corpus says which names are ordinary words:
# one that appears in far more messages than the project itself holds is being
# used as a word, not as a name. Measured 2026-09-22: a project called
# "search" resolved from "how did we fix the HNSW ef_search recall problem".
COMMON_RATIO = 5
_common_names: dict = {}

async def is_common_name(conn, project: str, own: int) -> bool:
    if project in _common_names:
        return _common_names[project]
    cap = max(200, COMMON_RATIO * own) + 1
    phrase = project_words(project)
    try:
        n = await conn.fetchval(
            "SELECT count(*) FROM (SELECT 1 FROM messages "
            " WHERE to_tsvector('english', content) @@ phraseto_tsquery('english', $1)"
            f" LIMIT {cap}) z", phrase)
    except Exception:
        return False   # fail open: an unmeasured name keeps its boost
    _common_names[project] = n >= cap
    return _common_names[project]


async def _load_automated(conn) -> frozenset:
    stored = bool(await conn.fetchval("SELECT to_regclass('message_fp') IS NOT NULL"))
    return frozenset(r["fp"] for r in await conn.fetch(automation_sql(stored)))


_projects_cache  = _Cached(PROJECT_TTL_S, _load_projects, {"index": {}, "counts": {}})
_automated_cache = _Cached(AUTOMATION_TTL_S, _load_automated, frozenset())


async def project_rows(conn, qvec: str, lexq: str, project: str, extra_sql: str,
                       extra_args: list) -> list:
    """The resolved project's best matches: top PROJECT_POOL by vector and by
    lexical rank, restricted to that project.

    The main pool cannot be relied on to hold them. The HNSW index returns the
    corpus-wide nearest rows and a project filter applied after it keeps almost
    none of a small project, so the vector arm here orders by an EXPRESSION
    (distance + 0) that the index cannot serve - Postgres then reads the
    project's rows through idx_messages_project and sorts them exactly."""
    cols = "id, session_id, project, role, content, model, source, created_at"
    sql = f"""
        (SELECT {cols}, 1 - (embedding <=> $1::vector) AS score
         FROM messages
         WHERE project = $3 AND embedding IS NOT NULL{extra_sql}
         ORDER BY (embedding <=> $1::vector) + 0
         LIMIT {PROJECT_POOL})
        UNION
        (SELECT {cols}, 0.0 AS score
         FROM messages
         WHERE project = $3
           AND to_tsvector('english', content) @@ to_tsquery('english', $2){extra_sql}
         ORDER BY ts_rank(to_tsvector('english', content),
                          to_tsquery('english', $2)) DESC
         LIMIT {PROJECT_POOL})
    """
    try:
        return list(await conn.fetch(sql, qvec, lexq, project, *extra_args))
    except Exception:
        return []


async def project_recent(conn, project: str, extra_sql: str, extra_args: list,
                         limit: int, automated: frozenset) -> list:
    """The project's newest sessions, newest first, one row per session.

    The representative row is the session's LATEST substantive message (80+
    chars, not a scripted prompt): for "when did I last work on X" the answer
    is when the work stopped, and a closing message usually says what was done.
    """
    sql = f"""
        WITH s AS (
            SELECT session_id, max(created_at) AS last_at
            FROM messages WHERE project = $1{extra_sql}
            GROUP BY session_id
            ORDER BY last_at DESC
            LIMIT {int(limit) * 2}
        )
        SELECT * FROM (
            SELECT m.id, m.session_id, m.project, m.role, m.content, m.model,
                   m.source, m.created_at, 0.0 AS score, s.last_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY m.session_id
                       ORDER BY (length(btrim(m.content)) >= 80) DESC,
                                m.created_at DESC) AS rn
            FROM messages m JOIN s ON s.session_id = m.session_id
            WHERE m.project = $1{extra_sql}
        ) t
        WHERE rn <= 6   -- a few spares in case the latest is a scripted prompt
        ORDER BY last_at DESC, session_id, rn
    """
    rows = await conn.fetch(sql, project, *extra_args)
    out, cur, best = [], None, None
    for r in rows:   # rows arrive grouped by session, best candidate first
        if r["session_id"] != cur:
            if best is not None:
                out.append(best)
            cur, best = r["session_id"], None
        if best is None and not is_automated(r, automated):
            best = r
    if best is not None:
        out.append(best)
    return out[: int(limit)]


def is_automated(r, automated: frozenset) -> bool:
    return (r["role"] == "user" and bool(automated)
            and lead_fingerprint(r["content"], AUTOMATION_CHARS) in automated)


async def judge_abstain(session, q: str, rows) -> tuple[bool, str]:
    """Decide whether the corpus actually holds an answer. Returns (abstain, why).

    A SCORE FLOOR CANNOT DO THIS, and the measurement is worth recording because
    the score floor is the obvious design and it is wrong. Measured 2026-09-10
    against subjects SQL-verified absent from the index:

      query                                  RRF     cosine   really present?
      gitleaks sweep                         0.0323  0.7684   yes
      Squarespace TipTap setContent          0.0170  0.6905   yes
      Knurl App Store submission             0.0169  0.8485   NO
      Cloudflare Workers AEO collector       0.0325  0.6770   NO

    Both absent subjects outscore a present one on one metric or the other, and
    the highest cosine of the whole set belongs to a subject that was never
    discussed. The reason is that absence here is COMPOSITIONAL: 'Knurl' is a
    real project and 'App Store' is a real topic, so the query embeds close to
    genuine material no matter what. No threshold on retrieval score separates
    'we discussed this' from 'we discussed things near this', because the
    geometry is the same in both cases.

    So the question has to be asked of a model that can read the passages and
    judge entailment, not of a number. This is the re-ranker's model, one extra
    call, on the FINAL rows only.

    Two deliberate choices:

    - the snippet is longer than the re-ranker's 420 chars. Ranking needs only
      enough to tell rows apart; abstention needs enough to see whether the
      answer is actually stated, and a truncated passage reads as a non-answer.
    - it FAILS OPEN. If the model errors or returns anything unparseable, the
      rows are returned. A false abstention silently hides a corpus that does
      contain the answer, and the user cannot tell that from a genuine absence -
      whereas a false answer is at least visible and checkable. The one failure
      mode this feature must not have is inventing absence.
    """
    if not rows:
        return False, "no rows"

    ctx = "\n\n".join(
        f"[{i+1}] {judge_snippet(r, q)}"
        for i, r in enumerate(rows[:ABSTAIN_ROWS])
    )
    prompt = (
        f"Question: {q}\n\n"
        f"Passages retrieved from a chat archive:\n{ctx}\n\n"
        "Is there anything in the passages above a person could use to answer "
        "the question - even partially, indirectly, or as background?\n\n"
        "Answer NO only if the passages are about entirely different subjects "
        "and contain nothing bearing on the question at all. When in doubt, "
        "answer YES.\n"
        "Answer with only YES or NO."
    )
    try:
        async with session.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": RERANK_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_predict": 4}},
            timeout=aiohttp.ClientTimeout(total=ABSTAIN_TIMEOUT),
        ) as r:
            if r.status != 200:
                return False, f"judge http {r.status}"
            text = (await r.json()).get("response", "").strip().upper()
    except Exception as e:
        return False, f"judge unavailable: {type(e).__name__}"

    if text.startswith("NO"):
        return True, "judge: no passage answers the question"
    if text.startswith("YES"):
        return False, "judge: answered"
    return False, f"judge unparseable: {text[:12]!r}"

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

    # Chunks embed concurrently with the message itself, so a long write costs
    # about one embed of latency, not one per chunk.
    chunks = split_chunks(msg.content) if app.state.chunks else []
    emb, *chunk_embs = await asyncio.gather(
        embed_text(app.state.http, msg.content),
        *(embed_text(app.state.http, c) for _, c in chunks))

    # COALESCE so an omitted created_at still takes the column default. Passing
    # NULL explicitly would violate NOT NULL rather than fall back.
    # One transaction, so a message is never visible without its chunks - under
    # CHUNK_SEARCH a long message is searched ONLY through them.
    async with app.state.pool.acquire() as conn, conn.transaction():
        if emb is not None:
            row = await conn.fetchrow(
                """
                INSERT INTO messages (session_id, project, machine, role, content, model, embedding, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector, COALESCE($8, now()))
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id, created_at
                """,
                msg.session_id, msg.project, msg.machine, msg.role,
                msg.content, msg.model, emb, msg.created_at,
            )
        else:
            row = await conn.fetchrow(
                """
                INSERT INTO messages (session_id, project, machine, role, content, model, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, COALESCE($7, now()))
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id, created_at
                """,
                msg.session_id, msg.project, msg.machine, msg.role,
                msg.content, msg.model, msg.created_at,
            )
        if row is not None and chunks:
            await conn.executemany(
                """
                INSERT INTO message_chunks (message_id, ord, start_char, content, embedding)
                VALUES ($1, $2, $3, $4, $5::vector)
                """,
                [(row["id"], i, s, c, e)
                 for i, ((s, c), e) in enumerate(zip(chunks, chunk_embs))],
            )

    # DO NOTHING returns no row. That is a successful no-op, not a failure - a
    # queue drain that overlaps an earlier one must not look like an error, or
    # the client retries forever.
    if row is None:
        return {"duplicate": True, "embedded": emb is not None}

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
    source: Optional[str] = Query(None),
    hybrid: bool = Query(True),
    rerank_: bool = Query(True, alias="rerank"),
    candidates: int = Query(60, ge=10, le=500),
    drop_self: bool = Query(True),
    recency: bool = Query(True),
    abstain: bool = Query(True),
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

    # Query intent (server/intent.py). An explicit project= always wins over a
    # project resolved from the question text.
    resolved = None
    if PROJECT_RESOLVE and not project:
        known = await _projects_cache.get(app.state.pool)
        resolved = resolve_project(q, known["index"])
        if resolved:
            async with app.state.pool.acquire() as conn:
                if await is_common_name(conn, resolved, known["counts"].get(resolved, 0)):
                    resolved = None
    target_project = project or resolved
    want_recency = RECENCY_INTENT and recency_intent(q)
    automated = (await _automated_cache.get(app.state.pool)
                 if AUTOMATION_DEMOTE else frozenset())

    def helper_filters(start: int) -> tuple:
        """role/source/days as SQL for the project helpers, placeholders from
        $start. The project itself is bound by the helper."""
        c, a = [], []
        for col, val in (("role", role), ("source", source)):
            if val:
                a.append(val); c.append(f"{col} = ${start + len(a) - 1}")
        if days:
            c.append(f"created_at > NOW() - INTERVAL '{int(days)} days'")
        return ("".join(" AND " + x for x in c), a)

    def is_self(r) -> bool:
        # Python mirror of self_pred below.
        body = re.sub(r"^/[a-z-]+\s+", "", (r["content"] or "").strip())
        return body.lower() == q.strip().lower()

    def hit(r) -> dict:
        return {
            "id": r["id"],
            "session_id": r["session_id"],
            "project": r["project"],
            "role": r["role"],
            "source": r["source"],
            "content": r["content"],
            "model": r["model"],
            "created_at": r["created_at"].isoformat(),
            "similarity": round(float(r["score"]), 4),
            **({"matched_chunk": r.get("matched_chunk")} if use_chunks else {}),
        }

    if want_recency and target_project:
        # "When did I last work on X" with X known: the answer is X's newest
        # sessions, in date order. Relevance ranking cannot produce it - the
        # rows that talk most about X are usually other projects mentioning it
        # - and the judge is skipped because a project with history is, by
        # definition, an answerable question.
        use_chunks = False
        _t1 = time.perf_counter()
        extra_sql, extra_args = helper_filters(2)
        async with app.state.pool.acquire() as conn:
            rows = await project_recent(conn, target_project, extra_sql, extra_args,
                                        int(limit) + 1, automated)
        rows = [r for r in rows if not (drop_self and is_self(r))][: int(limit)]
        return {
            "query": q,
            "abstained": False,
            "abstain": "skipped (recency intent)",
            "mode": "recency",
            "intent": "recency",
            "resolved_project": target_project,
            "lex_query": None,
            "rerank": "skipped (recency intent)",
            "timings_ms": {"embed": round(_t_embed*1000),
                           "sql": round((time.perf_counter() - _t1)*1000)},
            "count": len(rows),
            "results": [hit(r) for r in rows],
        }

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
    use_chunks = hybrid and CHUNK_SEARCH and app.state.chunks
    # Pull a deeper pool when re-ranking: the whole point is that the right
    # answer may sit below the RRF cut-off, so handing the model only the
    # top-`limit` rows would ask it to reorder a set the answer is not in.
    sql_limit = max(int(limit), RERANK_POOL) if do_rerank else int(limit)
    if (DEDUPE or AUTOMATION_DEMOTE) and not do_rerank:
        # Collapsing or demoting rows must not shorten the page.
        sql_limit = int(limit) * 2

    # Recency prior, deliberately WEAK. Adjacent RRF ranks differ by about
    # 1/(60+n) - 1/(61+n) ~ 0.00026 near the top, so RECENCY_WEIGHT is sized at
    # roughly three rank-gaps: enough to settle a near-tie in favour of the more
    # recent turn, nowhere near enough to lift an irrelevant new message over a
    # relevant old one. Anything stronger stops being a tie-break and starts
    # being a date sort wearing a relevance sort's clothes.
    #
    # The old `ORDER BY f.score DESC, m.created_at DESC` was effectively dead:
    # RRF sums of two float arms almost never tie exactly, so the second key
    # never fired.
    recency_term = (
        f"{RECENCY_WEIGHT} * exp(- EXTRACT(EPOCH FROM (now() - m.created_at)) "
        f"/ {RECENCY_HALFLIFE_DAYS * 86400.0})"
    ) if recency else "0"

    conds = []
    if project:
        conds.append(f"project = ${idx}")
        params.append(project); idx += 1
    if role:
        conds.append(f"role = ${idx}")
        params.append(role); idx += 1
    if source:
        conds.append(f"source = ${idx}")
        params.append(source); idx += 1
    if days:
        conds.append(f"created_at > NOW() - INTERVAL '{int(days)} days'")

    where_extra = (" AND " + " AND ".join(conds)) if conds else ""

    # Drop the caller's own question. When a hook indexes the user's prompt
    # before the search runs - which is exactly what a live install does - the
    # query matches itself perfectly and takes rank 1 on every single search,
    # burning a slot to hand the user back the words they just typed.
    #
    # Matched on normalised content rather than by session id, so it also
    # catches the same question asked in an earlier session, and needs no
    # session plumbing through the skill. A row that IS the question carries no
    # answer; the value is always in what came after it.
    #
    # Applied AFTER fusion, deliberately, NOT in where_extra. where_extra also
    # constrains pick_lex_query, so a condition there changes which tsquery the
    # lexical arm chooses and perturbs the whole candidate pool. Measured: doing
    # it as a filter cost multi_session 0.113 -> 0.025 Recall@5 while removing
    # nothing it was meant to remove. This is a presentation concern, so it
    # belongs at the presentation end.
    # Snapshot the filter params BEFORE the self-row param is appended. The
    # lexical probe binds only the filters, so handing it a trailing extra
    # argument makes asyncpg reject the statement, pick_lex_query returns None,
    # and RRF silently degrades to pure vector ranking - no error, no log, just
    # worse answers. Measured: that cost multi_session 0.113 -> 0.025 Recall@5.
    n_filter_params = len(params) - 2

    self_pred = ""
    if drop_self:
        self_pred = (
            f"lower(regexp_replace(btrim(m.content), '^/[a-z-]+\\s+', '')) "
            f"IS DISTINCT FROM lower(btrim(${idx}))"
        )
        params.append(q); idx += 1
        sql_limit += 1   # so removing the echo cannot shorten the result set

    if not hybrid:
        sql = f"""
            SELECT id, session_id, project, role, content, model, source, created_at,
                   1 - (embedding <=> $1::vector) AS score
            FROM messages m
            WHERE embedding IS NOT NULL{where_extra}
            {("AND " + self_pred) if self_pred else ""}
            ORDER BY embedding <=> $1::vector
            LIMIT {int(sql_limit)}
        """
    elif use_chunks:
        # Each arm ranks over (short messages UNION chunks of long ones), then
        # collapses to the parent by its best-ranked chunk. From there RRF,
        # recency and the re-ranker run unchanged on parent messages.
        #
        # A long message with no chunk rows (a partial backfill) stays
        # searchable whole, so switching the flag on early loses nothing.
        #
        # Chunks are over-fetched 3x because several can share a parent; the
        # collapsed arm is cut back to `candidates` parents. Cosine distance and
        # ts_rank are each comparable across the two sources (same embedder;
        # ts_rank is unnormalised and chunks are short-message sized), so each
        # arm merges its sources by raw score before ranking.
        whole = (f"(length(content) <= {CHUNK_MIN_CHARS} OR NOT EXISTS "
                 f"(SELECT 1 FROM message_chunks c WHERE c.message_id = messages.id))")
        cand, cand_c = int(candidates), int(candidates) * 3

        def collapse(raw: str, key: str, best: str) -> str:
            # Parent rows of one arm: best entry per parent, ranked by `best`.
            # With CHUNK_CAP > 0, at most CHUNK_CAP parents may come from a
            # chunk, so short/whole messages keep the rest of the pool.
            if CHUNK_CAP <= 0:
                return f"""
                SELECT mid AS id, (array_agg(sc ORDER BY {key}))[1] AS sc,
                       ROW_NUMBER() OVER (ORDER BY {best}) AS rnk
                FROM {raw} GROUP BY mid
                ORDER BY {best} LIMIT {cand}"""
            return f"""
                SELECT id, sc, ROW_NUMBER() OVER (ORDER BY b) AS rnk
                FROM (
                    SELECT id, sc, b, ROW_NUMBER() OVER (
                               PARTITION BY sc IS NULL ORDER BY b) AS src_rn
                    FROM (
                        SELECT mid AS id, (array_agg(sc ORDER BY {key}))[1] AS sc,
                               {best} AS b
                        FROM {raw} GROUP BY mid
                    ) g
                ) h
                WHERE sc IS NULL OR src_rn <= {CHUNK_CAP}
                ORDER BY b LIMIT {cand}"""

        sql = f"""
            WITH vec_raw AS (
                (SELECT id AS mid, embedding <=> $1::vector AS d, NULL::int AS sc
                 FROM messages
                 WHERE embedding IS NOT NULL{where_extra} AND {whole}
                 ORDER BY embedding <=> $1::vector
                 LIMIT {cand})
                UNION ALL
                (SELECT c.message_id, c.embedding <=> $1::vector, c.start_char
                 FROM message_chunks c JOIN messages m ON m.id = c.message_id
                 WHERE c.embedding IS NOT NULL{where_extra}
                 ORDER BY c.embedding <=> $1::vector
                 LIMIT {cand_c})
            ),
            vec AS ({collapse('vec_raw', 'd', 'min(d)')}
            ),
            lex_raw AS (
                (SELECT id AS mid, NULL::int AS sc,
                        ts_rank(to_tsvector('english', content),
                                to_tsquery('english', $2)) AS r
                 FROM messages
                 WHERE to_tsvector('english', content)
                       @@ to_tsquery('english', $2){where_extra} AND {whole}
                 ORDER BY r DESC
                 LIMIT {cand})
                UNION ALL
                (SELECT c.message_id, c.start_char,
                        ts_rank(to_tsvector('english', c.content),
                                to_tsquery('english', $2)) AS r
                 FROM message_chunks c JOIN messages m ON m.id = c.message_id
                 WHERE to_tsvector('english', c.content)
                       @@ to_tsquery('english', $2){where_extra}
                 ORDER BY r DESC
                 LIMIT {cand_c})
            ),
            lex AS ({collapse('lex_raw', 'r DESC', '-max(r)')}
            ),
            fused AS (
                SELECT COALESCE(v.id, l.id) AS id,
                       COALESCE(1.0 / (60 + v.rnk), 0)
                     + COALESCE(1.0 / (60 + l.rnk), 0) AS score,
                       CASE WHEN v.rnk IS NOT NULL AND (l.rnk IS NULL OR v.rnk <= l.rnk)
                            THEN v.sc ELSE l.sc END AS matched_chunk
                FROM vec v
                FULL OUTER JOIN lex l ON v.id = l.id
            )
            SELECT m.id, m.session_id, m.project, m.role, m.content, m.model,
                   m.source, m.created_at, f.score + {recency_term} AS score,
                   f.matched_chunk
            FROM fused f
            JOIN messages m ON m.id = f.id
            {("WHERE " + self_pred) if self_pred else ""}
            ORDER BY score DESC, m.created_at DESC
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
                   m.source, m.created_at, f.score + {recency_term} AS score
            FROM fused f
            JOIN messages m ON m.id = f.id
            {("WHERE " + self_pred) if self_pred else ""}
            ORDER BY score DESC, m.created_at DESC
            LIMIT {int(sql_limit)}
        """

    async with app.state.pool.acquire() as conn:
        # HNSW is an APPROXIMATE index and its default ef_search is 40. Asking
        # it for `candidates` rows while letting it explore only 40 is
        # incoherent, and it does not fail loudly - the graph search simply
        # stops early and the rows it never reached are indistinguishable from
        # rows that do not exist.
        #
        # Measured 2026-09-10 on 116,540 rows, for a question whose answer sits
        # at TRUE rank 16 of a 60-candidate request:
        #
        #   ef_search=40   ->  MISSED entirely
        #   ef_search=100  ->  rank 16
        #   ef_search=400  ->  rank 16
        #
        # This was the single cause of every "the answer is in the corpus and
        # /search cannot find it" failure, and it is invisible from inside: the
        # measurements are stable, repeatable and wrong, which reads exactly
        # like a weak embedding model or a ranking problem. Two of those were
        # hypothesised and neither was the cause.
        #
        # ef_search must exceed the number of candidates requested, with headroom
        # for recall.
        #
        # Plain SET, not SET LOCAL: SET LOCAL only applies inside an explicit
        # transaction, and this block runs in autocommit - it would silently do
        # nothing, which is the same class of quiet failure as the bug itself.
        # The value is session-scoped on a pooled connection, which is harmless
        # here because every /search sets it again from its own `candidates`.
        await conn.execute(
            f"SET hnsw.ef_search = {max(int(candidates) * EF_SEARCH_FACTOR, EF_SEARCH_MIN)}")

        if hybrid:
            # Filter placeholders start at $3 in the main query; the probe only
            # binds them, so renumber them to $2.. for its own statement.
            probe_where = where_extra
            for n in range(2 + n_filter_params, 2, -1):
                probe_where = probe_where.replace(f"${n}", f"${n - 1}")
            lexq = await pick_lex_query(
                conn, q, probe_where, params[2:2 + n_filter_params])
            if lexq is None:
                # No lexical arm at all - RRF degrades to the vector ranking,
                # which is the correct behaviour, not an error.
                lexq = "zzzz_no_lexical_match_zzzz"
            params[1] = lexq
        _t1 = time.perf_counter()
        rows = await conn.fetch(sql, *params)
        _t_sql = time.perf_counter() - _t1
        _t_probe = _t1 - _t0 - _t_embed
        pair_gate_hit = (
            await pair_uncovered(conn, q, rows)
            if (abstain and ABSTAIN_ENABLED and PAIR_GATE) else False)
        # A resolved project's own best matches, which the corpus-wide pool
        # often holds none of. Fetched here but merged only AFTER the abstain
        # decision below, so the pair gate and the judge see the same rows
        # they always have.
        extra_rows = []
        if resolved:
            extra_sql, extra_args = helper_filters(4)
            extra_rows = await project_rows(
                conn, qvec, params[1] if hybrid else "zzzz_no_lexical_match_zzzz",
                resolved, extra_sql, extra_args)

    # Abstain BEFORE re-ranking, deliberately. Measured 2026-09-10 on four
    # subjects SQL-verified absent from the index: judging the RRF order
    # abstained on 3 of 4, judging the re-ranked order abstained on 1 of 4.
    #
    # The two components pull against each other. The re-ranker's entire job is
    # to lift the most plausible-looking rows to the top, and a plausible-looking
    # near-miss is exactly what convinces a judge that the subject is present.
    # Re-ranking first therefore hands the judge the strongest possible case for
    # answering, on precisely the queries where it should refuse.
    abstained, abstain_note = False, "off"
    _t_abstain = 0.0
    if pair_gate_hit:
        # Decided without the judge, so this also saves the judge call.
        abstained, abstain_note = True, "pair gate: no passage mentions both"
    elif abstain and ABSTAIN_ENABLED:
        _t3 = time.perf_counter()
        abstained, abstain_note = await judge_abstain(
            app.state.http, q, list(rows[:ABSTAIN_ROWS]))
        _t_abstain = time.perf_counter() - _t3

    rank_note = "off"
    _t_rank = 0.0
    if abstained:
        rows = []
        rank_note = "skipped (abstained)"
    else:
        if extra_rows:
            rows = list(rows)
            have = {r["id"] for r in rows}
            for r in extra_rows:   # a row found by both arms arrives twice
                if r["id"] not in have and not (drop_self and is_self(r)):
                    have.add(r["id"])
                    rows.append(r)
        if do_rerank:
            _t2 = time.perf_counter()
            rows, rank_note = await rerank(app.state.http, q, list(rows), int(limit))
            _t_rank = time.perf_counter() - _t2
        if DEDUPE:
            rows = collapse_near_duplicates(rows, DEDUPE_CHARS)
        if resolved or AUTOMATION_DEMOTE:
            rows = reorder(
                rows, boost_project=resolved, lift=PROJECT_LIFT,
                demote=(lambda r: is_automated(r, automated)) if AUTOMATION_DEMOTE else None)
        rows = rows[: int(limit)]
        if want_recency:
            # No project to answer from: keep the relevant page, newest first.
            rows = sorted(rows, key=lambda r: r["created_at"], reverse=True)

    return {
        "query": q,
        "abstained": abstained,
        "abstain": abstain_note,
        "mode": ("hybrid+chunks" if use_chunks else "hybrid") if hybrid else "vector",
        "intent": "recency" if want_recency else "relevance",
        "resolved_project": target_project,
        "lex_query": (params[1] if hybrid else None),
        "rerank": rank_note,
        "timings_ms": {"embed": round(_t_embed*1000), "probe": round(_t_probe*1000),
                       "sql": round(_t_sql*1000), "rerank": round(_t_rank*1000),
                       "abstain": round(_t_abstain*1000)},
        "count": len(rows),
        "results": [hit(r) for r in rows],
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
