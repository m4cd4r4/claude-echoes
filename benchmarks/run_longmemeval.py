#!/usr/bin/env python3
"""
claude-echoes on LongMemEval
============================

Runs the LongMemEval benchmark using the same retrieval approach as the
claude-echoes live server (pgvector + nomic-embed-text).

For auditability this script is self-contained: no imports from server/app.py,
but it uses the *exact same embedding model, distance metric, and ranking*:

    - model: nomic-embed-text via local Ollama (768-dim)
    - distance: cosine (same as pgvector <=> operator)
    - ranking: top-k by ascending cosine distance

If you want to verify we're not cheating, the retrieval is ~30 lines of numpy.
Scroll to class EchoesRetriever and read it.

Usage:
    # Full pipeline (ingest -> retrieve+answer -> save):
    python run_longmemeval.py --dataset data/longmemeval_s_cleaned.json \
        --answer-model claude-haiku-4-5-20251001 --out results/haiku.jsonl

    # Just embed (Phase 2), save embeddings for reuse:
    python run_longmemeval.py --dataset data/longmemeval_s_cleaned.json --embed-only

    # Use pre-computed embeddings from a previous run:
    python run_longmemeval.py --dataset data/longmemeval_s_cleaned.json \
        --embeddings cache/embeddings.npz \
        --answer-model claude-haiku-4-5-20251001 --out results/haiku.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re as _re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter

# ============================================================
# Config
# ============================================================

OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL   = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM     = 768
MAX_CONTENT_CHARS = 8000
EMBED_WORKERS = int(os.environ.get("EMBED_WORKERS", "64"))
ANSWER_WORKERS = int(os.environ.get("ANSWER_WORKERS", "8"))
TOP_K         = int(os.environ.get("TOP_K", "10"))
CHECKPOINT_EVERY = 10_000   # save embeddings every N completed turns

# ============================================================
# HTTP session with connection pooling (avoids Windows port exhaustion)
# ============================================================

_session: Optional[requests.Session] = None
def get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        # One adapter, large pool, no retries here (we handle them above)
        adapter = HTTPAdapter(
            pool_connections=EMBED_WORKERS,
            pool_maxsize=EMBED_WORKERS * 2,
            max_retries=0,
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _session = s
    return _session

# ============================================================
# Embedding (local Ollama)
# ============================================================

def embed_one(text: str, attempts: int = 3) -> Optional[np.ndarray]:
    """
    Return a 768-d float32 vector, or None on failure.
    Uses pooled HTTP connections via requests.Session to avoid Windows
    ephemeral-port exhaustion under high concurrency.
    """
    s = get_session()
    delay = 0.5
    for i in range(attempts):
        try:
            r = s.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": (text or "")[:MAX_CONTENT_CHARS]},
                timeout=60,
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            v = r.json().get("embedding")
            if not v or len(v) != EMBED_DIM:
                raise RuntimeError(f"bad embedding: len={len(v) if v else 0}")
            return np.array(v, dtype=np.float32)
        except Exception as e:
            if i == attempts - 1:
                print(f"  embed failed after {attempts} attempts: {e}",
                      file=sys.stderr, flush=True)
                return None
            time.sleep(delay)
            delay *= 2
    return None

def embed_many(texts: list[str], workers: int = EMBED_WORKERS,
               checkpoint_path: Optional[Path] = None,
               existing: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Embed a list of texts concurrently. Returns (N, 768) float32 array.

    - Failed embeddings are filled with zeros (re-runnable via fix_zero_rows.py).
    - If checkpoint_path is given, saves progress every CHECKPOINT_EVERY rows.
    - If existing is given (resume), only embeds rows that are still zero.
    """
    if existing is not None:
        assert existing.shape == (len(texts), EMBED_DIM), \
            f"existing shape {existing.shape} != ({len(texts)}, {EMBED_DIM})"
        out = existing.copy()
        # Find rows that are still zero (failed or never attempted)
        norms = np.linalg.norm(out, axis=1)
        todo_indices = np.where(norms == 0)[0].tolist()
        print(f"  resume mode: {len(todo_indices)} rows still need embedding "
              f"(out of {len(texts)})")
    else:
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        todo_indices = list(range(len(texts)))

    if not todo_indices:
        return out

    t0 = time.time()
    done = 0
    last_checkpoint = 0

    def save_checkpoint():
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, emb=out)
            # Windows: must remove target before rename if it exists
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                except Exception:
                    pass
            tmp.replace(checkpoint_path)
            print(f"  checkpoint saved: {checkpoint_path}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(embed_one, texts[i]): i for i in todo_indices}
        for fut in as_completed(futures):
            i = futures[fut]
            v = fut.result()
            # Strict shape check - prevents the (0,) -> (768,) crash
            if v is not None and v.shape == (EMBED_DIM,):
                out[i] = v
            done += 1
            if done % 500 == 0 or done == len(todo_indices):
                rate = done / max(time.time() - t0, 0.001)
                eta  = (len(todo_indices) - done) / max(rate, 0.001)
                print(f"  embedded {done}/{len(todo_indices)}  {rate:.1f}/s  "
                      f"eta={eta/60:.1f}min", flush=True)
            if done - last_checkpoint >= CHECKPOINT_EVERY:
                save_checkpoint()
                last_checkpoint = done

    save_checkpoint()
    return out

# ============================================================
# Retriever: mirrors server/app.py /search logic exactly
# ============================================================

@dataclass
class Turn:
    question_id: str      # the benchmark question this turn belongs to
    session_id:  str
    turn_idx:    int
    role:        str      # "user" or "assistant"
    content:     str
    timestamp:   Optional[str]

def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25: lowercase, split on non-alphanumeric,
    drop very short and very long tokens. No stemming — keeps it auditable.
    """
    text = (text or "").lower()
    toks = _re.findall(r"[a-z0-9]+", text)
    return [t for t in toks if 2 <= len(t) <= 30]


class EchoesRetriever:
    """
    Per-question retrieval.

    Two indices, both per-question-id for O(1) filtering:
      - cosine: normalized 768-d nomic embeddings, dot product
      - bm25:   classic bag-of-words lexical index (built lazily)

    The cosine index matches the live server (`ORDER BY embedding <=> $1::vector`).
    BM25 is only used when hybrid search is explicitly enabled, for benchmarking
    experiments. The production server is cosine-only for simplicity.
    """
    def __init__(self, turns: list[Turn], embeddings: np.ndarray):
        assert len(turns) == embeddings.shape[0], \
            f"turns/embeddings mismatch: {len(turns)} vs {embeddings.shape[0]}"
        self.turns = turns
        # Normalize rows for cosine similarity. Zero rows stay zero (failed embeds).
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.emb = (embeddings / norms).astype(np.float32)
        # Index per question_id for O(1) filtering
        self.by_qid: dict[str, np.ndarray] = {}
        for i, t in enumerate(turns):
            self.by_qid.setdefault(t.question_id, []).append(i)
        for qid in self.by_qid:
            self.by_qid[qid] = np.array(self.by_qid[qid], dtype=np.int64)
        # Lazy BM25 cache, keyed by qid. Only built on first hybrid query for that qid.
        self._bm25_cache: dict[str, tuple] = {}

    def _get_bm25(self, qid: str):
        """Build and cache a BM25 index for a single question's haystack."""
        if qid in self._bm25_cache:
            return self._bm25_cache[qid]
        from rank_bm25 import BM25Okapi
        idxs = self.by_qid[qid]
        corpus_toks = [_tokenize(self.turns[int(i)].content) for i in idxs]
        bm25 = BM25Okapi(corpus_toks)
        self._bm25_cache[qid] = (bm25, corpus_toks)
        return self._bm25_cache[qid]

    def search_hybrid(self, qid: str, query_text: str, query_vec: np.ndarray,
                      k: int = TOP_K, wide_k: int = 30,
                      rrf_k: int = 60) -> list[tuple[float, Turn]]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion (RRF) of:
          1. cosine similarity on nomic embeddings (semantic)
          2. BM25 lexical scores (keyword/rare-term match)

        RRF formula: score(doc) = sum over rankers of 1 / (rrf_k + rank)
        where rank is 1-indexed. rrf_k=60 is the standard value from
        the Cormack et al. 2009 RRF paper.

        Both rankers contribute their top-`wide_k` candidates. The union
        is scored with RRF and the final top-k is returned.
        """
        idxs = self.by_qid.get(qid)
        if idxs is None or idxs.size == 0:
            return []

        # --- Rank 1: cosine similarity ---
        q = query_vec / max(np.linalg.norm(query_vec), 1e-9)
        sims = self.emb[idxs] @ q
        wk = min(wide_k, sims.size)
        cos_top = np.argpartition(-sims, wk - 1)[:wk]
        cos_top = cos_top[np.argsort(-sims[cos_top])]   # sorted best-first
        cos_ranks = {int(local_idx): rank + 1 for rank, local_idx in enumerate(cos_top)}

        # --- Rank 2: BM25 ---
        bm25, _ = self._get_bm25(qid)
        query_toks = _tokenize(query_text)
        if query_toks:
            bm25_scores = bm25.get_scores(query_toks)
            bm25_top = np.argpartition(-bm25_scores, wk - 1)[:wk]
            bm25_top = bm25_top[np.argsort(-bm25_scores[bm25_top])]
            bm25_ranks = {int(local_idx): rank + 1 for rank, local_idx in enumerate(bm25_top)}
        else:
            bm25_ranks = {}

        # --- RRF fusion ---
        all_candidates = set(cos_ranks) | set(bm25_ranks)
        fused: list[tuple[float, int]] = []
        for local_idx in all_candidates:
            score = 0.0
            if local_idx in cos_ranks:
                score += 1.0 / (rrf_k + cos_ranks[local_idx])
            if local_idx in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[local_idx])
            fused.append((score, local_idx))
        fused.sort(key=lambda x: -x[0])

        # Return top-k as (score, Turn) tuples
        out = []
        for score, local_idx in fused[:k]:
            out.append((float(score), self.turns[int(idxs[local_idx])]))
        return out

    def search_hybrid_temporal(self, qid: str, query_text: str,
                               query_vec: np.ndarray, question_date: str,
                               k: int = TOP_K, wide_k: int = 50,
                               rrf_k: int = 60,
                               temporal_weight: float = 0.15) -> list[tuple[float, Turn]]:
        """
        Hybrid RRF + temporal re-ranking + session diversity.
        Combines the strengths of both approaches:
          - BM25 catches rare-term queries (preference category)
          - Cosine catches semantic paraphrase
          - Temporal weight biases toward recency
          - Session diversity enables cross-session reasoning
        """
        idxs = self.by_qid.get(qid)
        if idxs is None or idxs.size == 0:
            return []

        # Get wide_k candidates from each ranker
        q = query_vec / max(np.linalg.norm(query_vec), 1e-9)
        sims = self.emb[idxs] @ q
        wk = min(wide_k, sims.size)
        cos_top = np.argpartition(-sims, wk - 1)[:wk]
        cos_top = cos_top[np.argsort(-sims[cos_top])]
        cos_ranks = {int(i): rank + 1 for rank, i in enumerate(cos_top)}

        bm25, _ = self._get_bm25(qid)
        query_toks = _tokenize(query_text)
        if query_toks:
            bm25_scores = bm25.get_scores(query_toks)
            bm25_top = np.argpartition(-bm25_scores, wk - 1)[:wk]
            bm25_top = bm25_top[np.argsort(-bm25_scores[bm25_top])]
            bm25_ranks = {int(i): rank + 1 for rank, i in enumerate(bm25_top)}
        else:
            bm25_ranks = {}

        # RRF fusion
        all_candidates = list(set(cos_ranks) | set(bm25_ranks))
        rrf_scores = np.zeros(len(all_candidates), dtype=np.float32)
        for j, local_idx in enumerate(all_candidates):
            if local_idx in cos_ranks:
                rrf_scores[j] += 1.0 / (rrf_k + cos_ranks[local_idx])
            if local_idx in bm25_ranks:
                rrf_scores[j] += 1.0 / (rrf_k + bm25_ranks[local_idx])

        # Temporal scoring on the fused candidates
        q_ts = _parse_date(question_date)
        t_scores = np.zeros(len(all_candidates), dtype=np.float32)
        for j, local_idx in enumerate(all_candidates):
            turn = self.turns[int(idxs[local_idx])]
            t_ts = _parse_date(turn.timestamp)
            if q_ts is not None and t_ts is not None:
                delta = max((q_ts - t_ts).days, 0)
                t_scores[j] = max(1.0 - delta / 365.0, 0.0)

        # Normalize RRF to [0, 1]
        rrf_min, rrf_max = rrf_scores.min(), rrf_scores.max()
        if rrf_max > rrf_min:
            rrf_norm = (rrf_scores - rrf_min) / (rrf_max - rrf_min)
        else:
            rrf_norm = np.ones_like(rrf_scores)

        blended = (1.0 - temporal_weight) * rrf_norm + temporal_weight * t_scores

        # Session diversity pass
        sorted_order = np.argsort(-blended)
        selected: list[int] = []
        seen_sessions: dict[str, int] = {}
        for order_idx in sorted_order:
            local_idx = all_candidates[order_idx]
            turn = self.turns[int(idxs[local_idx])]
            selected.append(int(order_idx))
            seen_sessions[turn.session_id] = seen_sessions.get(turn.session_id, 0) + 1
            if len(selected) >= k:
                break

        if len(seen_sessions) < 3 and len(sorted_order) > k:
            for order_idx in sorted_order[k:]:
                local_idx = all_candidates[int(order_idx)]
                turn = self.turns[int(idxs[local_idx])]
                if turn.session_id not in seen_sessions:
                    for r in reversed(selected):
                        rturn = self.turns[int(idxs[all_candidates[r]])]
                        if seen_sessions.get(rturn.session_id, 0) > 1:
                            seen_sessions[rturn.session_id] -= 1
                            selected.remove(r)
                            selected.append(int(order_idx))
                            seen_sessions[turn.session_id] = 1
                            break
                    if len(seen_sessions) >= 3:
                        break

        return [(float(blended[i]),
                 self.turns[int(idxs[all_candidates[i]])])
                for i in selected[:k]]

    def search(self, qid: str, query_vec: np.ndarray, k: int = TOP_K) -> list[tuple[float, Turn]]:
        idxs = self.by_qid.get(qid)
        if idxs is None or idxs.size == 0:
            return []
        q = query_vec / max(np.linalg.norm(query_vec), 1e-9)
        sims = self.emb[idxs] @ q        # (n,) cosine similarity in [-1, 1]
        # top-k by descending similarity (equivalent to ascending distance)
        k = min(k, sims.size)
        top = np.argpartition(-sims, k-1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(float(sims[i]), self.turns[int(idxs[i])]) for i in top]

    def search_temporal(self, qid: str, query_vec: np.ndarray,
                        question_date: str, k: int = TOP_K,
                        wide_k: int = 50,
                        temporal_weight: float = 0.15) -> list[tuple[float, Turn]]:
        """
        Temporal-aware retrieval. Three improvements over vanilla search:

        1. Widen: retrieve top-50 by cosine similarity (not just top-10)
        2. Re-rank: blend semantic similarity with temporal proximity score
           - Each hit gets a temporal score based on how recently it occurred
             relative to the question date (recency bias: more recent = higher)
        3. Diversify: ensure hits come from at least 3 different sessions
           when possible, so cross-session temporal reasoning has material

        The temporal_weight (default 0.15) controls how much dates matter
        vs. pure semantic similarity. 0.0 = pure semantic, 1.0 = pure temporal.
        """
        idxs = self.by_qid.get(qid)
        if idxs is None or idxs.size == 0:
            return []
        q = query_vec / max(np.linalg.norm(query_vec), 1e-9)
        sims = self.emb[idxs] @ q

        # Step 1: get wide_k candidates by semantic similarity
        wk = min(wide_k, sims.size)
        wide_top = np.argpartition(-sims, wk - 1)[:wk]

        # Step 2: compute temporal scores for the candidates
        q_ts = _parse_date(question_date)
        t_scores = np.zeros(wk, dtype=np.float32)
        for j, local_idx in enumerate(wide_top):
            turn = self.turns[int(idxs[local_idx])]
            t_ts = _parse_date(turn.timestamp)
            if q_ts is not None and t_ts is not None:
                # Days before question. More recent = higher score.
                # Normalize: 0 days -> 1.0, 365+ days -> ~0.0
                delta = max((q_ts - t_ts).days, 0)
                t_scores[j] = max(1.0 - delta / 365.0, 0.0)

        # Step 3: blend scores
        sem_scores = sims[wide_top]
        # Normalize semantic to [0, 1] range within this candidate set
        sem_min, sem_max = sem_scores.min(), sem_scores.max()
        if sem_max > sem_min:
            sem_norm = (sem_scores - sem_min) / (sem_max - sem_min)
        else:
            sem_norm = np.ones_like(sem_scores)

        blended = (1.0 - temporal_weight) * sem_norm + temporal_weight * t_scores

        # Step 4: session diversity - ensure at least 3 sessions in top-k
        sorted_by_blend = np.argsort(-blended)
        selected = []
        seen_sessions: dict[str, int] = {}
        # First pass: pick top hits while tracking session diversity
        for idx in sorted_by_blend:
            turn = self.turns[int(idxs[wide_top[idx]])]
            sid = turn.session_id
            seen_sessions[sid] = seen_sessions.get(sid, 0) + 1
            selected.append(idx)
            if len(selected) >= k:
                break

        # If we have too few sessions, swap out some redundant hits
        if len(seen_sessions) < 3 and len(sorted_by_blend) > k:
            for idx in sorted_by_blend[k:]:
                turn = self.turns[int(idxs[wide_top[idx]])]
                if turn.session_id not in seen_sessions:
                    # Replace the lowest-scoring redundant hit
                    for r in reversed(selected):
                        rturn = self.turns[int(idxs[wide_top[r]])]
                        if seen_sessions.get(rturn.session_id, 0) > 1:
                            seen_sessions[rturn.session_id] -= 1
                            selected.remove(r)
                            selected.append(idx)
                            seen_sessions[turn.session_id] = 1
                            break
                    if len(seen_sessions) >= 3:
                        break

        results = [(float(blended[i]), self.turns[int(idxs[wide_top[i]])])
                   for i in selected[:k]]
        return results


# Temporal-intent keywords: if any of these patterns appear in a question,
# use temporal-aware retrieval. Deliberately conservative — false negatives
# (missing a temporal question) are better than false positives (breaking
# a non-temporal question with chronological reordering).
_TEMPORAL_PATTERNS = [
    r"\bhow many (days|weeks|months|years)\b",
    r"\bhow long (ago|since|before|after|between)\b",
    r"\bwhen did\b",
    r"\bwhen was\b",
    r"\bbefore or after\b",
    r"\bfirst time\b",
    r"\blast time\b",
    r"\bmost recent\b",
    r"\bhow recently\b",
    r"\bpassed between\b",
    r"\bpassed since\b",
    r"\bago did\b",
    r"\bdays? (passed|between|since|ago)\b",
    r"\bweeks? (passed|between|since|ago)\b",
    r"\bmonths? (passed|between|since|ago)\b",
]
_TEMPORAL_RE = _re.compile("|".join(_TEMPORAL_PATTERNS), _re.IGNORECASE)

def _is_temporal_question(question: str) -> bool:
    """Detect whether a question requires temporal reasoning."""
    return bool(_TEMPORAL_RE.search(question))


def _parse_date(date_str: Optional[str]):
    """Parse a LongMemEval date string like '2023/04/10 (Mon) 23:07' into a date."""
    if not date_str:
        return None
    from datetime import datetime
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d (%a)", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip().split(" (")[0] if "(" in date_str else date_str.strip(), fmt.split(" (")[0] if "(" in fmt else fmt).date()
        except ValueError:
            continue
    # Try just the date part
    try:
        return datetime.strptime(date_str.strip()[:10], "%Y/%m/%d").date()
    except Exception:
        return None

# ============================================================
# Dataset loader
# ============================================================

def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"loaded {len(data)} questions from {path}")
    return data

def flatten_turns(dataset: list[dict]) -> list[Turn]:
    """Convert the LongMemEval nested haystack into a flat list of turns."""
    turns: list[Turn] = []
    for q in dataset:
        qid = q["question_id"]
        sessions = q.get("haystack_sessions", [])
        session_ids = q.get("haystack_session_ids", [])
        dates = q.get("haystack_dates", [])
        for s_idx, session in enumerate(sessions):
            sid = session_ids[s_idx] if s_idx < len(session_ids) else f"{qid}-s{s_idx}"
            date = dates[s_idx] if s_idx < len(dates) else None
            for t_idx, turn in enumerate(session):
                turns.append(Turn(
                    question_id=qid,
                    session_id=sid,
                    turn_idx=t_idx,
                    role=turn.get("role", "user"),
                    content=turn.get("content", ""),
                    timestamp=date,
                ))
    print(f"flattened to {len(turns):,} turns")
    return turns

# ============================================================
# Answer generation
# ============================================================

ANSWER_SYSTEM = """You are answering a question based ONLY on retrieved past chat messages.

You will be given:
  1. A question about things the user discussed in past sessions.
  2. A list of retrieved messages ranked by relevance.

Your job: answer the question using ONLY information in the retrieved messages.

Rules:
- If the retrieved messages contain the answer, give it concisely and directly.
- If the question is about "when" something happened, quote the exact date from the message timestamp if given.
- For counting questions ("how many X"), scan ALL retrieved messages and enumerate every distinct instance. Do not stop at the first match.
- For preference questions ("what would I like", "recommend based on my taste"), infer the user's likely preference from ANY relevant past context - even a single mention of a related topic, hobby, or prior request. Synthesize; do not refuse just because the evidence is indirect.
- Only say "not enough information" if the retrieved messages are genuinely unrelated to the question. If there is ANY topical overlap, attempt an answer.
- Keep answers short. No preamble. No "based on the retrieved messages..." filler.
"""

ANSWER_SYSTEM_TEMPORAL = """You are answering a question based ONLY on retrieved past chat messages.

You will be given:
  1. A question about things the user discussed in past sessions.
  2. A list of retrieved messages sorted in CHRONOLOGICAL ORDER (earliest first), with timestamps.
  3. The date the question was asked.

Your job: answer the question using ONLY information in the retrieved messages.

Rules:
- Messages are presented in chronological order. Use the timestamps to reason about sequence, duration, and recency.
- When the question asks "how many days/weeks/months" between events, compute the answer from the timestamps. Count the calendar days between the dates mentioned or implied.
- When the question asks about "the first" or "the last" occurrence, use chronological order to determine which one.
- If a fact was updated across multiple messages, the most recent version is the current answer.
- If the retrieved messages contain the answer, give it concisely and directly.
- If the retrieved messages do NOT contain enough information to answer, say so explicitly. Do not guess.
- Keep answers short. No preamble.
"""

def format_hits(hits: list[tuple[float, Turn]], chronological: bool = False) -> str:
    if chronological:
        # Sort by timestamp, then by turn_idx within same timestamp.
        # Use session-prefixed format to help LLM reason across sessions.
        sorted_hits = sorted(hits, key=lambda h: (h[1].timestamp or "", h[1].turn_idx))
        lines = []
        for score, t in sorted_hits:
            ts = f"[{t.timestamp}] " if t.timestamp else ""
            sid_short = t.session_id[:8] if t.session_id else ""
            lines.append(f"[session:{sid_short}] {ts}{t.role}: {t.content}")
        return "\n".join(lines)
    else:
        # Vanilla: relevance-ordered with rank numbers (not raw similarity scores,
        # which are tiny RRF values that can mislead the LLM into thinking
        # low-score = low-confidence = "refuse to answer").
        lines = []
        for rank, (score, t) in enumerate(hits, 1):
            ts = f"[{t.timestamp}] " if t.timestamp else ""
            lines.append(f"(#{rank}) {ts}{t.role}: {t.content}")
        return "\n".join(lines)

def answer_with_claude(client, model: str, question: str, hits_text: str,
                       question_date: str, thinking_budget: int = 0,
                       system_prompt: str = ANSWER_SYSTEM,
                       chronological: bool = False) -> str:
    label = "chronological order" if chronological else "ranked by relevance"
    user_msg = (
        f"Question (asked on {question_date}):\n{question}\n\n"
        f"Retrieved past messages ({label}):\n{hits_text}\n\n"
        f"Your answer:"
    )
    kwargs = {
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_msg}],
    }
    if thinking_budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        kwargs["max_tokens"] = thinking_budget + 1024
    r = client.messages.create(**kwargs)
    # Concatenate all text blocks, skip thinking blocks
    parts = []
    for block in r.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()

# ============================================================
# Pipeline
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to longmemeval_*.json")
    ap.add_argument("--embeddings", help="path to cached embeddings .npz (skip embed)")
    ap.add_argument("--cache-embeddings", default="cache/embeddings.npz",
                    help="where to save embeddings after computing")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--answer-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--out", help="output JSONL for answers (required unless --embed-only)")
    ap.add_argument("--limit", type=int, help="only process first N questions (for smoke test)")
    ap.add_argument("--resume", action="store_true", help="skip questions already in --out")
    ap.add_argument("--temporal", action="store_true",
                    help="enable temporal-aware retrieval + chronological prompting")
    ap.add_argument("--temporal-weight", type=float, default=0.15,
                    help="blend weight for temporal score (0=pure semantic, 1=pure temporal)")
    ap.add_argument("--hybrid-search", action="store_true",
                    help="enable BM25 + cosine RRF hybrid retrieval")
    args = ap.parse_args()

    full_dataset = load_dataset(args.dataset)
    turns = flatten_turns(full_dataset)

    # --limit restricts how many questions we ANSWER, not how many we embed.
    # This keeps embedding alignment stable across partial and full runs.
    dataset = full_dataset[:args.limit] if args.limit else full_dataset
    if args.limit:
        print(f"LIMIT: answering first {len(dataset)} of {len(full_dataset)} questions")

    # --- Phase A: embed turns ---
    cache_path = Path(args.cache_embeddings)
    existing = None
    # If a previous (partial) cache exists at the target path, load it for resume
    if args.embeddings and Path(args.embeddings).exists():
        print(f"loading cached embeddings from {args.embeddings}")
        cached = np.load(args.embeddings)
        emb = cached["emb"]
        if emb.shape[0] != len(turns):
            sys.exit(f"embeddings/turn count mismatch: {emb.shape[0]} vs {len(turns)}")
        print(f"  loaded {emb.shape}")
    else:
        if cache_path.exists():
            try:
                with np.load(cache_path) as cached:
                    arr = cached["emb"]
                    if arr.shape == (len(turns), EMBED_DIM):
                        existing = arr.copy()  # force read so file handle closes
                        print(f"found existing checkpoint at {cache_path}, resuming")
            except Exception as e:
                print(f"  could not load existing checkpoint: {e}")
        print(f"embedding {len(turns):,} turns with {EMBED_MODEL} "
              f"({EMBED_WORKERS} workers)...")
        texts = [t.content for t in turns]
        emb = embed_many(texts, workers=EMBED_WORKERS,
                         checkpoint_path=cache_path, existing=existing)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, emb=emb)
        print(f"saved embeddings to {cache_path}")

    if args.embed_only:
        print("embed-only mode: done.")
        return

    retriever = EchoesRetriever(turns, emb)

    # --- Phase B: retrieve + answer ---
    if not args.out:
        sys.exit("--out is required for the full pipeline")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["question_id"])
                except Exception:
                    pass
        print(f"resume: skipping {len(done_ids)} already-answered questions")

    from anthropic import Anthropic
    client = Anthropic()   # picks up ANTHROPIC_API_KEY from env

    todo = [q for q in dataset if q["question_id"] not in done_ids]
    print(f"answering {len(todo)} questions with {args.answer_model} "
          f"(thinking={args.thinking_budget}, top_k={args.top_k}, "
          f"workers={ANSWER_WORKERS})...")

    f_out = open(out_path, "a", encoding="utf-8", buffering=1)  # line-buffered
    t0 = time.time()
    done = 0

    def process_one(q: dict) -> dict:
        qid  = q["question_id"]
        qtxt = q["question"]
        qdate = q.get("question_date", "unknown date")
        qtype = q.get("question_type", "")
        query_vec = embed_one(qtxt)
        if query_vec is None:
            return {"question_id": qid, "hypothesis": "", "error": "query embed failed"}

        # Dynamic top-k: counting questions need broader recall
        is_count = bool(_re.search(r"\bhow many\b|\btotal number\b|\bhow much\b", qtxt, _re.I))
        effective_k = max(args.top_k, 25) if is_count else args.top_k

        use_temporal = args.temporal and _is_temporal_question(qtxt)
        if use_temporal and args.hybrid_search:
            hits = retriever.search_hybrid_temporal(
                qid, qtxt, query_vec, question_date=qdate, k=effective_k,
                temporal_weight=args.temporal_weight)
            use_chrono = True
            system = ANSWER_SYSTEM_TEMPORAL
        elif use_temporal:
            hits = retriever.search_temporal(
                qid, query_vec, question_date=qdate, k=effective_k,
                temporal_weight=args.temporal_weight)
            use_chrono = True
            system = ANSWER_SYSTEM_TEMPORAL
        elif args.hybrid_search:
            hits = retriever.search_hybrid(qid, qtxt, query_vec, k=effective_k)
            use_chrono = False
            system = ANSWER_SYSTEM
        else:
            hits = retriever.search(qid, query_vec, k=effective_k)
            use_chrono = False
            system = ANSWER_SYSTEM

        hits_text = format_hits(hits, chronological=use_chrono)
        try:
            hyp = answer_with_claude(client, args.answer_model, qtxt, hits_text,
                                     qdate, thinking_budget=args.thinking_budget,
                                     system_prompt=system, chronological=use_chrono)
        except Exception as e:
            return {"question_id": qid, "hypothesis": "", "error": str(e)}
        return {
            "question_id": qid,
            "question_type": qtype,
            "hypothesis": hyp,
            "retrieved": [{"score": s, "session_id": t.session_id,
                           "role": t.role, "content": t.content[:500],
                           "timestamp": t.timestamp} for s, t in hits],
        }

    with ThreadPoolExecutor(max_workers=ANSWER_WORKERS) as ex:
        futures = {ex.submit(process_one, q): q for q in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            f_out.write(json.dumps(rec) + "\n")
            done += 1
            if done % 10 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 0.001)
                eta  = (len(todo) - done) / max(rate, 0.001)
                print(f"  {done}/{len(todo)}  rate={rate:.2f}/s  eta={eta/60:.1f}min",
                      flush=True)

    f_out.close()
    print(f"done. wrote {out_path}")

if __name__ == "__main__":
    main()
