#!/usr/bin/env python3
"""Index the CURATED memory layers alongside the verbatim chat corpus.

Why this exists
---------------
claude-echoes indexed 108,570 chat messages and none of the three ledgers that
sit beside them. Measured 2026-08-29: asking "what did we decide about the ERF
hero video" returned the right conversations and missed the logged decision,
because the logged decision was never in the index.

The ledgers are the highest signal-per-row material in the whole setup. They are
hand-written, dated, and - unlike chat - they say what SUPERSEDED what. 759
entries averaging 1,403 characters, against 108,570 chat messages: about 0.7% of
the rows, carrying a large share of the actual conclusions.

Each ledger keeps its own `source` value so a result can be attributed, and its
own real timestamp so the temporal filters keep working.

Idempotent: content_hash is a unique index, so re-running only adds new entries.

usage: python scripts/ingest_ledgers.py [--dry-run]
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

DSN = os.environ.get(
    "ECHOES_DB_DSN",
    "postgresql://echoes:echoes@127.0.0.1:5434/echoes",  # secrets-guard: allow
)
OLLAMA = os.environ.get("ECHOES_OLLAMA_URL", "http://127.0.0.1:11435")
MODEL = os.environ.get("ECHOES_OLLAMA_MODEL", "nomic-embed-text")
STATE = Path.home() / ".claude" / "state"
MAX_CHARS = 8000
BATCH = 64


def embed(texts):
    """Batch embed. Halves on failure so one oversized row cannot kill a run."""
    if not texts:
        return []
    payload = json.dumps({"model": MODEL, "input": texts, "keep_alive": -1}).encode()
    req = urllib.request.Request(
        OLLAMA + "/api/embed", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.load(r).get("embeddings")
        if out and len(out) == len(texts):
            return out
        raise ValueError("length mismatch")
    except Exception:
        if len(texts) == 1:
            return [None]
        mid = len(texts) // 2
        return embed(texts[:mid]) + embed(texts[mid:])


def iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def read(name):
    p = STATE / name
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def build():
    """Flatten the three ledgers into (session_id, project, role, content, ts, source).

    role is 'note' rather than user/assistant: these are neither, and mislabelling
    them would corrupt the --role filter that /recall offers over real turns.
    """
    docs = []

    for r in read("changes.jsonl"):
        ts = iso(r.get("ts")) or iso(r.get("when"))
        if not ts:
            continue
        head = f"CHANGE LOG [{r.get('where', '?')}] {r.get('project', '?')}"
        body = r.get("what", "")
        ref = r.get("ref")
        docs.append((
            "ledger:changes", r.get("project") or "unknown", "note",
            head + "\n\n" + body + (f"\n\nref: {ref}" if ref else ""),
            ts, "change-log",
        ))

    for r in read("client-decisions.jsonl"):
        ts = iso(r.get("ts"))
        if not ts:
            continue
        head = f"CLIENT DECISION [{r.get('client', '?')}] said by {r.get('who', '?')}"
        parts = [r.get("what", "")]
        if r.get("why"):
            parts.append("Why: " + r["why"])
        if r.get("source"):
            parts.append("Source: " + str(r["source"]))
        if r.get("ref"):
            parts.append("ref: " + str(r["ref"]))
        docs.append((
            "ledger:client-decisions", r.get("client") or "unknown", "note",
            head + "\n\n" + "\n\n".join(p for p in parts if p),
            ts, "client-log",
        ))

    for r in read("infra-changelog.jsonl"):
        ts = iso(r.get("ts"))
        if not ts:
            continue
        docs.append((
            "ledger:infra", "infrastructure", "note",
            f"INFRASTRUCTURE [{r.get('kind', '?')}]\n\n{r.get('text', '')}",
            ts, "infra-log",
        ))

    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    docs = build()
    by_source = {}
    for d in docs:
        by_source[d[5]] = by_source.get(d[5], 0) + 1
    print("built:", ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())), f"total={len(docs)}")
    if args.dry_run:
        for d in docs[:2]:
            print("---", d[5], d[4].date(), "|", d[3][:160].replace("\n", " "))
        return 0

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor()

    # Skip before embedding, not after. Embedding is the expensive half, and a
    # re-run should cost seconds rather than re-embedding everything to have the
    # unique index reject it.
    cur.execute("SELECT session_id, created_at, left(content, 200) FROM messages WHERE source <> 'chat'")
    seen = {(a, b.astimezone(timezone.utc), c) for a, b, c in cur.fetchall()}
    todo = [d for d in docs if (d[0], d[4], d[3][:200]) not in seen]
    print(f"already indexed: {len(docs) - len(todo)}   to embed: {len(todo)}")
    if not todo:
        print("nothing to do")
        return 0

    added = 0
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        vecs = embed([d[3][:MAX_CHARS] for d in chunk])
        rows = []
        for d, v in zip(chunk, vecs):
            if v is None:
                continue
            rows.append((d[0], d[1], d[2], d[3], d[4], d[5],
                         "[" + ",".join(f"{x:.6f}" for x in v) + "]"))
        if rows:
            got = execute_values(
                cur,
                "INSERT INTO messages (session_id, project, role, content, created_at, source, embedding)"
                " VALUES %s ON CONFLICT (content_hash) DO NOTHING RETURNING 1",
                rows,
                template="(%s,%s,%s,%s,%s,%s,%s::vector)",
                fetch=True, page_size=len(rows),
            )
            added += len(got)
            conn.commit()
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)}  added={added}", flush=True)

    print(f"done: {added} indexed in {time.time() - t0:.1f}s")
    cur.execute("SELECT source, count(*) FROM messages GROUP BY source ORDER BY 2 DESC")
    for s, n in cur.fetchall():
        print(f"  {s:14} {n}")
    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
