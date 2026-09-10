#!/usr/bin/env python3
"""
claude-echoes: ingest ~/.claude/chat-history into the echoes database.

WHY THIS EXISTS, given scripts/backfill.py already backfills.

backfill.py parses ~/.claude/daily-logs/*.md - a hand-rolled markdown format.
The chat-logger hook that most installs actually run writes JSONL instead, one
object per message, already carrying exactly the fields the schema wants:

    {"role", "content", "timestamp", "session_id", "project"}

So the corpus is right there, and backfill.py cannot read a line of it.

Three defects in backfill.py that mattered more than the format, fixed here
rather than patched there, because the entry point differs anyway:

1. IT DISCARDS THE TIMESTAMP. backfill.py never sets created_at, so every row
   lands at NOW(). For a memory system whose whole selling point is answering
   "when did we...", importing eight months of history as though it all happened
   this afternoon destroys the one column that makes the answer true. This
   script carries the real timestamp through.

2. IT EMBEDS ONE MESSAGE PER HTTP CALL. Its BATCH_SIZE batches the INSERT, not
   the embedding. Measured on a Quadro RTX 5000, nomic-embed-text: 2051 ms/msg
   one at a time, 58 ms/msg at /api/embed batch=64, 35 ms/msg at batch=256. For
   108k messages that is 62 hours against 1.05 - the difference between "index
   your existing history" being a feature and being a slogan.

3. IT DEDUPES IN PYTHON, pulling every existing row into a set on each run.
   Idempotency now lives in a unique index (sql/002_content_hash.sql), so
   ON CONFLICT DO NOTHING makes a re-run free, and resuming an interrupted run
   is just running it again.

Usage:
    python scripts/ingest_chat_history.py --dry-run
    python scripts/ingest_chat_history.py --since 2026-06-01
    python scripts/ingest_chat_history.py --project solaisoft
    python scripts/ingest_chat_history.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    sys.exit("psycopg2 required: pip install psycopg2-binary")

# Local-only default, identical to docker-compose.yml and backfill.py: the
# container binds to 127.0.0.1 and the credentials are the project's published
# dev pair. Override with ECHOES_DB_DSN for any real deployment.
_DEFAULT_DSN = "postgresql://echoes:echoes@localhost:5434/echoes"  # secrets-guard: allow
DB_DSN = os.environ.get("ECHOES_DB_DSN", _DEFAULT_DSN)
OLLAMA_URL = os.environ.get("ECHOES_OLLAMA_URL", "http://localhost:11435")
OLLAMA_MODEL = os.environ.get("ECHOES_OLLAMA_MODEL", "nomic-embed-text")
DEFAULT_DIR = Path.home() / ".claude" / "chat-history"

# 256 measured fastest on a 16GB card. The ceiling is VRAM, not the API, so a
# smaller card should lower this rather than discover it as an OOM mid-run.
EMBED_BATCH = int(os.environ.get("ECHOES_EMBED_BATCH", "256"))

# nomic-embed-text is a 2048-token model; ~8000 chars is roughly that in English.
# Longer content is truncated for the EMBEDDING ONLY. The full text is still
# stored and still returned by /recall - truncating what the user is shown would
# misrepresent the record.
MAX_EMBED_CHARS = 8000

# A CPU-only container returned HTTP 500 on an 8000-char input (it had ~3.8GiB).
# That is a memory failure, not a context limit, so it is worth degrading rather
# than dying: on failure the batch is retried in halves, down to single messages.
MIN_SPLIT = 1


def parse_ts(raw):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_messages(root: Path, since, project) -> Iterator[dict]:
    for day_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(day_dir.glob("*.jsonl")):
            text = f.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = (o.get("content") or "").strip()
                role = o.get("role")
                if not content or role not in ("user", "assistant"):
                    continue
                ts = parse_ts(o.get("timestamp") or "")
                if ts is None:
                    continue
                if since and ts < since:
                    continue
                proj = o.get("project") or "unknown"
                if project and proj != project:
                    continue
                yield {
                    "session_id": (o.get("session_id") or "unknown")[:255],
                    "project": proj[:255],
                    "role": role,
                    "content": content,
                    "created_at": ts,
                }


def embed_batch(texts):
    payload = {"model": OLLAMA_MODEL,
               "input": [t[:MAX_EMBED_CHARS] for t in texts]}
    req = urllib.request.Request(
        OLLAMA_URL + "/api/embed",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()).get("embeddings")


def embed_resilient(texts):
    """Embed, halving the batch on failure. Returns (index, vector) pairs.

    A whole batch is never silently dropped: anything that fails on its own is
    reported to stderr and skipped, and the caller counts it.
    """
    out = []
    stack = [(0, texts)]
    while stack:
        offset, chunk = stack.pop()
        try:
            vecs = embed_batch(chunk)
            if vecs is None or len(vecs) != len(chunk):
                got = 0 if vecs is None else len(vecs)
                raise ValueError("expected %d vectors, got %d" % (len(chunk), got))
            for i, v in enumerate(vecs):
                out.append((offset + i, v))
        except Exception as e:
            if len(chunk) <= MIN_SPLIT:
                print("  SKIP one message (%d chars): %s" % (len(chunk[0]), e),
                      file=sys.stderr, flush=True)
                continue
            mid = len(chunk) // 2
            stack.append((offset + mid, chunk[mid:]))
            stack.append((offset, chunk[:mid]))
    return out


def vec_literal(v):
    return "[" + ",".join("%.6f" % x for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--since", help="ISO date, e.g. 2026-06-01")
    ap.add_argument("--project", help="only this project")
    ap.add_argument("--limit", type=int, help="stop after N messages")
    args = ap.parse_args()

    root = Path(args.log_dir)
    if not root.exists():
        sys.exit("log dir not found: %s" % root)

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    print("reading %s%s%s" % (
        root,
        ("  since=" + args.since) if args.since else "",
        ("  project=" + args.project) if args.project else ""))

    msgs = []
    for m in read_messages(root, since, args.project):
        msgs.append(m)
        if args.limit and len(msgs) >= args.limit:
            break

    if not msgs:
        sys.exit("no messages matched - nothing to do")

    lo = min(m["created_at"] for m in msgs)
    hi = max(m["created_at"] for m in msgs)
    chars = sum(len(m["content"]) for m in msgs)
    print("parsed {:,} messages  {} to {}  {:.1f}M chars".format(
        len(msgs), lo.date(), hi.date(), chars / 1e6))

    if args.dry_run:
        for m in msgs[:5]:
            print("  %s  %-18s %-9s %r" % (
                m["created_at"].strftime("%Y-%m-%d %H:%M"),
                m["project"][:18], m["role"], m["content"][:60]))
        print("dry run - nothing inserted")
        return

    def connect():
        c = psycopg2.connect(DB_DSN)
        c.autocommit = False
        return c, c.cursor()

    conn, cur = connect()

    # Skip work already done BEFORE embedding it, not after.
    #
    # ON CONFLICT makes a re-run correct, but not cheap: without this, a resumed
    # run re-embeds every message it already has and throws the vectors away at
    # the INSERT. Resuming a 108k backfill at 90% would burn ~45 minutes of GPU
    # to insert nothing.
    #
    # The key is (session_id, role, created_at) rather than the content_hash the
    # database generates. Recomputing that hash here would mean reproducing
    # Postgres's exact rendering of extract(epoch ...)::text in Python, and a
    # near-miss on the float formatting fails open - every row looks new, and
    # the optimisation silently stops working. This triple is already unique in
    # practice and cheap to compare; content_hash stays the real guarantee.
    cur.execute("SELECT session_id, role, created_at FROM messages")
    seen = {(r[0], r[1], r[2]) for r in cur.fetchall()}
    if seen:
        before = len(msgs)
        msgs = [m for m in msgs
                if (m["session_id"], m["role"], m["created_at"]) not in seen]
        print("skipping {:,} already ingested; {:,} to do".format(
            before - len(msgs), len(msgs)))
        if not msgs:
            print("nothing new - up to date")
            cur.close(); conn.close()
            return

    t0 = time.time()
    inserted = 0
    skipped_dupe = 0
    failed = 0

    for i in range(0, len(msgs), EMBED_BATCH):
        batch = msgs[i:i + EMBED_BATCH]
        pairs = embed_resilient([m["content"] for m in batch])
        failed += len(batch) - len(pairs)

        rows = []
        for idx, vec in pairs:
            b = batch[idx]
            rows.append((b["session_id"], b["project"], None, b["role"],
                         b["content"], None, vec_literal(vec), b["created_at"]))

        if rows:
            # RETURNING + fetch=True, not cur.rowcount. execute_values splits the
            # values list into pages internally, and rowcount reports only the
            # LAST page - so a 512-row load into an empty table reported
            # "new=112 dupe=400". The counters are the only thing telling the
            # operator what happened, so they have to be counted, not inferred.
            #
            # Reconnect-and-retry because this is an hour-long run against a
            # container. A restart of anything else in the compose stack drops
            # the connection, and a run that dies at minute 50 of 60 with
            #   OperationalError: could not receive data from server
            # has thrown away the whole hour. ON CONFLICT DO NOTHING plus the
            # unique index means a retried batch is free, so retrying is always
            # safe - there is no partial-write to reason about.
            for attempt in range(3):
                try:
                    returned = execute_values(
                        cur,
                        "INSERT INTO messages "
                        "(session_id, project, machine, role, content, model, "
                        " embedding, created_at) VALUES %s "
                        "ON CONFLICT (content_hash) DO NOTHING RETURNING 1",
                        rows,
                        template="(%s,%s,%s,%s,%s,%s,%s::vector,%s)",
                        page_size=len(rows),
                        fetch=True,
                    )
                    n = len(returned)
                    inserted += n
                    skipped_dupe += len(rows) - n
                    conn.commit()
                    break
                except psycopg2.OperationalError as e:
                    if attempt == 2:
                        raise
                    print("  DB connection lost (%s) - reconnecting, attempt %d/3"
                          % (str(e).strip().splitlines()[0], attempt + 2),
                          file=sys.stderr, flush=True)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    time.sleep(3 * (attempt + 1))
                    conn, cur = connect()

        done = min(i + EMBED_BATCH, len(msgs))
        el = max(time.time() - t0, 0.001)
        eta = (len(msgs) - done) / (done / el) / 60
        print("  {:>7,}/{:,}  new={:,} dupe={:,} fail={}  {:.0f}/s  eta {:.0f}m".format(
            done, len(msgs), inserted, skipped_dupe, failed, done / el, eta),
            flush=True)

    cur.close()
    conn.close()
    print("done in {:.1f}m  inserted={:,}  already-present={:,}  failed={}".format(
        (time.time() - t0) / 60, inserted, skipped_dupe, failed))
    if failed:
        print("WARNING: %d message(s) could not be embedded and were NOT stored. "
              "Re-run to retry them." % failed, file=sys.stderr)


if __name__ == "__main__":
    main()
