#!/usr/bin/env python3
"""
claude-echoes backfill

Reads existing Claude Code session logs from ~/.claude/daily-logs/*.md (or a
custom dir) and inserts them into the echoes database with embeddings.

Daily-log format expected:
    ## Session <id> — <project>
    **<timestamp>** — user
    <content>
    **<timestamp>** — assistant
    <content>

If your logs are in a different format, edit the `parse_daily_log` function.
This script is idempotent: messages are keyed by (session_id, role, content)
so re-running is safe.

Usage:
    python scripts/backfill.py                      # read ~/.claude/daily-logs
    python scripts/backfill.py /path/to/logs        # custom directory
    python scripts/backfill.py --dry-run            # parse but don't insert
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterator

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    sys.exit("psycopg2 required: pip install psycopg2-binary")

# --- config ---------------------------------------------------------------

DB_DSN       = os.environ.get("ECHOES_DB_DSN",
                              "postgresql://echoes:echoes@localhost:5434/echoes")
OLLAMA_URL   = os.environ.get("ECHOES_OLLAMA_URL", "http://localhost:11435")
OLLAMA_MODEL = os.environ.get("ECHOES_OLLAMA_MODEL", "nomic-embed-text")
DEFAULT_DIR  = Path.home() / ".claude" / "daily-logs"
BATCH_SIZE   = 32
MAX_CHARS    = 8000

# --- parsing --------------------------------------------------------------

SESSION_RE = re.compile(r"^##\s+Session\s+([^\s]+)\s*(?:—|-)\s*(.+)$", re.M)
ENTRY_RE   = re.compile(
    r"^\*\*([^*]+)\*\*\s*(?:—|-)\s*(user|assistant)\s*\n(.+?)(?=^\*\*|\Z)",
    re.M | re.S,
)

def parse_daily_log(text: str) -> Iterator[dict]:
    """
    Yield message dicts from one daily log file.
    Adjust this function if your log format differs.
    """
    sessions = []
    for m in SESSION_RE.finditer(text):
        sessions.append({
            "session_id": m.group(1).strip(),
            "project": m.group(2).strip(),
            "start": m.start(),
        })
    if not sessions:
        return

    # Slice the file into per-session blocks
    for i, s in enumerate(sessions):
        end = sessions[i+1]["start"] if i+1 < len(sessions) else len(text)
        block = text[s["start"]:end]
        for em in ENTRY_RE.finditer(block):
            content = em.group(3).strip()
            if not content:
                continue
            yield {
                "session_id": s["session_id"],
                "project":    s["project"],
                "role":       em.group(2),
                "content":    content,
            }

# --- embedding ------------------------------------------------------------

def embed(text: str) -> list[float] | None:
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=json.dumps({"model": OLLAMA_MODEL,
                             "prompt": text[:MAX_CHARS]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["embedding"]
    except Exception as e:
        print(f"  embed failed: {e}", flush=True)
        return None

def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

# --- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        sys.exit(f"log dir not found: {log_dir}")

    files = sorted(log_dir.glob("*.md"))
    print(f"found {len(files)} log files in {log_dir}")

    all_messages = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for msg in parse_daily_log(text):
            all_messages.append(msg)

    print(f"parsed {len(all_messages)} messages")

    if args.dry_run:
        for m in all_messages[:5]:
            print(f"  {m['project']:20} {m['role']:9} {m['content'][:60]!r}")
        print("dry run — nothing inserted")
        return

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    # Dedup against existing rows
    cur.execute("SELECT session_id, role, LEFT(content, 200) FROM messages")
    existing = {(r[0], r[1], r[2]) for r in cur.fetchall()}
    new = [m for m in all_messages
           if (m["session_id"], m["role"], m["content"][:200]) not in existing]
    print(f"{len(new)} new messages (skipping {len(all_messages) - len(new)} dupes)")

    t0 = time.time()
    for i in range(0, len(new), BATCH_SIZE):
        batch = new[i:i+BATCH_SIZE]
        rows = []
        for m in batch:
            v = embed(m["content"])
            if v is None:
                continue
            rows.append((
                m["session_id"], m["project"], None, m["role"],
                m["content"], None, vec_literal(v),
            ))
        if rows:
            execute_values(
                cur,
                """INSERT INTO messages
                   (session_id, project, machine, role, content, model, embedding)
                   VALUES %s""",
                rows,
                template="(%s,%s,%s,%s,%s,%s,%s::vector)",
            )
            conn.commit()
        done = min(i + BATCH_SIZE, len(new))
        rate = done / max(time.time() - t0, 0.001)
        print(f"  {done}/{len(new)}  rate={rate:.1f}/s")

    cur.close()
    conn.close()
    print("done.")

if __name__ == "__main__":
    main()
