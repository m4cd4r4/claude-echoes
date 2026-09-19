#!/usr/bin/env python3
"""
Chunk and embed every long message that has no chunks yet (sql/005_chunks.sql).

Resumable: a message is chunked in one transaction, and messages that already
have chunk rows are skipped, so a killed run picks up where it stopped.
Live writes chunk themselves once the table exists; this covers the history.

Usage:
    python scripts/backfill_chunks.py --limit=1000     # timed sample first
    python scripts/backfill_chunks.py                  # everything left
    python scripts/backfill_chunks.py --concurrency=16

Uses the same chunker as the server (server/chunking.py).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import aiohttp
import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from chunking import CHUNK_MIN_CHARS, split_chunks  # noqa: E402

DB_DSN       = os.environ.get("ECHOES_DB_DSN",
                              "postgresql://echoes:echoes@localhost:5434/echoes")  # secrets-guard: allow - compose's loopback-only default, same as backfill.py
OLLAMA_URL   = os.environ.get("ECHOES_OLLAMA_URL", "http://localhost:11435")
OLLAMA_MODEL = os.environ.get("ECHOES_OLLAMA_MODEL", "nomic-embed-text")
BATCH        = 200


async def embed(http, sem, text: str):
    async with sem:
        for attempt in range(3):
            try:
                async with http.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": OLLAMA_MODEL, "prompt": text, "keep_alive": -1},
                ) as r:
                    if r.status == 200:
                        vec = (await r.json()).get("embedding")
                        if vec:
                            return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            except Exception:
                pass
            await asyncio.sleep(1 + attempt)
        return None


async def do_message(pool, http, sem, mid: int, content: str, stats: dict):
    chunks = split_chunks(content)
    embs = await asyncio.gather(*(embed(http, sem, c) for _, c in chunks))
    async with pool.acquire() as conn, conn.transaction():
        # A live write may have chunked it since the batch was read.
        if await conn.fetchval(
                "SELECT 1 FROM message_chunks WHERE message_id = $1 LIMIT 1", mid):
            return
        await conn.executemany(
            "INSERT INTO message_chunks (message_id, ord, start_char, content, embedding) "
            "VALUES ($1, $2, $3, $4, $5::vector)",
            [(mid, i, s, c, e) for i, ((s, c), e) in enumerate(zip(chunks, embs))],
        )
    stats["messages"] += 1
    stats["chunks"] += len(chunks)
    stats["null_embeddings"] += sum(e is None for e in embs)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N messages (0 = all)")
    ap.add_argument("--concurrency", type=int, default=12, help="concurrent embed calls")
    args = ap.parse_args()

    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=8)
    http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"messages": 0, "chunks": 0, "null_embeddings": 0}

    remaining = await pool.fetchval(
        f"SELECT count(*) FROM messages m WHERE length(content) > {CHUNK_MIN_CHARS} "
        "AND NOT EXISTS (SELECT 1 FROM message_chunks c WHERE c.message_id = m.id)")
    target = min(remaining, args.limit) if args.limit else remaining
    print(f"long messages without chunks: {remaining}; this run: {target}", flush=True)

    t0, last_id = time.perf_counter(), 0
    try:
        while stats["messages"] < target:
            take = min(BATCH, target - stats["messages"])
            rows = await pool.fetch(
                f"SELECT id, content FROM messages m WHERE id > $1 "
                f"AND length(content) > {CHUNK_MIN_CHARS} "
                "AND NOT EXISTS (SELECT 1 FROM message_chunks c WHERE c.message_id = m.id) "
                "ORDER BY id LIMIT $2", last_id, take)
            if not rows:
                break
            last_id = rows[-1]["id"]
            # Messages run concurrently too; the semaphore caps embeds, not messages.
            await asyncio.gather(*(do_message(pool, http, sem, r["id"], r["content"], stats)
                                   for r in rows))
            el = time.perf_counter() - t0
            rate = stats["messages"] / el
            eta = (remaining - stats["messages"]) / rate if rate else 0
            print(f"{stats['messages']}/{target} msgs  {stats['chunks']} chunks  "
                  f"null_emb={stats['null_embeddings']}  {el:.0f}s  "
                  f"{rate:.1f} msg/s  full-remaining ETA {eta/60:.1f} min", flush=True)
    finally:
        await http.close()
        await pool.close()
    el = time.perf_counter() - t0
    print(f"done: {stats} in {el:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
