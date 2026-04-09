#!/usr/bin/env python3
"""
Re-embed any zero-vector rows in a saved embeddings cache.

Ollama occasionally returns HTTP 500 under heavy load. The main ingest script
fills failed rows with zeros. This script finds them and retries with backoff,
so transient failures don't permanently corrupt the benchmark inputs.

Usage:
    python fix_zero_rows.py --dataset data/longmemeval_s_cleaned.json \
                            --embeddings cache/s_embeddings.npz
"""
import argparse
import json
import sys
import time
import urllib.request

import numpy as np

from run_longmemeval import flatten_turns, load_dataset, MAX_CONTENT_CHARS, OLLAMA_URL, EMBED_MODEL

def embed_with_retry(text: str, attempts: int = 5) -> np.ndarray | None:
    delay = 0.5
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/embeddings",
                data=json.dumps({"model": EMBED_MODEL,
                                 "prompt": (text or "")[:MAX_CONTENT_CHARS]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                v = json.loads(r.read())["embedding"]
            return np.array(v, dtype=np.float32)
        except Exception as e:
            if i == attempts - 1:
                print(f"  giving up after {attempts} attempts: {e}", flush=True)
                return None
            time.sleep(delay)
            delay *= 2
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embeddings", required=True)
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    turns = flatten_turns(dataset)

    cached = np.load(args.embeddings)
    emb = cached["emb"]
    if emb.shape[0] != len(turns):
        sys.exit(f"mismatch: {emb.shape[0]} embeds vs {len(turns)} turns")

    norms = np.linalg.norm(emb, axis=1)
    zero_rows = np.where(norms == 0)[0]
    print(f"found {len(zero_rows)} zero-vector rows out of {len(turns)}")

    if len(zero_rows) == 0:
        print("nothing to fix.")
        return

    fixed = 0
    still_zero = 0
    for idx in zero_rows:
        content = turns[int(idx)].content
        if not content.strip():
            # Genuinely empty content — leave as zero, retrieval will skip it
            continue
        v = embed_with_retry(content)
        if v is None:
            still_zero += 1
            continue
        emb[idx] = v
        fixed += 1
        if fixed % 50 == 0:
            print(f"  fixed {fixed}/{len(zero_rows)}", flush=True)

    print(f"fixed {fixed}, still zero {still_zero}")
    np.savez_compressed(args.embeddings, emb=emb)
    print(f"saved updated cache to {args.embeddings}")

if __name__ == "__main__":
    main()
