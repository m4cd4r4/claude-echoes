#!/usr/bin/env python3
"""
Measure retrieval quality without any LLM answering or judging.

For each question, checks whether the correct answer appears in the
top-k retrieved messages. This is a fast, free proxy for retrieval
quality that lets you iterate on retrieval changes without API calls.

Usage:
    # Measure existing results file
    python measure_retrieval.py results/sonnet_reranked.jsonl data/longmemeval_s_cleaned.json

    # Run fresh retrieval and measure (no answering step)
    python measure_retrieval.py --run --hybrid-search --temporal --rerank \
        --embeddings cache/s_embeddings.npz data/longmemeval_s_cleaned.json
"""
import json
import sys
import re
from collections import Counter


def normalize(text: str) -> str:
    """Normalize text for fuzzy matching."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def answer_in_retrieved(answer: str, retrieved: list[dict], threshold: float = 0.5) -> bool:
    """
    Check if the answer (or enough of it) appears in retrieved content.

    For short answers (single word/number), requires exact match.
    For longer answers, checks if enough key terms appear.
    """
    ans = normalize(str(answer))
    retrieved_text = normalize(' '.join(h.get('content', '') for h in retrieved))

    # Short answer (number, single word, name): exact substring match
    if len(ans) < 30:
        return ans in retrieved_text

    # Longer answer: check if enough key terms appear
    # Extract significant words (>3 chars, not stop words)
    stop_words = {'the', 'and', 'was', 'that', 'with', 'for', 'are', 'but',
                  'not', 'you', 'all', 'can', 'had', 'her', 'one', 'our',
                  'this', 'from', 'they', 'been', 'have', 'said', 'each',
                  'which', 'their', 'would', 'about', 'could', 'other', 'were',
                  'more', 'some', 'than', 'them', 'very', 'when', 'what',
                  'your', 'also', 'into', 'just', 'like', 'make', 'many'}
    key_terms = [w for w in ans.split() if len(w) > 3 and w not in stop_words]
    if not key_terms:
        return ans in retrieved_text

    hits = sum(1 for t in key_terms if t in retrieved_text)
    return hits / len(key_terms) >= threshold


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results_or_dataset",
                    help="JSONL results file (with 'retrieved' field) or dataset JSON")
    ap.add_argument("ref_file", nargs="?",
                    help="Reference dataset JSON (required if first arg is results)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Fraction of key terms that must appear (default 0.5)")
    args = ap.parse_args()

    # Load references
    ref_path = args.ref_file or args.results_or_dataset
    refs = json.load(open(ref_path, encoding="utf-8"))
    qid2ref = {r['question_id']: r for r in refs}
    qid2type = {r['question_id']: r['question_type'] for r in refs}

    # Load results (must have 'retrieved' field)
    results_path = args.results_or_dataset
    if results_path.endswith('.jsonl'):
        results = [json.loads(l) for l in open(results_path, encoding='utf-8')]
    else:
        sys.exit("First argument must be a .jsonl results file with 'retrieved' field")

    # Measure
    type_hits = Counter()
    type_total = Counter()
    missed = []

    for r in results:
        qid = r['question_id']
        ref = qid2ref.get(qid)
        if not ref:
            continue
        qtype = qid2type[qid]
        retrieved = r.get('retrieved', [])
        answer = str(ref.get('answer', ''))

        hit = answer_in_retrieved(answer, retrieved, args.threshold)
        type_total[qtype] += 1
        if hit:
            type_hits[qtype] += 1
        else:
            missed.append((qid, qtype, answer[:80], len(retrieved)))

    # Report
    total_hit = sum(type_hits.values())
    total = sum(type_total.values())
    print(f"\nRetrieval hit rate: {total_hit}/{total} ({100*total_hit/total:.1f}%)")
    print(f"(answer found in top-k retrieved content)")
    print()
    print("By category:")
    for qtype in sorted(type_total.keys()):
        h = type_hits[qtype]
        t = type_total[qtype]
        print(f"  {qtype}: {h}/{t} ({100*h/t:.1f}%)")

    print(f"\nRetrieval misses: {len(missed)}")
    if missed:
        print("\nSample misses (first 10):")
        for qid, qtype, ans, n_retrieved in missed[:10]:
            print(f"  [{qtype}] answer='{ans}' (retrieved {n_retrieved} hits)")


if __name__ == "__main__":
    main()
