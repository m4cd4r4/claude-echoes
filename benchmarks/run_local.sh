#!/usr/bin/env bash
#
# Local benchmark iteration loop. Zero API cost.
#
# Usage:
#   ./run_local.sh                    # Full 500 questions, qwen2.5:7b
#   ./run_local.sh --limit 50         # Quick 50-question smoke test (~2 min)
#   ./run_local.sh --tag my-experiment # Custom output name
#
# The workflow:
#   1. Make a retrieval change in run_longmemeval.py
#   2. Run ./run_local.sh --limit 50
#   3. Compare the score to your baseline
#   4. If delta is positive, run ./run_local.sh (full 500)
#   5. If still positive, run with Sonnet for the real number
#

set -e
cd "$(dirname "$0")"

TAG="${TAG:-local}"
LIMIT=""
MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
EXTRA_ARGS=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="--limit $2"; shift 2;;
        --tag) TAG="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

OUT="results/${TAG}_$(date +%H%M).jsonl"
echo "=== Local benchmark: $TAG ==="
echo "  Answering model: $MODEL (local Ollama)"
echo "  Judge model: $MODEL (local)"
echo "  Output: $OUT"
echo ""

# Step 1: Retrieve + Answer (locally)
echo "--- Phase 1: Retrieve + Answer ---"
python run_longmemeval.py \
    --dataset data/longmemeval_s_cleaned.json \
    --embeddings cache/s_embeddings.npz \
    --ollama-answer --ollama-answer-model "$MODEL" \
    --temporal --hybrid-search --rerank --smart-temporal \
    --top-k 15 \
    --out "$OUT" \
    $LIMIT $EXTRA_ARGS

# Step 2: Judge (locally)
echo ""
echo "--- Phase 2: Judge ---"
python evaluate_local.py "$OUT" data/longmemeval_s_cleaned.json \
    --model "$MODEL" $LIMIT --quiet

echo ""
echo "=== Done. Compare to baseline: ==="
echo "  Previous Haiku+GPT4o-mini: 76.8%"
echo "  Previous Sonnet+GPT4o-mini: 86.4%"
echo "  NOTE: Local scores are lower in absolute terms."
echo "  What matters is the DELTA between local runs."
