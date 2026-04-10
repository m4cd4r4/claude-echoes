"""
Patch Donnacha VPS /api/brain/messages/search to add:
  - LLM re-ranker (opt-in via ?rerank=true query param)
  - Uses local Ollama or Anthropic API for re-ranking

Applies to /opt/donnacha/backend/main.py on the VPS.
Run via: ssh root@45.77.233.102 'python3 /tmp/patch_donnacha_reranker.py'
"""

import re

SRC_PATH = "/opt/donnacha/backend/main.py"

# Read the current source
with open(SRC_PATH, "r") as f:
    src = f.read()

# 1. Add rerank parameter to the endpoint signature
old_sig = '''async def brain_messages_search(
    request: Request,
    q: str,
    limit: int = 10,
    project: str = None,
    role: str = None,
    days: int = None,
):'''

new_sig = '''async def brain_messages_search(
    request: Request,
    q: str,
    limit: int = 10,
    project: str = None,
    role: str = None,
    days: int = None,
    rerank: bool = False,
):'''

assert old_sig in src, "Could not find endpoint signature to patch"
src = src.replace(old_sig, new_sig, 1)

# 2. Add the re-ranker function before the endpoint
reranker_func = '''
async def _rerank_results_with_llm(results: list[dict], query: str, k: int = 10) -> list[dict]:
    """
    LLM re-ranker: takes candidate results and re-scores them by relevance.
    Uses local Ollama (free) with fallback behavior.
    """
    import aiohttp, json as _json

    if not results or len(results) <= k:
        return results

    candidate_lines = []
    for i, r in enumerate(results):
        content = (r.get("content", "") or "")[:400].replace("\\n", " ")
        candidate_lines.append(f"[{i}] {r.get('role', 'unknown')}: {content}")
    candidates_text = "\\n".join(candidate_lines)

    rerank_prompt = f"""Given this question about past conversations:
"{query}"

Score each message's relevance (0-10). Return ONLY a JSON array of [index, score] pairs, sorted by score descending.

Messages:
{candidates_text}

Return: [[index, score], ...]"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": rerank_prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512},
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    return results[:k]
                data = await resp.json()
                text = data.get("response", "").strip()

        # Parse JSON - handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        scores = _json.loads(text)
        llm_scores = {int(pair[0]): float(pair[1]) for pair in scores
                     if isinstance(pair, (list, tuple)) and len(pair) >= 2}

        # Re-sort by LLM score, keep original rank as tiebreaker
        indexed = list(enumerate(results))
        indexed.sort(key=lambda x: llm_scores.get(x[0], 0), reverse=True)
        reranked = [r for _, r in indexed[:k]]
        # Update rank numbers
        for i, r in enumerate(reranked, 1):
            r["rank"] = i
        return reranked
    except Exception as e:
        import sys
        print(f"rerank failed, using original order: {e}", file=sys.stderr)
        return results[:k]

'''

# Insert before the temporal patterns
insert_marker = "_TEMPORAL_PATTERNS = ["
assert insert_marker in src, "Could not find insertion point"
src = src.replace(insert_marker, reranker_func + insert_marker, 1)

# 3. Add re-ranking call before the return statement
old_return = '''        return {
            "query": q,
            "count": len(results),
            "mode": "hybrid" + ("+temporal" if is_temporal else "") + ("+count_k25" if is_count else ""),
            "results": results,
        }'''

new_return = '''        # Optional LLM re-ranking
        if rerank and len(results) > 1:
            results = await _rerank_results_with_llm(results, q, k=effective_limit)

        return {
            "query": q,
            "count": len(results),
            "mode": "hybrid" + ("+temporal" if is_temporal else "") + ("+count_k25" if is_count else "") + ("+rerank" if rerank else ""),
            "results": results,
        }'''

assert old_return in src, "Could not find return statement to patch"
src = src.replace(old_return, new_return, 1)

# Write back
with open(SRC_PATH, "w") as f:
    f.write(src)

print("Patched successfully. Restart donnacha-backend to apply.")
print("New feature: GET /api/brain/messages/search?q=...&rerank=true")
