# The 7-case eval was overstating retrieval by about 2x

Measured 2026-08-30 against the generated case set (`scripts/build_eval_set.mjs`).
No verbatim archive content appears in this file - this repo is public, and the
generated cases stay local by design.

## Headline

| set | cases | Recall@5 |
|---|---|---|
| hand-written (`eval_retrieval.mjs`) | 7 | **0.857** (6/7) |
| generated, single-session | 34 | **0.441** |

The committed 7-case eval was written by the same session that built the
retriever, so it selected for what the retriever could already serve. It is a
smoke test. Every ranking decision taken on it is unsettled.

## Full run, re-ranker OFF (hybrid + RRF only), 67/67 cases, 0 errors

| category | n | Recall@5 | MRR | nDCG@5 | window | ms |
|---|---|---|---|---|---|---|
| single_session | 34 | 0.441 | 0.171 | 0.239 | 0.412 | 478 |
| temporal_include | 10 | 0.600 | 0.253 | 0.341 | 0.600 | 129 |
| temporal_exclude | 10 | **1.000** | 1.000 | 1.000 | 1.000 | 4383 |
| multi_session | 8 | **0.113** | 0.160 | 0.089 | 0.500 | 410 |
| abstention | 5 | **0.000** | 0.000 | 0.000 | 0.000 | 506 |
| OVERALL | 67 | 0.476 | 0.293 | 0.332 | 0.507 | 1003 |

`window` counts a hit on any assistant turn from the same stretch of the session.
It tracks strict recall closely here (0.412 vs 0.441 on single-session), which
says the strict metric is not badly understating - the misses are real misses,
not gold-nomination artefacts.

## What the categories say

**Abstention is 0.000, exactly as predicted.** `/search` has no score floor, so
it always returns rows. Five questions whose subjects are VERIFIED absent from
the corpus by SQL each came back with five confident results. This is the
clearest defect the wider set exposes, and it is invisible to a 7-case set that
only ever asks answerable questions.

**Multi-session aggregation is the weakest real capability at 0.113.** The
window figure for the same cases is 0.500, so the search reaches the right
sessions and then fails to surface the specific turns. That is a ranking
problem, not a reach problem, and it is the opposite diagnosis from the
`curl_cffi` case recorded on 2026-08-29.

**Temporal filtering is correct.** `temporal_exclude` is a clean 1.000: when the
date window predates the gold, the gold never comes back. A filter silently
ignored would score 0.000 here. Note the 4.4s mean - the exclude cases are the
slowest queries in the set, because a restrictive window leaves the lexical arm
little to match.

## A measurement that was wrong, and why

An earlier pass in this session measured a `days`-filtered search at **200
seconds** and recorded it as a production defect in the vector arm. It is not
real. `EXPLAIN ANALYZE` on the same query returns in **4.3ms** on an HNSW index
scan, and the endpoint now answers in 237ms.

The 200s figure was taken while Docker was wedged under host memory pressure.
The same window inflated the re-ranker from ~1.5s to 31-45s per query and the
embedding call from ~20ms to ~1000ms - a roughly 25x slowdown across every
component at once.

**A uniform slowdown across unrelated components is an environment fault, not a
regression.** The tell was that embed, probe, sql and rerank all degraded
together by a similar factor. Checking the query plan cost seconds and refuted
a defect that had already been written down and prioritised.

## Known limitation of the set

The self-containment judge accepts some questions that are still
context-bound - phrasings that name a subject but depend on an unstated
document. They are scored as misses, so the true single-session figure is
somewhere above 0.441. The set is a lower bound, and tightening that gate is the
cheapest available improvement to its precision.
