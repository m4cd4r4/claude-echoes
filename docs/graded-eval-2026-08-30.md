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

## Re-run 2026-09-10, after the `role=assistant` fix

`scripts/eval_graded.mjs` now sets `role=assistant` for every gold-bearing case,
removing the question's own row, which had been taking RRF rank 1 in each of
them and capping gold at rank 2. Same 67 cases, same corpus.

Re-ranker OFF:

| category | n | Recall@5 | MRR | nDCG@5 | ms |
|---|---|---|---|---|---|
| single_session | 34 | **0.500** | **0.350** | 0.388 | 387 |
| temporal_include (padded, see below) | 10 | 0.600 | 0.550 | 0.563 | 126 |
| temporal_exclude | 10 | 1.000 | 1.000 | 1.000 | 1526 |
| multi_session | 8 | 0.113 | 0.344 | 0.140 | 230 |
| abstention | 5 | 0.000 | 0.000 | 0.000 | 530 |

Re-ranker ON - the first clean full run achieved (66/67; one case errored after
retries and is excluded from every figure):

| category | n | Recall@5 | MRR | nDCG@5 | ms |
|---|---|---|---|---|---|
| single_session | 33 | **0.545** | **0.404** | 0.438 | 5798 |
| temporal_include (padded, see below) | 10 | 0.700 | 0.633 | 0.650 | 2820 |
| temporal_exclude | 10 | 1.000 | 1.000 | 1.000 | 2136 |
| multi_session | 8 | **0.081** | **0.219** | 0.091 | 1298 |
| abstention | 5 | 0.000 | 0.000 | 0.000 | 1968 |

The `role=assistant` fix roughly doubled single-session MRR, 0.171 to 0.350.

**The re-ranker is not a uniform win.** It gains single_session (+0.045 Recall@5)
and temporal_include (+0.100), and it loses multi_session (0.113 to 0.081, MRR
0.344 to 0.219) - the category that was already weakest. It costs about 15x in
latency, 387ms to 5798ms median on single-session. Both directions are one run
each; neither is a trend.

Abstention is unchanged at 0.000 with the re-ranker on. A judge that sees only
five retrieved rows cannot decide the corpus lacks the subject, so this stays a
score-floor problem rather than a ranking one.

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
window figure for the same cases is 0.500, but that does NOT license a
session-level diagnosis: `build_eval_set.mjs:105` sets `window_ids = gold_ids`
for this category, so 0.500 only says "at least one gold hit in half the cases".
There is no session-level check behind it. Whether this is a ranking problem or
a reach problem is unmeasured.

**Temporal filtering is correct.** `temporal_exclude` is a clean 1.000: when the
date window predates the gold, the gold never comes back. A filter silently
ignored would score 0.000 here. Note the 4.4s mean - the exclude cases are the
slowest queries in the set, because a restrictive window leaves the lexical arm
little to match.

**`temporal_include` has a shelf life, and the 0.600 above is already decayed.**
`build_eval_set.mjs` writes `days` as the gold row's *exact* age at build time,
so the window has zero margin and closes on the gold as wall-clock advances.
Measured 2026-09-10, twelve days after the set was built: all 10 golds sat 8-9
days outside their own window and the category scored **0.000** - not a
retrieval regression, a rotting denominator. Re-running the same 10 cases with
`days + 30` restores 0.600 (re-ranker off) and gives 0.700 (re-ranker on).

Treat any `temporal_include` figure as valid only on the day the set was built,
and pad the window before comparing runs taken on different dates. The fix is to
store an absolute cutoff timestamp per case instead of a relative day count.

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
