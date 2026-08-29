# Retrieval test, 2026-08-29

Corpus: 108,570 messages = exactly the source file count. 29,225 sessions,
2 Jan - 29 Aug 2026, 100% embedded (`count(embedding)` = `count(*)`).
Roles: 66,531 user / 42,039 assistant.

## What passed

| Query | Verdict |
|---|---|
| "why was ntfy retired and what replaced it" | HIT at rank 2 - the 2026-07-20 approval, the correct date |
| "decision to exclude Wix from site refresh" | HIT at rank 1 AND 2 - the decision, plus **Macdara's own words giving the reason** |
| `days=7` filter | Correct - narrows to 27 Aug only |

`created_at` preservation works. That was the whole reason for replacing
`backfill.py`, which stamped every row `NOW()`, and it is now verified rather
than assumed.

## The defect: a natural-language QUESTION collapses the lexical arm

| Query form | Rows the lexical arm can see |
|---|---|
| `how many certifications and questions does AzurePrep actually have` | **5** |
| `AzurePrep certifications questions` | **511** |
| `certification \| question \| azureprep` (OR) | 29,324 |

`websearch_to_tsquery` **ANDs every content word**. Add "how many", "actually",
"have" and the conjunction excludes the answer. The hybrid then silently
degrades to vector-only, and the vector arm is weak on "which number was it" -
so the query returned three irrelevant results while four messages in the
corpus state `32 certifications, 9,637 questions, 9,615 flashcards`.

Proven not to be a recall-depth problem: `candidates=500` returns the same
three misses as `candidates=60`. It is ranking, not reach.

Rephrasing to `AzurePrep 9,637 questions 9,615 flashcards` hits at rank 1 -
which is useless, because knowing the number is the thing you were asking for.

**This is the product's core use case failing.** "When did we decide X, and
why" is a question, and questions are the input form that breaks it.

Fix direction: when the AND query yields under ~20 rows, retry the lexical arm
with the content words OR-ed and let `ts_rank` order them. Keep AND first -
it is precise when it works.

## Two smaller things found

- **`/health` returns `{"ok":true}` while search is 503.** It checks the DB and
  reads a model name from config; it never embeds anything. A health check that
  cannot fail when the service is down is decoration.
- **The 503 said "embedding service unavailable" when the real fault was SQL.**
  The running container was a stale image carrying the pre-fix
  `params = [qvec, q]`, which asyncpg rejected on the vector-only path. Ollama
  was healthy the whole time. `docker compose up -d --build server` fixed it.
  A rebuild is not automatic after editing `server/app.py`.

---

# Fix, and what it measured

`scripts/eval_retrieval.mjs` is the harness. Every expected fact was confirmed
present by SQL first, so a miss is a retrieval failure and never a data gap.

| | score | slowest query |
|---|---|---|
| Baseline | **4/7** | - |
| After the fix | **6/7** | **606ms** |

Per-query server timings are in the output, so a regression shows up as a
number rather than a feeling.

## What worked: relax the AND, in one statement

Content words only (stopwords and question words dropped), AND-ed, longest
first, relaxed one word at a time until something matches.

## Three things that were tried and measured WORSE

1. **Pure OR fallback.** 29,324 rows, 29-46 SECONDS a query. `ts_rank` has no
   IDF term, so a wide OR buries the rare word under common ones. Rejected on
   both speed and quality.

2. **The relaxation as a LOOP of prepared probes: 71-139 SECONDS.** Each probe
   is 2ms under `EXPLAIN ANALYZE`, which is exactly why the loop looked safe.
   asyncpg prepares the statement, Postgres switches to a **generic plan** after
   a few executions, the generic plan cannot use the tsquery's selectivity, and
   it falls back to a sequential scan recomputing `to_tsvector` over every row.
   Collapsing the tiers into ONE `unnest(...) WITH ORDINALITY` statement took
   the worst probe from **139,031ms to 17ms**.

   The general lesson: *a per-statement timing measured in isolation does not
   predict the cost of that statement executed repeatedly through a prepared
   connection.* Measure the loop, not the query.

3. **Ordering terms by real document frequency instead of length: 5/7, down
   from 6/7.** It was designed to fix the one failing case - the question
   "what fixes a scraper getting 429 on every request" kept `request`
   (df 3,827) and dropped `429` (df 222). Ordering rarest-first did keep `429`,
   and broke a case that had been passing, because rarest-first builds an
   over-specific conjunction that matches SOMETHING early and the loop stops at
   the first non-empty tier. Seven cases is far too small to tune on, so the
   simpler rule stays and the negative result is recorded here.

## Still failing, honestly

`what fixes a scraper getting 429 on every request` -> wants `curl_cffi`.
The question and the answer share almost no vocabulary: the answer is about TLS
fingerprinting, and the only distinctive token in the question is `429`. This
is the case a real re-ranker would catch, and the server has none.

## Also fixed

- `/health` now embeds a probe string and returns 503 when either dependency is
  down. It previously returned `ok:true` throughout an outage because it only
  read a model name out of config.
- `README.md` no longer implies the served pipeline is the benchmarked one. The
  86.4% figure comes from `benchmarks/run_longmemeval.py`, which has temporal
  re-ranking and an LLM re-ranker; `/search` has neither.

## The cold-load, fixed

`ollama ps` reported the embedding model expiring **"2 minutes from now"**, so
any question after a short pause paid a **~50 second** reload while every
follow-up answered in ~200ms. The first question of a session looked broken and
nothing in the logs said why.

`OLLAMA_KEEP_ALIVE: "-1"` on the container, plus `keep_alive: -1` on each
request so it also holds against an ollama the user started themselves.
`ollama ps` now reads **`UNTIL  Forever`** for 376 MB resident.

Worth recording: the new `/health` caught this fix landing. It returned
**503 `embeddings: down`** in the seconds between container start and the model
finishing its load - which is the first time that endpoint has ever been able
to report a real fault, and is exactly the behaviour the old `ok:true`-always
version could not produce.

## Final numbers, warm

| | |
|---|---|
| Score | **6/7 at top-5** |
| Slowest query | **731ms** |
| Median | ~130ms |
| Embed | 39-71ms |
| Probe | 3-40ms (was up to 139,031ms) |
