---
name: recall
description: Semantic search across verbatim Claude Code chat history from every session and project. Backed by the local claude-echoes server (pgvector + nomic-embed). Use when the user asks "when did we/I...", "find that conversation about...", "what did I say about...", "recall...", or "/recall".
user-invocable: true
---

# /recall — Semantic Recall Over Verbatim Chat History

Queries the local `claude-echoes` server for past Claude Code messages matching a natural-language query. Returns verbatim content with project, role, date, and session context.

**Server:** `http://localhost:8088` by default. Override with `ECHOES_URL` env var for remote deployments.

---

## Quick reference

| Command | What it does |
|---|---|
| `/recall <query>` | Top 10 semantic hits across all projects |
| `/recall <query> --project <name>` | Filter to one project |
| `/recall <query> --days <N>` | Only messages from last N days |
| `/recall <query> --role user` | Only user prompts |
| `/recall <query> --role assistant` | Only assistant responses |
| `/recall <query> --limit <N>` | Change result count (default 10) |
| `/recall <query> --vector-only` | Dense-vector search only, skipping the lexical arm |

---

## Execution

**Step 1 — Parse the query.** Extract the natural-language query and any `--project`, `--days`, `--role`, `--limit` flags.

**Step 2 — Call the search endpoint.**

```bash
curl -sG "${ECHOES_URL:-http://localhost:8088}/search" \
  --data-urlencode "q=<the query>" \
  --data-urlencode "limit=<limit or 10>" \
  [--data-urlencode "project=<project>"] \
  [--data-urlencode "role=<role>"] \
  [--data-urlencode "days=<days>"]
```

Pass `hybrid=false` only for `--vector-only`.

Search is hybrid by default: dense vector and Postgres full-text, fused with
Reciprocal Rank Fusion. The lexical arm is what catches exact tokens an
embedding blurs - an error string, a flag, a commit sha, a port number - so
prefer the default. `--vector-only` exists to compare the two on the same data,
which is the only honest way to tell whether a disappointing result is the
retrieval or the corpus.

If the server is unreachable, say so plainly. Do not fall back to guessing or summarising from your own context.

**A miss is a real answer.** An empty result means the conversation is not in the
index, and the index starts at 2026-01-02. Say that, rather than reaching into
your own context and presenting a recollection as a search result - the whole
value of this tool is that its answers are verbatim records.

**Step 3 — Present results.** For each hit, show:

- Rank, as `1.` `2.` `3.` - NOT the `similarity` field. Under Reciprocal Rank
  Fusion that number is ~0.016-0.033 on every hit; it orders results and means
  nothing on its own, so showing it invites the user to read a 0.03 as a bad match
- Project name + role + date (convert UTC to user's local timezone)
- First 200 chars of content as a snippet, with ellipsis if truncated
- Session ID in `[abbreviated]` form (first 8 chars) so the user can ask for full context

Format example:

```
1.  azureprep    user       2026-04-08 15:09   [ada9a195]
    "I would like semantic search across verbatim chat history..."

2.  cosmos-collective  assistant  2026-04-05 22:14   [b831cc02]
    "The partition key for the users container must be /email..."
```

**Step 4 — Offer follow-up.** After showing results, ask if the user wants the full surrounding conversation for any hit. If yes, call:

```bash
curl -s "${ECHOES_URL:-http://localhost:8088}/session/<session_id>"
```

And render the full message list in chronological order.

---

## When NOT to use recall

- The user is asking about the current session — that's already in your context
- The answer is in code or git history — read files or `git log` instead
- The user wants a project summary — recall is for specific conversations, not overviews
- The query mentions "today" or "right now" without specifying historical intent

---

## Triggers

Use this skill when the user says any of:

- `/recall ...`
- "recall when I/we..."
- "find that conversation about..."
- "what did I say about..."
- "when did we fix/build/discuss..."
- "do you remember when..."
- "pull up the session where..."
