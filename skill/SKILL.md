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

If the server is unreachable, say so plainly. Do not fall back to guessing or summarising from your own context.

**Step 3 — Present results.** For each hit, show:

- Similarity score (3 decimals)
- Project name + role + date (convert UTC to user's local timezone)
- First 200 chars of content as a snippet, with ellipsis if truncated
- Session ID in `[abbreviated]` form (first 8 chars) so the user can ask for full context

Format example:

```
0.731  azureprep    user       2026-04-08 15:09   [ada9a195]
       "I would like semantic search across verbatim chat history..."

0.684  cosmos-collective  assistant  2026-04-05 22:14   [b831cc02]
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
