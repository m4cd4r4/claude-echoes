# Backfilling existing history

If you already have Claude Code session logs on disk (e.g. in `~/.claude/daily-logs/`), the backfill script will parse them and insert them into the echoes database with embeddings.

## Quick start

```bash
pip install psycopg2-binary
python scripts/backfill.py
```

By default it reads `~/.claude/daily-logs/*.md` and connects to `postgresql://echoes:echoes@localhost:5434/echoes` (matching the docker-compose defaults).

## Custom log dir

```bash
python scripts/backfill.py /path/to/your/logs
```

## Dry run

```bash
python scripts/backfill.py --dry-run
```

Prints what would be inserted without touching the database. Use this to verify the parser handles your log format before committing.

## Log format

The default parser expects markdown files like:

```
## Session abc123 — my-project
**2026-04-08T15:09:10** — user
I would like semantic search across verbatim chat history.

**2026-04-08T15:14:16** — assistant
Good problem to solve. Let me check what's actually being stored today...
```

If your logs are in a different format, edit `parse_daily_log` in [scripts/backfill.py](../scripts/backfill.py). The function just needs to yield dicts with `session_id`, `project`, `role`, `content`.

## Expected runtime

On a CPU-only VPS, nomic-embed-text runs at roughly 2-5 messages per second. So:

- 1,000 messages ≈ 5-8 minutes
- 10,000 messages ≈ 50-80 minutes
- 100,000 messages ≈ 8-14 hours

The script is idempotent — safe to cancel and re-run. It skips rows that already exist (keyed by `session_id + role + first 200 chars of content`).

## Idempotency caveat

The dedup key is `(session_id, role, first 200 chars)`. If two different messages in the same session start with identical first 200 chars, only the first is kept. This is a deliberate trade-off for speed. If that matters to you, drop the check and let Postgres enforce a stricter unique constraint of your choosing.
