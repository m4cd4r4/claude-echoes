#!/usr/bin/env python3
"""
Ingest the STANDARD Claude Code transcripts: ~/.claude/projects/<slug>/<uuid>.jsonl

This is the file layout every Claude Code user has. The sibling script,
ingest_chat_history.py, reads a bespoke ~/.claude/chat-history/{date}/{project}.jsonl
that only exists if you run a particular logging hook - so without this script
the tool indexes nobody's history but its author's, which is not much of an
open-source memory tool.

WHAT IT KEEPS, and why the filtering is most of the work.

A transcript line is not a message. In one 840-line sample the `type` field took
thirteen values, of which only two were conversation:

    attachment 306   assistant 231   user 116   bridge-session 36   mode 35
    atis-latch 35    last-prompt 34  queue-operation 32  system 6
    file-history-snapshot 4  file-history-delta 3  ai-title 1  cost-state 1

Everything outside {user, assistant} is editor and harness bookkeeping. Indexing
it would bury real conversation under machine chatter, and the retrieval cost
lands on exactly the queries this tool exists to answer.

Within those two, four more things are dropped:

  - tool traffic. An assistant turn is a list of content blocks; a turn made
    only of tool_use has no prose and no recall value. A user turn whose content
    is a list is almost always tool_result being fed back, not something a
    person typed.
  - isMeta turns - hook output, system reminders, injected context. Written by
    machinery, attributed to the user, and never said by anyone.
  - sidechains (isSidechain) - subagent conversations. Off by default because a
    fan-out of twelve agents can outweigh the session that spawned them, and
    they are the model talking to itself. --include-subagents keeps them.
  - command scaffolding - <command-name>, <local-command-stdout> and friends,
    which are UI plumbing around a slash command rather than its text.

SOURCE TAGGING. Rows are written with source='transcript'. If you also run the
hook-based ingest, both layers describe the SAME conversations from different
angles: the hook truncates content at 5,000 chars and stamps its own time, the
transcript carries the full text and the harness's time. Neither content_hash
nor the (session_id, role, created_at) resume key will match across them, so the
two WILL coexist as near-duplicates. That is a deliberate, visible outcome
rather than a silent merge - `source` is what lets you tell them apart, count
the overlap, and decide. Run --overlap to measure it before you commit to it.

    python scripts/ingest_transcripts.py --dry-run
    python scripts/ingest_transcripts.py --overlap
    python scripts/ingest_transcripts.py --since 2026-08-01
    python scripts/ingest_transcripts.py --project solaisoft
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_chat_history import (          # noqa: E402  - shares the tested path
    DB_DSN, embed_resilient, vec_literal, psycopg2, execute_values,
)

DEFAULT_DIR = Path.home() / ".claude" / "projects"
BATCH_SIZE = 256
MIN_CHARS = 2          # a bare "y" is still a turn; empty strings are not

# UI plumbing wrapped around slash commands. The text a person typed is the
# command's argument, which arrives separately; these tags are the harness.
_SCAFFOLD = (
    "<command-name>", "<command-message>", "<command-args>",
    "<local-command-stdout>", "<local-command-stderr>",
    "<user-memory-input>", "<system-reminder>",
)


def parse_ts(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def text_of(message: dict) -> str:
    """Prose only. Returns '' for a turn that carried no human-readable text."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        # 'text' is prose. 'tool_use' and 'tool_result' are machine traffic;
        # 'thinking' is deliberately excluded - it is not what was said.
        if block.get("type") == "text":
            t = block.get("text") or ""
            if t.strip():
                parts.append(t.strip())
    return "\n\n".join(parts).strip()


def project_of(rec: dict, fallback: str) -> str:
    """Project name from cwd, matching how the hook-based ingest names things.

    The directory slug (C--Scratch-azure-practice-exam-platform) encodes the
    same path with separators mangled, so cwd is both cleaner and more accurate.
    """
    cwd = rec.get("cwd") or ""
    if cwd:
        base = os.path.basename(cwd.replace("\\", "/").rstrip("/"))
        if base:
            return base
    return fallback


def iter_files(root: Path, include_subagents: bool) -> Iterator[Path]:
    """os.walk, not glob.

    A glob over this tree expands to five figures of paths, and on Windows an
    argument list that long fails in ways that look like 'no files found'
    rather than an error. Measured here: `ls ~/.claude/projects/*/*.jsonl`
    returned 0 while the tree held 12,308 files.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        if not include_subagents and os.path.basename(dirpath) == "subagents":
            continue
        for fn in filenames:
            if fn.endswith(".jsonl"):
                yield Path(dirpath) / fn


def read_messages(root: Path, since, project_filter, include_subagents) -> Iterator[dict]:
    for path in iter_files(root, include_subagents):
        slug = path.parent.name
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue          # a partial trailing write is normal

                if rec.get("type") not in ("user", "assistant"):
                    continue
                if rec.get("isMeta"):
                    continue
                if rec.get("isSidechain") and not include_subagents:
                    continue

                msg = rec.get("message")
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role not in ("user", "assistant"):
                    continue

                content = text_of(msg)
                if len(content) < MIN_CHARS:
                    continue
                if any(tag in content for tag in _SCAFFOLD):
                    continue

                ts = parse_ts(rec.get("timestamp") or "")
                if ts is None:
                    continue          # no honest timestamp, no row
                if since and ts < since:
                    continue

                proj = project_of(rec, slug)
                if project_filter and proj != project_filter:
                    continue

                sid = rec.get("sessionId") or path.stem
                yield {
                    "session_id": sid,
                    "project": proj,
                    "role": role,
                    "content": content,
                    "model": msg.get("model"),
                    "created_at": ts,
                }


def report_overlap(cur, msgs):
    """Say plainly how much of this duplicates what is already indexed."""
    cur.execute("SELECT count(*) FROM messages WHERE source = 'transcript'")
    have_t = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM messages WHERE source <> 'transcript'")
    have_other = cur.fetchone()[0]
    cur.execute("SELECT session_id, role, created_at FROM messages")
    seen = {(r[0], r[1], r[2]) for r in cur.fetchall()}
    new = [m for m in msgs
           if (m["session_id"], m["role"], m["created_at"]) not in seen]
    sessions = {m["session_id"] for m in msgs}
    cur.execute("SELECT DISTINCT session_id FROM messages WHERE source <> 'transcript'")
    other_sessions = {r[0] for r in cur.fetchall()}
    shared = sessions & other_sessions

    print()
    print("already indexed:  {:,} transcript rows, {:,} from other sources".format(
        have_t, have_other))
    print("this run would add {:,} of {:,} parsed rows".format(len(new), len(msgs)))
    print("sessions in these transcripts: {:,}".format(len(sessions)))
    print("...of which {:,} ALSO appear under another source".format(len(shared)))
    if shared:
        print()
        print("Those {:,} sessions are already represented by a different layer.".format(
            len(shared)))
        print("The two will coexist: the hook truncates at 5,000 chars and stamps")
        print("its own time, so no dedupe key matches across them. Use source= to")
        print("tell them apart. This is visible duplication, not a silent merge.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overlap", action="store_true",
                    help="report duplication against what is already indexed, then stop")
    ap.add_argument("--since", help="ISO date, e.g. 2026-06-01")
    ap.add_argument("--project", help="only this project")
    ap.add_argument("--limit", type=int, help="stop after N messages")
    ap.add_argument("--include-subagents", action="store_true",
                    help="also index sidechain/subagent turns (off by default)")
    args = ap.parse_args()

    root = Path(args.log_dir)
    if not root.exists():
        sys.exit("transcript dir not found: %s" % root)

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    print("reading %s%s%s%s" % (
        root,
        ("  since=" + args.since) if args.since else "",
        ("  project=" + args.project) if args.project else "",
        "  +subagents" if args.include_subagents else ""))

    msgs = []
    for m in read_messages(root, since, args.project, args.include_subagents):
        msgs.append(m)
        if args.limit and len(msgs) >= args.limit:
            break

    if not msgs:
        sys.exit("no messages matched - nothing to do")

    lo = min(m["created_at"] for m in msgs)
    hi = max(m["created_at"] for m in msgs)
    chars = sum(len(m["content"]) for m in msgs)
    projects = {m["project"] for m in msgs}
    print("parsed {:,} messages  {} to {}  {:.1f}M chars  {:,} projects".format(
        len(msgs), lo.date(), hi.date(), chars / 1e6, len(projects)))

    if args.dry_run:
        for m in msgs[:5]:
            print("  %s  %-22s %-9s %r" % (
                m["created_at"].strftime("%Y-%m-%d %H:%M"),
                m["project"][:22], m["role"], m["content"][:60]))
        print("dry run - nothing inserted")
        return

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    if args.overlap:
        report_overlap(cur, msgs)
        cur.close(); conn.close()
        return

    # Skip work already done BEFORE embedding it. Same reasoning as the sibling
    # script: ON CONFLICT makes a re-run correct but not cheap, and resuming a
    # large backfill at 90% would otherwise burn the GPU to insert nothing.
    cur.execute("SELECT session_id, role, created_at FROM messages")
    seen = {(r[0], r[1], r[2]) for r in cur.fetchall()}
    if seen:
        before = len(msgs)
        msgs = [m for m in msgs
                if (m["session_id"], m["role"], m["created_at"]) not in seen]
        print("skipping {:,} already ingested; {:,} to do".format(
            before - len(msgs), len(msgs)))
        if not msgs:
            print("nothing new - up to date")
            cur.close(); conn.close()
            return

    t0 = time.time()
    inserted = failed = 0
    for start in range(0, len(msgs), BATCH_SIZE):
        chunk = msgs[start:start + BATCH_SIZE]
        pairs = embed_resilient([m["content"] for m in chunk])
        failed += len(chunk) - len(pairs)

        rows = [
            (chunk[i]["session_id"], chunk[i]["project"], None, chunk[i]["role"],
             chunk[i]["content"], chunk[i]["model"], vec_literal(v),
             chunk[i]["created_at"], "transcript")
            for i, v in pairs
        ]
        if rows:
            execute_values(
                cur,
                """
                INSERT INTO messages
                    (session_id, project, machine, role, content, model,
                     embedding, created_at, source)
                VALUES %s
                ON CONFLICT (content_hash) DO NOTHING
                """,
                rows,
                template="(%s,%s,%s,%s,%s,%s,%s::vector,%s,%s)",
            )
            inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            conn.commit()

        done = min(start + BATCH_SIZE, len(msgs))
        rate = done / max(time.time() - t0, 1e-6)
        eta = (len(msgs) - done) / max(rate, 1e-6)
        print("  {:,}/{:,}  inserted={:,} fail={:,}  {:.0f}/s  eta {:.0f}m".format(
            done, len(msgs), inserted, failed, rate, eta / 60), flush=True)

    cur.close(); conn.close()
    print("done in {:.1f}m  inserted={:,}  failed={:,}".format(
        (time.time() - t0) / 60, inserted, failed))
    if failed:
        print("%d message(s) could not be embedded and were skipped. "
              "Re-run to retry them." % failed, file=sys.stderr)


if __name__ == "__main__":
    main()
