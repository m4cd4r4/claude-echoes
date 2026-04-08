# HN Launch Post Draft

## Title (80 char HN limit)

**Show HN: Claude-echoes – verbatim semantic memory across Claude Code sessions**

## Alternatives to test

- Show HN: Cross-session memory for Claude Code (pgvector + local embeddings)
- Show HN: I got tired of losing context between Claude Code sessions
- Show HN: Claude-echoes – recall any past Claude Code conversation by meaning

---

## Body

Every Claude Code session starts from scratch. I run 3-5 parallel sessions per day across different projects and lose track of which one fixed what. "When did we debug that partition key bug? Which session had the vercel deploy fix?" — tab through logs, grep files, give up.

Claude-echoes fixes this. It's a ~800-line open-source stack that:

1. Hooks into Claude Code's `UserPromptSubmit` and `Stop` events to capture every user prompt and assistant response verbatim
2. Stores them in Postgres with pgvector + HNSW index
3. Embeds each message with a local Ollama model (`nomic-embed-text`, 768-dim, runs on CPU)
4. Exposes a `/recall` skill inside Claude Code for natural-language search across every past session

```
/recall when did we fix the partition key bug
/recall cors issue with azure functions --project azureprep
/recall vercel deployment failure --days 7
```

Returns ranked hits with verbatim content, project, date, role, and session ID. 200ms round-trip. All local. No cloud. No API keys. No summarisation — your history is kept word-for-word.

**Why another memory system?**

MemPalace, mem0, letta, and others all exist. They're good at what they do. What they don't do is the Claude Code integration: the hook, the skill, the zero-friction capture-while-you-work loop. If you already run Postgres and don't use Claude Code, MemPalace is probably better for you. If you live in Claude Code all day and keep losing context, this was built for exactly that pain.

**What it deliberately avoids:**

- No LLM deciding what's "important" — everything is kept
- No summarisation — verbatim only
- No cloud — local or your own VPS
- No vendor lock-in — pgvector is just Postgres, ollama is just a binary
- No API keys
- Under 1000 LOC for the core. Readable in one sitting.

**Architecture:**

```
Claude Code → hook → FastAPI (embed+insert) → Postgres+pgvector
                ↑
             /recall skill → FastAPI search → HNSW vector search
```

**Install:**

```bash
git clone https://github.com/m4cd4r4/claude-echoes
cd claude-echoes
./scripts/install.sh
```

Ten minutes to a working local install (most of which is pulling the ollama model).

**Caveats, honestly:**

- Anthropic could ship this natively in Claude Code tomorrow and commoditise it. That's fine — the project stays useful as a self-hostable, auditable alternative for people who want their data off someone else's server.
- The embedding model is fine, not great. It's CPU-friendly and free, which matters more to me than the last 3% of retrieval quality. Swap it for any Ollama-compatible embed model via env var.
- Retrieval quality depends on how much history you've accumulated. First day: meh. First week: genuinely useful. First month: "how did I work without this."

**Roadmap** (in priority order, PRs welcome):

1. Team mode — shared memory across multiple devs on the same repo
2. SQLite + sqlite-vec backend for the "I don't want Docker" crowd
3. Web UI for browsing and exporting
4. Session replay from any hit
5. Prompt-level clustering ("what am I working on most?")

**Not on the roadmap:** hosted SaaS (yet), auto-summarisation, LLM-based "smart" memory.

Repo: https://github.com/m4cd4r4/claude-echoes

Built in a weekend. MIT licensed. Feedback and PRs very welcome, especially from other heavy Claude Code users who've hit the same pain.

---

## Reply templates for expected comments

**"Why not just use mem0 / letta / MemPalace?"**
> Those are all good — if you're willing to write the Claude Code integration yourself. This project *is* that integration, plus a deliberately tiny retrieval backend. If you already run one of the above and want to bolt it onto Claude Code, the hook in `hooks/chat-logger.js` is ~150 lines and easy to redirect.

**"Isn't this just pgvector with a hook?"**
> Yes, exactly. That's the entire point. The value isn't the retrieval tech — it's the zero-friction Claude Code wiring. A lot of "AI memory" products are 3000+ lines of LLM orchestration solving a problem I didn't have. I wanted something I could audit in a lunch break.

**"Anthropic will build this."**
> Probably. When they do, this repo still works for people who want to keep their data local or off someone else's server. The implementation is small enough that it's not a huge loss either way.

**"Privacy?"**
> Data stays on the machine running the Postgres. If you run the server on a remote VPS, use TLS + auth (docs/remote-deployment.md has an nginx + bearer-token example). Don't expose port 8088 to the internet without auth — your verbatim Claude prompts contain a lot.

**"Does it work with <IDE / other Claude product>?"**
> The hook is Claude Code specific. The HTTP API (`POST /message`, `GET /search`) is generic — if your tool can fire a POST on each user/assistant turn, it'll work.
