# HN Launch Post Draft

## Title (80 char HN limit)

**Show HN: 86% on LongMemEval with pgvector + BM25 - memory for Claude Code**

## Alternatives to test

- Show HN: Claude-echoes - 86% on LongMemEval with boring retrieval, no magic
- Show HN: I got tired of losing context between Claude Code sessions (86% LongMemEval)
- Show HN: Cross-session memory for Claude Code - 100% on single-session retrieval

---

## Body

Every Claude Code session starts from scratch. I run 3-5 parallel sessions per day across different projects and lose track of which one fixed what. "When did we debug that partition key bug? Which session had the vercel deploy fix?" — tab through logs, grep files, give up.

Claude-echoes fixes this. It's an ~800-line open-source stack that:

1. Hooks into Claude Code's `UserPromptSubmit` and `Stop` events to capture every user prompt and assistant response verbatim
2. Stores them in Postgres with pgvector + BM25 hybrid search (cosine + lexical, merged via Reciprocal Rank Fusion)
3. Embeds each message with a local Ollama model (`nomic-embed-text`, 768-dim, runs on CPU)
4. Exposes a `/recall` skill inside Claude Code for natural-language search across every past session

```
/recall when did we fix the partition key bug
/recall cors issue with azure functions --project azureprep
/recall vercel deployment failure --days 7
```

Returns ranked hits with verbatim content, project, date, role, and session ID. 200ms round-trip. All local. No cloud. No API keys for retrieval. No summarisation — your history is kept word-for-word.

**Benchmarked honestly: 86.4% on LongMemEval_S (ICLR 2025)**

We ran the full 500-question LongMemEval benchmark — the standard eval for long-term conversational memory. Per-category scores with Sonnet 4.6:

- Single-session user retrieval: **100%** (70/70 - solved)
- Single-session assistant: **98.2%**
- Knowledge updates: **93.6%**
- Temporal reasoning: **84.2%**
- Preference recall: **76.7%**
- Multi-session synthesis: **74.4%**

With Haiku 4.5 (cheapest model, ~$5 per 500 queries): **76.8%**.

The full journey from vanilla pgvector to 86.4%: cosine-only baseline (58.6%) + temporal-aware hybrid retrieval (61.6%) + BM25 RRF fusion (76.4%) + prompt engineering (81.0%) + LLM re-ranker + smart temporal parser (86.4%). Every intermediate result, raw JSONL output, and the GPT-4o-mini judge verdict is in the repo under `benchmarks/`. Run it yourself in ~15 minutes.

**The most interesting finding:** BM25 improved temporal-reasoning by +15.8 points more than our actual temporal logic did. Temporal questions contain specific entities ("the Nordstrom sale", "my crystal chandelier") that cosine embeddings gloss over with semantic neighbors, but BM25 nails by exact keyword match. The lexical signal mattered more than the recency bias. If you're building a RAG system and only using embeddings, you're leaving points on the table.

**Why another memory system?**

MemPalace claimed 96.6% on this exact benchmark. It was [called out by @banteg](https://x.com/banteg/status/2041427374487605614) for hardcoded benchmark patterns and "benchmaxx fraud." We took the opposite approach: boringly conventional retrieval, honest per-category reporting including where we lose (multi-session at 67.7%, preference at 76.7%), and every line of the eval pipeline in one readable file.

If you already run Postgres and don't use Claude Code, you don't need this. If you live in Claude Code all day and keep losing context, this was built for exactly that pain.

**What it deliberately avoids:**

- No LLM deciding what's "important" — everything is kept
- No summarisation — verbatim only
- No cloud — local or your own VPS
- No vendor lock-in — pgvector is just Postgres, ollama is just a binary
- No API keys for the retrieval layer
- Under 1000 LOC for the core. Readable in one sitting.

**Architecture:**

```
Claude Code -> hook -> FastAPI (embed+insert) -> Postgres+pgvector+BM25
                ^
             /recall skill -> FastAPI search -> hybrid RRF retrieval
```

**Install:**

```bash
git clone https://github.com/m4cd4r4/claude-echoes
cd claude-echoes
./scripts/install.sh
```

Ten minutes to a working local install (most of which is pulling the ollama model).

**Caveats, honestly:**

- Anthropic could ship this natively in Claude Code tomorrow and commoditise it. That's fine — the project stays useful as a self-hostable, auditable alternative.
- Multi-session synthesis (74.4%) is still the hardest category. The LLM re-ranker helped (+6.7 points) but cross-session reasoning remains a genuine retrieval ceiling.
- The embedding model is fine, not great. It's CPU-friendly and free, which matters more to me than the last 3% of retrieval quality.

**Roadmap** (in priority order, PRs welcome):

1. Team mode - shared memory across multiple devs on the same repo
3. SQLite + sqlite-vec backend for the "I don't want Docker" crowd
4. Web UI for browsing and exporting
5. Session replay from any hit

Repo: https://github.com/m4cd4r4/claude-echoes

Built in a weekend. MIT licensed. Feedback and PRs very welcome, especially from other heavy Claude Code users who've hit the same pain.

---

## Reply templates for expected comments

**"Why not just use mem0 / letta / MemPalace?"**
> Those are all good — if you're willing to write the Claude Code integration yourself. This project *is* that integration, plus a deliberately tiny retrieval backend. If you already run one of the above and want to bolt it onto Claude Code, the hook in `hooks/chat-logger.js` is ~150 lines and easy to redirect. Also, we actually benchmarked ours and published the raw outputs. Compare that to MemPalace's "96.6%" that nobody can reproduce.

**"Isn't this just pgvector + BM25 with a hook?"**
> Yes, exactly. That's the entire point. The value isn't the retrieval tech — it's the zero-friction Claude Code wiring. A lot of "AI memory" products are 3000+ lines of LLM orchestration solving a problem I didn't have. I wanted something I could audit in a lunch break. The fact that boring retrieval gets 81% on LongMemEval is kind of the point.

**"86% isn't that impressive / Atlas Memory gets 90%"**
> We're within 4 points of Atlas Memory now, using local free embeddings and standard pgvector. Atlas uses a more sophisticated pipeline with summarization. We're at a different point on the pareto frontier: local embeddings (free), standard Postgres, auditable in one sitting. Also: our 86.4% is independently reproducible with raw outputs checked in. Run it yourself.

**"Anthropic will build this."**
> Probably. When they do, this repo still works for people who want to keep their data local or off someone else's server. The implementation is small enough that it's not a huge loss either way.

**"The BM25 finding is interesting"**
> Yeah, that surprised us too. We added BM25 expecting it to help with preference questions (scattered rare terms). Instead the biggest gain was on temporal reasoning (+15.8 points on Sonnet). Temporal questions ask about specific entities ("the Nordstrom sale", "my new keyboard") and cosine similarity was returning semantic neighbors instead of exact matches. BM25 nailed them. If you're building RAG and only using vector search, hybrid RRF is probably your single biggest free upgrade.

**"Privacy?"**
> Data stays on the machine running the Postgres. If you run the server on a remote VPS, use TLS + auth (docs/remote-deployment.md has an nginx + bearer-token example). Don't expose port 8088 to the internet without auth — your verbatim Claude prompts contain everything you've ever said to Claude across every project.

**"Does it work with <IDE / other Claude product>?"**
> The hook is Claude Code specific. The HTTP API (`POST /message`, `GET /search`) is generic — if your tool can fire a POST on each user/assistant turn, it'll work.

**"100% on single-session-user? Really?"**
> 70 out of 70. These are questions like "What did I say about X?" where the answer is in one specific past message. BM25 + cosine hybrid finds it every time - the exact keyword match from BM25 covers what cosine misses and vice versa. This is the core use case for claude-echoes and it works.

---

## HN Update Comment (post on existing thread)

**Update: 86.4% on LongMemEval (up from 81%)**

Two new retrieval improvements landed today:

1. **LLM re-ranker** - retrieve 30 candidates, use Haiku ($0.001/query) to score actual relevance, take top-k. Catches cases where cosine ranks semantically similar but wrong content higher than the real answer.

2. **Smart temporal parser** - for "how long between X and Y" questions, use Haiku to extract both events as separate search queries. Solves the problem where cosine only finds one side of the comparison.

Per-category with Sonnet 4.6:
- Single-session user: 100% (still perfect)
- Single-session assistant: 98.2% (+5.3)
- Knowledge updates: 93.6% (+5.1)
- Temporal reasoning: 84.2% (+8.3)
- Multi-session: 74.4% (+6.7)
- Preference: 76.7% (flat)

Haiku also jumped: 64.4% -> 76.8%. The re-ranker helped preference questions most dramatically (33% -> 83% on Haiku).

All raw outputs and the updated pipeline are in `benchmarks/`. Same deal as before - run it yourself, GPT-4o-mini judge, no hardcoded patterns.

The "caveats" section from the original post noted multi-session and temporal as genuine weak points and mentioned an LLM re-ranker as the next lever. Turns out that was right - it was the single biggest improvement since BM25.

---

## Reddit Post (r/ClaudeAI and r/LocalLLaMA)

**Title:** I built cross-session memory for Claude Code - 86.4% on LongMemEval with boring retrieval

**Body:**

Every Claude Code session starts from scratch. I run multiple parallel sessions per day and constantly lose track of which one fixed what bug, which session had the deploy fix, when we discussed that architecture decision.

**claude-echoes** fixes this. It hooks into Claude Code events, stores every message verbatim in Postgres with pgvector + BM25 hybrid search, and gives you a `/recall` command to search your entire history in natural language:

```
/recall when did we fix the partition key bug
/recall cors issue with azure functions --project azureprep
/recall vercel deployment failure --days 7
```

200ms round-trip. All local. No cloud. No API keys for retrieval. Under 1000 LOC.

**Benchmarked honestly on LongMemEval (ICLR 2025):**

| Category | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|
| Single-session user | **100%** (70/70) | 95.7% |
| Single-session assistant | 98.2% | 92.9% |
| Knowledge updates | 93.6% | 92.3% |
| Temporal reasoning | 84.2% | 63.2% |
| Multi-session | 74.4% | 63.2% |
| Preference | 76.7% | 83.3% |
| **Overall** | **86.4%** | **76.8%** |

The journey from 58.6% baseline to 86.4%: vanilla pgvector -> +BM25 hybrid (RRF fusion) -> +temporal re-ranking -> +prompt engineering -> +LLM re-ranker -> +smart temporal parser. Every intermediate result and the GPT-4o-mini judge output is in the repo.

**Most interesting finding:** BM25 improved temporal reasoning more than our actual temporal logic did. Temporal questions contain specific entities that cosine embeddings gloss over but BM25 nails by exact match. If you're building RAG and only using embeddings, hybrid search is probably your biggest free upgrade.

**What it deliberately avoids:** no LLM deciding what's "important", no summarisation, no cloud, no vendor lock-in, readable in one sitting.

Repo: https://github.com/m4cd4r4/claude-echoes

MIT licensed. Built with Claude Code. Feedback welcome.

---

## Twitter/X Post (@m4cd4r4)

**Thread (3 tweets):**

**1/3**
claude-echoes now scores 86.4% on LongMemEval - up from 81% two days ago.

Two new features:
- LLM re-ranker (Haiku scores relevance for $0.001/query)
- Smart temporal parser (extracts both events from "how long between X and Y")

100% on single-session retrieval. 84% on temporal reasoning.

All with local pgvector + nomic-embed-text. No magic.

**2/3**
The most surprising result: Haiku jumped from 64% to 77% - and preference questions went 33% -> 83%.

The re-ranker solved what months of prompt engineering couldn't. Turns out asking a cheap LLM "is this actually relevant?" beats cosine similarity at judging relevance. Who knew.

Full raw outputs + GPT-4o-mini judge results in the repo.

**3/3**
For context: MemPalace claimed 96.6% on this same benchmark before getting caught with hardcoded patterns. Atlas Memory reports 90.18%.

We're at 86.4% with boringly conventional retrieval and every line auditable. Gap closing.

Repo: https://github.com/m4cd4r4/claude-echoes

Built with @ClaudeCode in a weekend. MIT licensed.
