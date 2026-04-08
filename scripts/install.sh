#!/usr/bin/env bash
# claude-echoes installer
# Spins up docker stack, pulls the embed model, wires up the Claude Code hook + skill.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLAUDE_DIR="${CLAUDE_DIR:-$HOME/.claude}"
HOOK_DIR="$CLAUDE_DIR/hooks/claude-echoes"
SKILL_DIR="$CLAUDE_DIR/skills/recall"

echo "=== claude-echoes installer ==="
echo "repo:   $REPO_DIR"
echo "claude: $CLAUDE_DIR"
echo

# ---- preflight -----------------------------------------------------------
command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: docker compose not found"; exit 1; }

if [[ ! -d "$CLAUDE_DIR" ]]; then
  echo "ERROR: $CLAUDE_DIR not found. Is Claude Code installed?"
  exit 1
fi

# ---- docker stack --------------------------------------------------------
echo "==> starting docker stack (postgres + ollama + server)..."
cd "$REPO_DIR"
docker compose up -d --build

echo "==> waiting for postgres..."
for i in {1..30}; do
  if docker exec echoes-postgres pg_isready -U echoes >/dev/null 2>&1; then
    echo "    ready"
    break
  fi
  sleep 1
done

echo "==> applying schema..."
docker exec -i echoes-postgres psql -U echoes -d echoes < "$REPO_DIR/sql/001_init.sql"

echo "==> pulling nomic-embed-text model (~275MB)..."
docker exec echoes-ollama ollama pull nomic-embed-text

echo "==> waiting for server..."
for i in {1..30}; do
  if curl -sf http://localhost:8088/health >/dev/null 2>&1; then
    echo "    ready"
    break
  fi
  sleep 1
done

# ---- hook ----------------------------------------------------------------
echo "==> installing Claude Code hook..."
mkdir -p "$HOOK_DIR"
cp "$REPO_DIR/hooks/chat-logger.js" "$HOOK_DIR/chat-logger.js"
chmod +x "$HOOK_DIR/chat-logger.js"

# ---- skill ---------------------------------------------------------------
echo "==> installing /recall skill..."
mkdir -p "$SKILL_DIR"
cp "$REPO_DIR/skill/SKILL.md" "$SKILL_DIR/SKILL.md"

# ---- settings.json hint --------------------------------------------------
SETTINGS="$CLAUDE_DIR/settings.json"
echo
echo "==> hook script installed at: $HOOK_DIR/chat-logger.js"
echo
echo "You need to register the hook in $SETTINGS manually. Add this to your"
echo "existing 'hooks' object (merge, don't replace):"
echo
cat <<'JSON'
  "hooks": {
    "UserPromptSubmit": [
      { "command": "node ~/.claude/hooks/claude-echoes/chat-logger.js" }
    ],
    "Stop": [
      { "command": "node ~/.claude/hooks/claude-echoes/chat-logger.js" }
    ]
  }
JSON
echo

# ---- smoke test ----------------------------------------------------------
echo "==> smoke test..."
curl -sf http://localhost:8088/health && echo

echo "==> writing a test message..."
curl -s -X POST http://localhost:8088/message \
  -H "Content-Type: application/json" \
  -d '{"session_id":"install-test","project":"claude-echoes","role":"user","content":"hello from the installer"}' | grep -q '"embedded":true' \
  && echo "    write OK" || echo "    WARN: embed did not complete (ollama may still be warming up)"

echo "==> searching..."
sleep 1
curl -s "http://localhost:8088/search?q=installer%20test&limit=1" | head -c 400
echo
echo

echo "=== done ==="
echo
echo "Next steps:"
echo "  1. Add the hook to $SETTINGS (see snippet above)"
echo "  2. Start a new Claude Code session"
echo "  3. After a few exchanges, try: /recall installer test"
echo
echo "To backfill existing session logs:"
echo "  pip install psycopg2-binary"
echo "  python scripts/backfill.py"
