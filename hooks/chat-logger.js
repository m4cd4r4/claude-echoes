#!/usr/bin/env node
/**
 * claude-echoes chat logger
 *
 * Wired into Claude Code via settings.json as a UserPromptSubmit + Stop hook.
 * Reads the hook event from stdin, extracts the role+content, and POSTs it
 * to the echoes server. Fire-and-forget: never blocks the prompt.
 *
 * Env vars:
 *   ECHOES_URL   - base URL of the echoes server (default: http://localhost:8088)
 *   ECHOES_TOKEN - optional bearer token for remote deployments
 *   DEBUG_ECHOES - any truthy value enables stderr logging
 */

const http  = require('http');
const https = require('https');
const path  = require('path');
const fs    = require('fs');

const ECHOES_URL  = process.env.ECHOES_URL  || 'http://localhost:8088';
const ECHOES_TOKEN = process.env.ECHOES_TOKEN || '';
const DEBUG       = !!process.env.DEBUG_ECHOES;

function log(...args) { if (DEBUG) console.error('[echoes]', ...args); }

function readStdin() {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.on('data', chunk => data += chunk);
    process.stdin.on('end', () => {
      try { resolve(JSON.parse(data)); }
      catch { resolve({}); }
    });
    // Safety: if no stdin arrives within 500ms, bail.
    setTimeout(() => resolve({}), 500);
  });
}

function detectProject(hookInput) {
  // Prefer explicit project name from the hook payload.
  if (hookInput.project) return hookInput.project;
  if (hookInput.cwd) return path.basename(hookInput.cwd);
  return path.basename(process.cwd());
}

// The Stop hook does NOT carry the assistant's reply in its payload - it carries
// `transcript_path`, the session's JSONL. Read the last assistant entry out of
// it. An earlier version read `hookInput.response.blocks`, a field that does not
// exist, so no install ever stored a single assistant message.
function getLastAssistantMessage(transcriptPath) {
  try {
    if (!transcriptPath || !fs.existsSync(transcriptPath)) return null;
    const lines = fs.readFileSync(transcriptPath, 'utf8').trim().split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      if (!lines[i]) continue;
      let entry;
      try { entry = JSON.parse(lines[i]); } catch { continue; }
      if (entry.type !== 'assistant' || !entry.message) continue;

      const content = entry.message.content;
      let text = '';
      if (typeof content === 'string') {
        text = content;
      } else if (Array.isArray(content)) {
        text = content.filter(c => c && c.type === 'text').map(c => c.text || '').join('\n');
      }
      text = text.trim();
      // Tool-only turns carry no text. Keep walking back rather than storing an
      // empty row for every tool call.
      if (!text) continue;
      return { text: text.slice(0, 10000), model: entry.message.model || null };
    }
  } catch (e) { log('transcript read failed:', e.message); }
  return null;
}

function extractEntry(hookInput) {
  const event = hookInput.hook_event_name || '';

  if (event === 'UserPromptSubmit' && hookInput.prompt) {
    return { role: 'user', content: String(hookInput.prompt) };
  }

  if (event === 'Stop') {
    const last = getLastAssistantMessage(hookInput.transcript_path);
    if (!last) return null;
    return { role: 'assistant', content: last.text, model: last.model };
  }

  return null;
}

function postMessage(payload) {
  return new Promise((resolve) => {
    const body = JSON.stringify(payload);
    const url  = new URL(`${ECHOES_URL}/message`);
    const lib  = url.protocol === 'https:' ? https : http;
    const headers = {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body),
    };
    if (ECHOES_TOKEN) headers['Authorization'] = `Bearer ${ECHOES_TOKEN}`;

    const req = lib.request(
      {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname,
        method: 'POST',
        headers,
        rejectUnauthorized: false,   // allow self-signed on remote deployments
      },
      (res) => {
        res.on('data', () => {});
        res.on('end', () => resolve(res.statusCode < 300));
      }
    );
    req.on('error', (e) => { log('post error:', e.message); resolve(false); });
    req.setTimeout(2000, () => { req.destroy(); resolve(false); });
    req.write(body);
    req.end();
  });
}

async function main() {
  const hookInput = await readStdin();
  const entry = extractEntry(hookInput);
  if (!entry) return;

  const payload = {
    session_id: hookInput.session_id || 'unknown',
    project:    detectProject(hookInput),
    machine:    process.env.COMPUTERNAME || process.env.HOSTNAME || null,
    role:       entry.role,
    content:    entry.content,
    model:      entry.model || null,
  };

  const ok = await postMessage(payload);
  log(entry.role, ok ? 'ok' : 'fail', payload.project);
}

main().catch(e => log('fatal:', e.message));
