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

function extractEntry(hookInput) {
  const event = hookInput.hook_event_name || '';

  if (event === 'UserPromptSubmit' && hookInput.prompt) {
    return { role: 'user', content: String(hookInput.prompt) };
  }

  if (event === 'Stop') {
    const blocks = (hookInput.response && hookInput.response.blocks) || [];
    const text = blocks
      .filter(b => b && b.type === 'text')
      .map(b => b.text || '')
      .join('\n')
      .trim();
    if (!text) return null;
    const model = (hookInput.response && hookInput.response.model) || null;
    return { role: 'assistant', content: text, model };
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
