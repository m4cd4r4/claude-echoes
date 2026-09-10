// Graded retrieval eval. Every expected fact was confirmed present in the
// corpus by SQL first, so a miss is a RETRIEVAL failure and never a data gap.
// Queries are deliberately in QUESTION form - that is the input shape the
// product exists to serve, and the shape that broke.
//
// usage: node scripts/eval_retrieval.mjs [baseUrl] [--limit N]
const BASE = process.argv[2]?.startsWith('http') ? process.argv[2] : 'http://127.0.0.1:8088';
const LIMIT = Number((process.argv.find(a => a.startsWith('--limit=')) || '').split('=')[1] || 5);

const CASES = [
  { q: 'why was ntfy retired and what replaced it',                     want: /telegram/i,      note: 'replacement channel' },
  { q: 'was Wix included or excluded from the site refresh service',    want: /wix/i,           note: 'the exclusion decision' },
  { q: 'how many certifications and questions does AzurePrep have',     want: /9,?637/,         note: 'the actual question count' },
  { q: 'what is the liability cap in the client agreement',             want: /liability cap/i, note: 'greater of 3mo fees or total' },
  { q: 'why did the Lighthouse bisect produce a fake regression',       want: /stale server/i,  note: 'stale server served 11 commits' },
  { q: 'what breaks Turbopack on a junctioned worktree',                want: /turbopack/i,     note: 'symlink out of filesystem root' },
  { q: 'what fixes a scraper getting 429 on every request',             want: /curl_cffi/i,     note: 'TLS fingerprinting, not rate limit' },
];

let pass = 0, rr = 0;
for (const c of CASES) {
  const url = `${BASE}/search?q=${encodeURIComponent(c.q)}&limit=${LIMIT}`;
  let hit = false, rank = 0, err = '';
  const t0 = Date.now();
  try {
    const r = await fetch(url);
    const j = await r.json();
    if (j.detail) err = JSON.stringify(j.detail).slice(0, 80);
    var T = j.timings_ms ? JSON.stringify(j.timings_ms) : '';
    globalThis.__t = T;
    const rows = j.results || [];
    rows.forEach((m, i) => { if (!hit && c.want.test(m.content || '')) { hit = true; rank = i + 1; } });
  } catch (e) { err = e.message; }
  if (hit) { pass++; rr += 1 / rank; }
  console.log(`${hit ? 'PASS' : 'FAIL'} ${hit ? '@' + rank : '  '}  ${String(Date.now()-t0).padStart(6)}ms  ${c.q}  ${globalThis.__t||''}`);
  if (!hit) console.log(`        want ${c.want} - ${c.note}${err ? ' | ERR ' + err : ''}`);
}
console.log(`\n${pass}/${CASES.length} at top-${LIMIT}   MRR ${(rr / CASES.length).toFixed(3)}`);
process.exit(pass === CASES.length ? 0 : 1);
