// Graded retrieval eval over the generated case set.
//
// Reports Recall@k, MRR and nDCG@k per LongMemEval category. Two numbers are
// deliberately reported side by side:
//   strict  - the exact structural gold turn was retrieved
//   window  - any assistant turn from the same stretch of that session
// Strict is the headline. Window bounds how much strict understates, because a
// long answer spans several turns and only one of them is nominated gold.
//
// Every request retries: this box wedges Docker under host memory pressure, and
// a run that dies at case 30 of 67 has measured nothing. A case that still fails
// after the retries is recorded as an ERROR and excluded from the denominator,
// so a dead stack can never be silently scored as a retrieval miss.
//
// usage: node scripts/eval_graded.mjs [baseUrl] [--k=5] [--rerank=1] [--cases=F]
import { readFileSync, writeFileSync } from 'node:fs';

const arg = (k, d) => (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=')[1] ?? d;
const BASE   = process.argv[2]?.startsWith('http') ? process.argv[2] : 'http://127.0.0.1:8088';
const K      = Number(arg('k', 5));
const RERANK = arg('rerank', '1') !== '0';
const OUT    = arg('out', '');
const CASES  = JSON.parse(readFileSync(arg('cases', 'benchmarks/eval_cases.json'), 'utf8'));

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const dcg = (gains) => gains.reduce((s, g, i) => s + g / Math.log2(i + 2), 0);

async function search(params) {
  let lastErr = '';
  for (let attempt = 1; attempt <= 4; attempt++) {
    try {
      const r = await fetch(`${BASE}/search?${params}`, { signal: AbortSignal.timeout(120000) });
      const j = await r.json();
      if (j.detail) { lastErr = JSON.stringify(j.detail).slice(0, 90); }
      else return { rows: j.results || [], err: '' };
    } catch (e) { lastErr = e.message; }
    // Back off and let the stack come back up before giving the case away.
    await sleep(attempt * 5000);
  }
  return { rows: [], err: lastErr };
}

const stats = {};
const bump = (cat) => (stats[cat] ??= { n: 0, hit: 0, win: 0, rr: 0, ndcg: 0, ms: 0, err: 0 });
const rowsOut = [];

for (const c of CASES) {
  const s = bump(c.category);
  const params = new URLSearchParams({ q: c.q, limit: String(K) });
  if (!RERANK) params.set('rerank', 'false');
  if (c.days) params.set('days', String(c.days));

  const t0 = Date.now();
  const { rows, err } = await search(params);
  const ms = Date.now() - t0;

  if (err) {
    s.err++;
    console.log(`ERROR  ${c.category.padEnd(17)}  ${c.q.slice(0, 62)} | ${err}`);
    rowsOut.push({ ...c, error: err });
    continue;
  }
  s.n++; s.ms += ms;

  const ids = rows.map(r => Number(r.id));
  const gold = new Set(c.gold_ids || []);
  const win  = new Set(c.window_ids || []);

  if (c.expect_empty || c.expect_absent) {
    const bad = c.expect_empty ? rows.length > 0 : ids.some(i => gold.has(i));
    if (!bad) { s.hit++; s.win++; s.rr += 1; s.ndcg += 1; }
    console.log(`${bad ? 'FAIL' : 'PASS'}   ${c.category.padEnd(17)} ${String(rows.length).padStart(2)} rows  ${c.q.slice(0, 62)}`);
    rowsOut.push({ ...c, returned: rows.length, pass: !bad });
    continue;
  }

  const hitRanks = ids.map((id, i) => (gold.has(id) ? i + 1 : 0)).filter(Boolean);
  const recall = gold.size ? hitRanks.length / Math.min(gold.size, K) : 0;
  const firstRank = hitRanks[0] || 0;

  if (hitRanks.length) { s.hit += recall; s.rr += 1 / firstRank; }
  const inWindow = ids.some(i => win.has(i));
  if (inWindow) s.win++;
  const ideal = dcg(Array(Math.min(gold.size, K)).fill(1));
  s.ndcg += ideal ? dcg(ids.map(id => (gold.has(id) ? 1 : 0))) / ideal : 0;

  const mark = firstRank ? `PASS @${firstRank}` : (inWindow ? 'WIN  ~ ' : 'FAIL   ');
  console.log(`${mark} ${c.category.padEnd(17)} ${String(rows.length).padStart(2)} rows  ${c.q.slice(0, 62)}`);
  rowsOut.push({ ...c, rank: firstRank, in_window: inWindow, recall });
}

console.log(`\n${'category'.padEnd(18)} ${'n'.padStart(3)} ${'err'.padStart(4)}  Recall@${K}     MRR  nDCG@${K}  window      ms`);
const tot = { n: 0, hit: 0, win: 0, rr: 0, ndcg: 0, ms: 0, err: 0 };
for (const [cat, s] of Object.entries(stats)) {
  for (const k of Object.keys(tot)) tot[k] += s[k];
  if (!s.n) { console.log(`${cat.padEnd(18)} ${String(s.n).padStart(3)} ${String(s.err).padStart(4)}   (no scored cases)`); continue; }
  console.log(`${cat.padEnd(18)} ${String(s.n).padStart(3)} ${String(s.err).padStart(4)}   ${(s.hit/s.n).toFixed(3)}   ${(s.rr/s.n).toFixed(3)}   ${(s.ndcg/s.n).toFixed(3)}   ${(s.win/s.n).toFixed(3)}  ${String(Math.round(s.ms/s.n)).padStart(6)}`);
}
if (tot.n) console.log(`${'OVERALL'.padEnd(18)} ${String(tot.n).padStart(3)} ${String(tot.err).padStart(4)}   ${(tot.hit/tot.n).toFixed(3)}   ${(tot.rr/tot.n).toFixed(3)}   ${(tot.ndcg/tot.n).toFixed(3)}   ${(tot.win/tot.n).toFixed(3)}  ${String(Math.round(tot.ms/tot.n)).padStart(6)}`);
if (tot.err) console.log(`\n${tot.err} case(s) errored after retries and are EXCLUDED from every figure above.`);
if (OUT) writeFileSync(OUT, JSON.stringify(rowsOut, null, 2) + '\n');
