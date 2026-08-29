// Graded retrieval eval over the generated case set.
//
// Reports Recall@k, MRR and nDCG@k per LongMemEval category. Two numbers are
// deliberately reported side by side:
//   strict  - the exact structural gold turn was retrieved
//   window  - any assistant turn from the same stretch of that session
// Strict is the headline. Window bounds how much strict understates, because a
// long answer spans several turns and only one of them is nominated gold.
//
// usage: node scripts/eval_graded.mjs [baseUrl] [--k=5] [--rerank=1] [--cases=F]
import { readFileSync } from 'node:fs';

const arg = (k, d) => (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=')[1] ?? d;
const BASE   = process.argv[2]?.startsWith('http') ? process.argv[2] : 'http://127.0.0.1:8088';
const K      = Number(arg('k', 5));
const RERANK = arg('rerank', '1') !== '0';
const CASES  = JSON.parse(readFileSync(arg('cases', 'benchmarks/eval_cases.json'), 'utf8'));

const dcg = (gains) => gains.reduce((s, g, i) => s + g / Math.log2(i + 2), 0);

const stats = {};
const bump = (cat) => (stats[cat] ??= { n: 0, hit: 0, win: 0, rr: 0, ndcg: 0, ms: 0 });

for (const c of CASES) {
  const s = bump(c.category);
  s.n++;
  const params = new URLSearchParams({ q: c.q, limit: String(K) });
  if (!RERANK) params.set('rerank', 'false');
  if (c.days) params.set('days', String(c.days));

  const t0 = Date.now();
  let rows = [], err = '';
  try {
    const j = await (await fetch(`${BASE}/search?${params}`)).json();
    rows = j.results || [];
    if (j.detail) err = JSON.stringify(j.detail).slice(0, 90);
  } catch (e) { err = e.message; }
  s.ms += Date.now() - t0;

  const ids = rows.map(r => Number(r.id));
  const gold = new Set(c.gold_ids || []);
  const win  = new Set(c.window_ids || []);

  if (c.expect_empty || c.expect_absent) {
    // Abstention and the exclude half of temporal: success is returning nothing
    // relevant. expect_empty demands zero rows outright.
    const bad = c.expect_empty ? rows.length > 0 : ids.some(i => gold.has(i));
    if (!bad) { s.hit++; s.win++; s.rr += 1; s.ndcg += 1; }
    console.log(`${bad ? 'FAIL' : 'PASS'}  ${c.category.padEnd(17)} ${String(rows.length).padStart(2)} rows  ${c.q.slice(0, 68)}${err ? ' | ERR ' + err : ''}`);
    continue;
  }

  const hitRanks = ids.map((id, i) => (gold.has(id) ? i + 1 : 0)).filter(Boolean);
  const found = hitRanks.length;
  const recall = gold.size ? found / Math.min(gold.size, K) : 0;
  const firstRank = hitRanks[0] || 0;

  if (found) { s.hit += recall; s.rr += 1 / firstRank; }
  if (ids.some(i => win.has(i))) s.win++;
  const ideal = dcg(Array(Math.min(gold.size, K)).fill(1));
  s.ndcg += ideal ? dcg(ids.map(id => (gold.has(id) ? 1 : 0))) / ideal : 0;

  const mark = found ? `PASS @${firstRank}` : (ids.some(i => win.has(i)) ? 'WIN  ~ ' : 'FAIL   ');
  console.log(`${mark} ${c.category.padEnd(17)} ${String(rows.length).padStart(2)} rows  ${c.q.slice(0, 68)}${err ? ' | ERR ' + err : ''}`);
}

console.log(`\n${'category'.padEnd(18)} ${'n'.padStart(3)}  ${'Recall@'+K}  ${'MRR'.padStart(6)}  ${('nDCG@'+K).padStart(7)}  ${'window'.padStart(6)}  ${'ms'.padStart(6)}`);
let tot = { n: 0, hit: 0, win: 0, rr: 0, ndcg: 0, ms: 0 };
for (const [cat, s] of Object.entries(stats)) {
  for (const k of Object.keys(tot)) tot[k] += s[k];
  console.log(`${cat.padEnd(18)} ${String(s.n).padStart(3)}  ${(s.hit/s.n).toFixed(3).padStart(8)}  ${(s.rr/s.n).toFixed(3)}  ${(s.ndcg/s.n).toFixed(3).padStart(7)}  ${(s.win/s.n).toFixed(3).padStart(6)}  ${String(Math.round(s.ms/s.n)).padStart(6)}`);
}
console.log(`${'OVERALL'.padEnd(18)} ${String(tot.n).padStart(3)}  ${(tot.hit/tot.n).toFixed(3).padStart(8)}  ${(tot.rr/tot.n).toFixed(3)}  ${(tot.ndcg/tot.n).toFixed(3).padStart(7)}  ${(tot.win/tot.n).toFixed(3).padStart(6)}  ${String(Math.round(tot.ms/tot.n)).padStart(6)}`);
