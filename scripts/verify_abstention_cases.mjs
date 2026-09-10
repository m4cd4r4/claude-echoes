#!/usr/bin/env node
/**
 * Verify that every compositional abstention case is still ABSENT from the
 * corpus, then optionally score the server's abstention on them.
 *
 * WHY THIS EXISTS. A negative test case is only a negative while the subject is
 * genuinely missing, and asking the question in a live session PUTS IT IN THE
 * INDEX. Measured 2026-09-10: one negative case submission' went from 0 matching
 * messages to 1 within an hour of being used as a test, because the hook indexed
 * the test itself. A decayed case scores as a model failure and reads as a
 * regression - the same class of fault as the temporal windows that rot against
 * a moving now().
 *
 * So: verify first, score second, and treat a decayed case as a BROKEN TEST
 * rather than as evidence about the model.
 *
 *   node scripts/verify_abstention_cases.mjs              # verify only
 *   node scripts/verify_abstention_cases.mjs --score      # verify, then measure
 *   node scripts/verify_abstention_cases.mjs --since 2026-09-10   # ignore rows
 *                                                          # from this date on
 *
 * --since is how you exclude the pollution your own testing created: pass the
 * date you started testing and the check asks whether the subject was absent
 * BEFORE you went near it.
 */
import { readFileSync } from 'node:fs';
let pg;
try {
  pg = (await import('pg')).default;
} catch {
  console.error("this script needs the 'pg' client: npm install");
  process.exit(2);
}

const arg = (k, d) => (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=')[1] ?? d;
const SCORE = process.argv.includes('--score');
const SINCE = arg('since', '');
const BASE  = process.env.ECHOES_URL || 'http://127.0.0.1:8088';
const DSN   = process.env.ECHOES_DB_DSN
  || 'postgresql://echoes:echoes@localhost:5434/echoes';  // secrets-guard: allow

const CASES_FILE = arg('cases', 'benchmarks/abstention_hard.json');
let cases;
try {
  ({ cases } = JSON.parse(readFileSync(CASES_FILE, 'utf8')));
} catch {
  console.error(`no ${CASES_FILE}. Copy benchmarks/abstention_hard.example.json to it`);
  console.error('and fill it with compounds from YOUR corpus - both components real,');
  console.error('the combination absent. It is gitignored because it names real subjects.');
  process.exit(2);
}

const client = new pg.Client({ connectionString: DSN });
await client.connect();

let broken = 0, abstained = 0, scored = 0;
console.log(`${cases.length} compositional abstention cases`);
if (SINCE) console.log(`ignoring rows created on/after ${SINCE}\n`);

for (const c of cases) {
  const where = c.terms.map((_, i) => `content ILIKE $${i + 1}`).join(' AND ');
  const params = c.terms.map(t => `%${t}%`);
  const sinceSql = SINCE ? ` AND created_at < $${params.length + 1}` : '';
  if (SINCE) params.push(SINCE);

  const { rows } = await client.query(
    `SELECT count(*)::int AS n FROM messages WHERE ${where}${sinceSql}`, params);
  const n = rows[0].n;

  if (n > 0) {
    broken++;
    console.log(`  DECAYED  ${n} match(es)  ${c.q}`);
    console.log(`           terms: ${c.terms.join(' + ')} - this is no longer a negative case`);
    continue;
  }
  console.log(`  absent   ${c.q}`);

  if (SCORE) {
    const u = new URL('/search', BASE);
    u.searchParams.set('q', c.q);
    u.searchParams.set('limit', '5');
    try {
      const r = await fetch(u, { signal: AbortSignal.timeout(120000) });
      const d = await r.json();
      scored++;
      if (d.abstained) { abstained++; console.log(`           -> ABSTAINED (correct)`); }
      else console.log(`           -> answered with ${d.count} rows (fabrication)`);
    } catch (e) {
      console.log(`           -> ERROR ${e.name} (excluded from the score)`);
    }
  }
}

await client.end();

console.log();
if (broken) {
  console.log(`${broken} of ${cases.length} cases have DECAYED and must be replaced.`);
  console.log('Pick a new compound whose components are both real and verify it reads 0.');
}
if (SCORE && scored) {
  console.log(`compositional abstention: ${abstained}/${scored} = ${(abstained / scored).toFixed(3)}`);
}
// Non-zero on decay: a silently-rotten negative set is the failure this guards.
process.exit(broken ? 1 : 0);
