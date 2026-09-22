// Assembles the graded eval set from the judged pool.
//
// Categories follow LongMemEval. Ground truth is STRUCTURAL everywhere - the
// next substantive assistant turn in the session, or a corpus property decided
// by SQL - so no relevance judgement of ours can leak into the set. The only
// model in the loop is the self-containment judge, and it sees the question
// alone, never the answer or the ranking.
//
// usage: node scripts/build_eval_set.mjs [--n=34]
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { ageInDays, temporalDays } from './temporal_window.mjs';

const arg = (k, d) => (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=')[1] || d;
const N = Number(arg('n', 34));

const psql = (sql) => execFileSync('docker', [
  'compose', 'exec', '-T', 'postgres', 'psql', '-U', 'echoes', '-d', 'echoes', '-t', '-A', '-c', sql,
], { cwd: process.cwd(), encoding: 'utf8', maxBuffer: 1 << 28 });

const sqlLit = (s) => "'" + String(s).replace(/'/g, "''") + "'";

// Residual deictics the LLM let through. Cheap, and still gold-blind.
const STILL_BOUND = /\b(option \d|that issue|this issue|that error|the plan|as discussed|instead of that)\b/i;

const judged = readFileSync('benchmarks/eval_pool_judged.jsonl', 'utf8')
  .trim().split('\n').filter(Boolean).map(l => JSON.parse(l))
  .filter(c => c.self_contained && !STILL_BOUND.test(c.question));

const picked = judged.slice(0, N);
console.error(`cat1 pool ${judged.length}, taking ${picked.length}`);

// Window scoring: an answer often spans several assistant turns, so a hit on a
// neighbouring turn of the same session is counted separately. Strict gold stays
// the headline metric; the window figure bounds how much it understates.
const ids = picked.map(c => c.gold_id).join(',');
const winRows = psql(`
  WITH gold AS (SELECT id, session_id, created_at FROM messages WHERE id IN (${ids}))
  SELECT g.id, string_agg(m.id::text, ',') FROM gold g
  JOIN LATERAL (
    SELECT id FROM messages m2
    WHERE m2.session_id = g.session_id AND m2.role='assistant' AND m2.source='chat'
      AND m2.created_at BETWEEN g.created_at - INTERVAL '10 minutes'
                            AND g.created_at + INTERVAL '10 minutes'
    ORDER BY m2.created_at LIMIT 5
  ) m ON true
  GROUP BY g.id;`).trim().split('\n').filter(Boolean);
const windowOf = new Map(winRows.map(l => {
  const [id, list] = l.split('|');
  return [Number(id), list.split(',').map(Number)];
}));

const cases = [];

// --- Category 1: single-session fact retrieval -----------------------------
for (const c of picked) {
  cases.push({
    category: 'single_session', q: c.question,
    gold_ids: [c.gold_id], window_ids: windowOf.get(c.gold_id) || [c.gold_id],
    session_id: c.session_id, project: c.project, asked_at: c.asked_at,
  });
}

// --- Category 4: temporal reasoning ---------------------------------------
// Two directions, because a filter that silently ignores the constraint passes
// the first and fails the second.
for (const c of picked.slice(0, 10)) {
  const ageDays = ageInDays(c.asked_at);
  cases.push({
    category: 'temporal_include', q: c.question, days: temporalDays('temporal_include', ageDays),
    gold_ids: [c.gold_id], window_ids: windowOf.get(c.gold_id) || [c.gold_id],
    session_id: c.session_id, note: 'window contains the gold - it must still be found',
  });
  cases.push({
    category: 'temporal_exclude', q: c.question, days: temporalDays('temporal_exclude', ageDays),
    gold_ids: [c.gold_id], window_ids: [], expect_absent: true,
    session_id: c.session_id, note: 'window predates the gold - it must NOT be returned',
  });
}

// --- Category 2: multi-session aggregation --------------------------------
// A subject is a genuine aggregation case only if the archive discusses it in
// several DIFFERENT sessions. That is a property of the corpus, decided by SQL.
//
// Rebuilt 2026-09-19. The first version matched subject words as SUBSTRINGS
// (ILIKE, so 'test' hit 'latest'), then took the first 6 qualifying sessions in
// session_id order - a UUID sort, i.e. an arbitrary pick. Measured: subjects had
// up to 251 qualifying sessions, so a retriever returning any of the other 245
// scored zero, and multi_session read 0.056 while the returned rows were on topic.
//
// Now: whole-word match through the same english tsvector the lexical arm uses,
// only subjects discussed in 3..MS_MAX_SESSIONS sessions (above that the subject
// is too generic for "what do we know" to have a checkable answer), and gold is
// EVERY qualifying row. Scoring counts distinct gold sessions, not rows - see
// eval_graded.mjs.
const MS_MAX_SESSIONS = 12;
function buildMultiSession() {
  const out = [];
  const subjects = [...new Set(judged.map(c => (c.subject || '').trim().toLowerCase())
    .filter(s => s.split(/\s+/).length >= 2 && s.length >= 8 && /^[a-z0-9 .-]+$/.test(s)))];
  for (const subj of subjects) {
    if (out.length >= 8) break;
    const words = subj.split(/\s+/).filter(w => w.length >= 4).slice(0, 3);
    if (words.length < 2) continue;
    const tsq = `plainto_tsquery('english', ${sqlLit(words.join(' '))})`;
    let rows;
    try {
      rows = psql(`
        SELECT id, session_id FROM messages
        WHERE role='assistant' AND source='chat' AND length(content) >= 200
          AND to_tsvector('english', content) @@ ${tsq}
        ORDER BY session_id, id;`).trim();
    } catch { continue; }
    const parsed = rows ? rows.split('\n').filter(Boolean).map(l => l.split('|')) : [];
    const sessions = new Set(parsed.map(p => p[1]));
    if (sessions.size < 3 || sessions.size > MS_MAX_SESSIONS) {
      console.error(`  multi_session rejected (${sessions.size} sessions): ${subj}`);
      continue;
    }
    out.push({
      category: 'multi_session', q: `what do we know about ${subj}`,
      gold_ids: parsed.map(p => Number(p[0])),
      gold_session_of: Object.fromEntries(parsed.map(p => [p[0], p[1]])),
      gold_sessions: [...sessions],
      window_ids: parsed.map(p => Number(p[0])),
      note: `${sessions.size} distinct sessions discuss this; any of their matching rows counts`,
    });
  }
  return out;
}

// --only=multi_session rewrites just that category in the existing set, so the
// other categories (and every result measured against them) stay comparable.
if (arg('only', '') === 'multi_session') {
  const prev = JSON.parse(readFileSync('benchmarks/eval_cases.json', 'utf8'));
  const next = [...prev.filter(c => c.category !== 'multi_session'), ...buildMultiSession()];
  writeFileSync('benchmarks/eval_cases.json', JSON.stringify(next, null, 2) + '\n');
  console.error(`rewrote multi_session only: ${next.filter(c => c.category === 'multi_session').length} cases`);
  process.exit(0);
}
// --only=single_session grows (or shrinks) that category to --n. The pool order
// is fixed, so the first cases stay identical and the old run remains comparable
// on them; every other category is left exactly as it was.
if (arg('only', '') === 'single_session') {
  const prev = JSON.parse(readFileSync('benchmarks/eval_cases.json', 'utf8'));
  const single = cases.filter(c => c.category === 'single_session');
  const next = [...single, ...prev.filter(c => c.category !== 'single_session')];
  writeFileSync('benchmarks/eval_cases.json', JSON.stringify(next, null, 2) + '\n');
  console.error(`rewrote single_session only: ${single.length} cases`);
  process.exit(0);
}
cases.push(...buildMultiSession());

// --- Category 5: abstention ------------------------------------------------
// Well-formed questions about subjects the archive has never mentioned. The
// absence is VERIFIED by SQL, so a case survives only if the corpus really has
// nothing - which makes returning any row the wrong answer.
const ABSENT_CANDIDATES = [
  'the Fastly CDN migration', 'the Kubernetes rollout on the Melbourne cluster',
  'the Snowflake data warehouse contract', 'the CockroachDB failover drill',
  'the Elasticsearch reindex for the Brisbane tenant', 'the SAP integration budget',
  'the Kafka consumer lag incident', 'the Terraform state corruption on staging',
];
for (const subj of ABSENT_CANDIDATES) {
  const words = subj.replace(/^the /, '').split(/\s+/).filter(w => w.length >= 5).slice(0, 2);
  if (words.length < 2) continue;
  const conds = words.map(w => `content ILIKE ${sqlLit('%' + w + '%')}`).join(' AND ');
  let hits;
  try { hits = Number(psql(`SELECT count(*) FROM messages WHERE ${conds};`).trim()); }
  catch { continue; }
  if (hits !== 0) { console.error(`  abstention candidate rejected (${hits} corpus hits): ${subj}`); continue; }
  cases.push({
    category: 'abstention', q: `what did we decide about ${subj}`,
    gold_ids: [], window_ids: [], expect_empty: true,
    note: 'verified absent from the corpus - the correct answer is no results',
  });
}

writeFileSync('benchmarks/eval_cases.json', JSON.stringify(cases, null, 2) + '\n');
const byCat = {};
cases.forEach(c => byCat[c.category] = (byCat[c.category] || 0) + 1);
console.error('wrote benchmarks/eval_cases.json', JSON.stringify(byCat));
