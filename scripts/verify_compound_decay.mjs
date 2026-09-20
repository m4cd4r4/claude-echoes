#!/usr/bin/env node
/**
 * Decay check for the generated set's `abstention_compound` cases.
 *
 * verify_abstention_cases.mjs covers the hand-written compounds in
 * benchmarks/abstention_hard.json. The 15 generated compounds inside
 * benchmarks/eval_cases.json had no check at all, and they decay the same way:
 * a negative case stops being a negative the moment its subject enters the
 * index, and asking it in a live session is enough to put it there.
 *
 * This prints case NUMBERS, never subjects - the sibling script prints the
 * question text, which is itself a way to re-index a case you are trying to
 * protect. Look the number up in eval_cases.json when you need to replace one.
 *
 *   node scripts/verify_compound_decay.mjs                # all rows
 *   node scripts/verify_compound_decay.mjs 2026-09-19     # ignore rows from
 *                                                         # this date on, i.e.
 *                                                         # your own testing
 * Exit 1 if any case has decayed.
 */
import pg from 'pg';
import { readFileSync } from 'node:fs';

const SINCE = process.argv[2];
const DSN = process.env.ECHOES_DSN
  || 'postgresql://echoes:echoes@localhost:5434/echoes';  // secrets-guard: allow

const cases = JSON.parse(readFileSync('benchmarks/eval_cases.json', 'utf8'))
  .filter(x => x.category === 'abstention_compound');

const client = new pg.Client({ connectionString: DSN });
await client.connect();

let broken = 0;
for (const [i, c] of cases.entries()) {
  const where = c.terms.map((_, j) => `content ILIKE $${j + 1}`).join(' AND ');
  const params = c.terms.map(t => `%${t}%`);
  const since = SINCE ? ` AND created_at < $${params.length + 1}` : '';
  if (SINCE) params.push(SINCE);
  const { rows } = await client.query(
    `SELECT count(*)::int AS n FROM messages WHERE ${where}${since}`, params);
  if (rows[0].n > 0) {
    broken++;
    console.log(`  case #${i + 1}: DECAYED (${rows[0].n} matching message(s))`);
  }
}
await client.end();

console.log(`${cases.length} compound cases, ${broken} decayed`
  + (SINCE ? ` (rows from ${SINCE} on ignored)` : ''));
process.exit(broken ? 1 : 0);
