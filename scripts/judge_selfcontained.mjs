// Gold-blind self-containment filter.
//
// The brief assumed the ~5,400 archive questions matching "length + ?" were
// self-contained. They are not - a sample showed most are follow-ups that only
// mean anything inside their session ("what about these bullets?"). This judge
// keeps the ones a stranger could understand.
//
// It is shown the QUESTION ONLY. It never sees the gold answer, the retrieved
// rows, or the ranking, so accepting a question cannot encode any preference
// for what the retriever happens to be good at.
//
// usage: node scripts/judge_selfcontained.mjs [--in F] [--out F] [--batch N]
import { readFileSync, writeFileSync } from 'node:fs';

const arg = (k, d) => (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=')[1] || d;
const IN     = arg('in',  'benchmarks/eval_pool.jsonl');
const OUT    = arg('out', 'benchmarks/eval_pool_judged.jsonl');
const BATCH  = Number(arg('batch', 6));
const OLLAMA = arg('ollama', 'http://127.0.0.1:11435');
const MODEL  = arg('model', 'qwen2.5:7b-instruct');

const pool = readFileSync(IN, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));

const RULES = `You are filtering questions for a search benchmark.

A question is SELF_CONTAINED only if a person who never saw the conversation can tell
exactly what it is about, because the question NAMES its own subject.

Reject as CONTEXT_BOUND if it depends on something outside its own words:
- points at unstated context ("is that what we agreed", "is this right", "what about the other one")
- asks about the current session or process ("any outstanding tasks", "safe to end the session", "what's next")
- is a bare approval, preference or reaction ("sounds better, right?")
- names no concrete subject a search engine could look for

Judge the WORDING ONLY. Do not consider whether an answer exists anywhere.

Reply with ONLY a JSON object holding one entry per question, no prose. Ollama's
JSON mode emits an object, never a bare array, so the array is wrapped:
{"verdicts":[{"n":1,"verdict":"SELF_CONTAINED","subject":"<=6 words"},
             {"n":2,"verdict":"CONTEXT_BOUND","subject":""}]}`;

async function judge(batch) {
  const listing = batch.map((c, i) => `${i + 1}. ${c.question}`).join('\n');
  const r = await fetch(`${OLLAMA}/api/generate`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, stream: false, format: 'json',
      options: { temperature: 0, num_predict: 700 },
      prompt: `${RULES}\n\nQuestions:\n${listing}\n\nJSON:`,
    }),
  });
  const txt = (await r.json()).response || '';
  const m = txt.match(/\{[\s\S]*\}/);
  if (!m) throw new Error('no JSON object in response');
  const parsed = JSON.parse(m[0]);
  const list = parsed.verdicts || parsed.results || (Array.isArray(parsed) ? parsed : null);
  if (!Array.isArray(list)) throw new Error('no verdicts array');
  return list;
}

const out = [];
let kept = 0;
for (let i = 0; i < pool.length; i += BATCH) {
  const batch = pool.slice(i, i + BATCH);
  let verdicts = [];
  try { verdicts = await judge(batch); }
  catch (e) { console.error(`batch ${i} failed: ${e.message}`); continue; }
  batch.forEach((c, j) => {
    const v = verdicts.find(x => Number(x.n) === j + 1);
    if (!v) return;
    const ok = String(v.verdict).toUpperCase() === 'SELF_CONTAINED';
    if (ok) kept++;
    out.push({ ...c, self_contained: ok, subject: v.subject || '' });
  });
  process.stderr.write(`\r judged ${Math.min(i + BATCH, pool.length)}/${pool.length}  kept ${kept}`);
}
writeFileSync(OUT, out.map(o => JSON.stringify(o)).join('\n') + '\n');
console.error(`\nwrote ${OUT}  ${kept}/${out.length} self-contained`);
