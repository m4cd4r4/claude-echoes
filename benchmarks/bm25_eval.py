"""BM25 baseline on Echoes' own eval set (benchmarks/eval_cases.json).

Question: does plain local keyword search match Echoes' hybrid vector search?
Corpus: messages-export-2026-09-28.jsonl (Echoes `messages` table, same ids as
gold_ids). Scoring mirrors scripts/eval_graded.mjs: role=assistant on gold
cases, days window re-derived from the gold row's age, drop_self echo filter,
gold_session_of counts sessions, expect_empty / expect_absent pass rules.

BM25 has no abstain step (Echoes uses an LLM judge), so the two abstention
categories are reported but are not a like-for-like comparison.

Params are the textbook defaults (k1=1.2, b=0.75), untuned. Headline variant
declared before the run: `stem` (English stopwords + Porter), because Echoes'
lexical arm is Postgres to_tsvector('english'), which also stems.

usage: python benchmarks/bm25_eval.py
"""
import json, math, re, sys, time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy import sparse
from nltk.stem import PorterStemmer

K, K1, B = 5, 1.2, 0.75
CORPUS = 'benchmarks/messages-export-2026-09-28.jsonl'
CASES = 'benchmarks/eval_cases.json'

STOP = set("""a about above after again against all am an and any are as at be because been before
being below between both but by can did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
you your yours yourself yourselves""".split())
TOK = re.compile(r"[a-z0-9]+")
_stem = PorterStemmer().stem
_cache = {}


def stem(t):
    s = _cache.get(t)
    if s is None:
        s = _cache[t] = _stem(t)
    return s


def tokens(text, variant):
    toks = TOK.findall(text.lower())
    if variant == 'plain':
        return toks
    return [stem(t) for t in toks if t not in STOP]


def load():
    rows = []
    with open(CORPUS, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            r['ts'] = datetime.fromisoformat(r['created_at']).timestamp()
            rows.append(r)
    return rows


def build(rows, variant):
    vocab, data, ri, ci, dl = {}, [], [], [], np.zeros(len(rows))
    for d, r in enumerate(rows):
        tf = defaultdict(int)
        for t in tokens(r['content'], variant):
            tf[t] += 1
        dl[d] = sum(tf.values())
        for t, n in tf.items():
            ri.append(d); ci.append(vocab.setdefault(t, len(vocab))); data.append(n)
    tf = sparse.csr_matrix((np.array(data, float), (ri, ci)), shape=(len(rows), len(vocab)))
    df = np.bincount(ci, minlength=len(vocab))
    idf = np.log(1 + (len(rows) - df + 0.5) / (df + 0.5))
    norm = K1 * (1 - B + B * dl / dl.mean())
    tf = tf.tocoo()
    w = tf.data * (K1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
    return vocab, sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)


def age_days(ts, now):
    return max(1, math.ceil((now - ts) / 86400))


def temporal_days(cat, age):
    return max(1, age // 3) if cat == 'temporal_exclude' else age + 2


def dcg(g):
    return sum(x / math.log2(i + 2) for i, x in enumerate(g))


def run(rows, variant, cases, now):
    t0 = time.time()
    vocab, W = build(rows, variant)
    print(f'[{variant}] index {W.shape[0]} docs x {W.shape[1]} terms in {time.time()-t0:.0f}s', file=sys.stderr)
    ids = np.array([r['id'] for r in rows])
    by_id = {r['id']: i for i, r in enumerate(rows)}
    ts = np.array([r['ts'] for r in rows])
    is_asst = np.array([r['role'] == 'assistant' for r in rows])
    body = [re.sub(r'^/[a-z-]+\s+', '', (r['content'] or '').strip()).lower() for r in rows]

    stats, out = {}, []
    for c in cases:
        s = stats.setdefault(c['category'], dict(n=0, hit=0, win=0, rr=0, ndcg=0))
        cols = [vocab[t] for t in set(tokens(c['q'], variant)) if t in vocab]
        score = np.asarray(W[:, cols].sum(axis=1)).ravel() if cols else np.zeros(len(rows))
        mask = score > 0
        gold = set(c.get('gold_ids') or [])
        if gold:
            mask &= is_asst
        if c.get('days'):
            g = by_id.get(c['gold_ids'][0])
            days = temporal_days(c['category'], age_days(ts[g], now))
            mask &= ts > now - days * 86400
        cand = np.flatnonzero(mask)
        order = cand[np.argsort(-score[cand], kind='stable')]
        qn = c['q'].strip().lower()
        top = [i for i in order[:K + 20] if body[i] != qn][:K]  # drop_self
        got = [int(ids[i]) for i in top]
        s['n'] += 1

        if c.get('expect_empty') or c.get('expect_absent'):
            bad = len(got) > 0 if c.get('expect_empty') else any(i in gold for i in got)
            if not bad:
                for k in ('hit', 'win', 'rr', 'ndcg'):
                    s[k] += 1
            out.append(dict(q=c['q'], category=c['category'], pass_=not bad, returned=len(got)))
            continue

        sess_of = c.get('gold_session_of')
        seen, ranks = set(), []
        for i, gid in enumerate(got):
            if gid not in gold:
                continue
            if sess_of:
                sid = sess_of.get(str(gid), sess_of.get(gid))
                if sid in seen:
                    continue
                seen.add(sid)
            ranks.append(i + 1)
        units = len(set(sess_of.values())) if sess_of else len(gold)
        recall = len(ranks) / min(units, K) if units else 0
        if ranks:
            s['hit'] += recall; s['rr'] += 1 / ranks[0]
        in_win = any(i in set(c.get('window_ids') or []) for i in got)
        s['win'] += in_win
        ideal = dcg([1] * min(units, K))
        hs = set(ranks)
        s['ndcg'] += dcg([1 if i + 1 in hs else 0 for i in range(len(got))]) / ideal if ideal else 0
        out.append(dict(q=c['q'], category=c['category'], rank=ranks[0] if ranks else 0,
                        in_window=in_win, recall=recall, got=got))
    return stats, out


def table(variant, stats):
    print(f'\n== BM25 {variant} ==')
    print(f"{'category':<20}{'n':>4}  R@{K}    MRR  nDCG@{K}  window")
    tot = dict(n=0, hit=0, win=0, rr=0, ndcg=0)
    for cat, s in stats.items():
        for k in tot:
            tot[k] += s[k]
        print(f"{cat:<20}{s['n']:>4}  {s['hit']/s['n']:.3f}  {s['rr']/s['n']:.3f}  {s['ndcg']/s['n']:.3f}  {s['win']/s['n']:.3f}")
    print(f"{'OVERALL':<20}{tot['n']:>4}  {tot['hit']/tot['n']:.3f}  {tot['rr']/tot['n']:.3f}  {tot['ndcg']/tot['n']:.3f}  {tot['win']/tot['n']:.3f}")


if __name__ == '__main__':
    now = datetime.now(timezone.utc).timestamp()
    rows = load()
    cases = json.load(open(CASES, encoding='utf-8'))
    result = {}
    for variant in ('stem', 'plain'):
        stats, out = run(rows, variant, cases, now)
        table(variant, stats)
        result[variant] = dict(stats=stats, cases=out)
    json.dump(result, open('benchmarks/bm25_eval_2026-09-28.json', 'w', encoding='utf-8'), indent=1)
