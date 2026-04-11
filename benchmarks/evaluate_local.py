#!/usr/bin/env python3
"""
Local judge for LongMemEval using Ollama.

Drop-in replacement for official evaluate_qa.py but uses a local Ollama model
instead of OpenAI. Scores won't match GPT-4o-mini exactly, but the RELATIVE
delta between runs is what matters for iteration.

Usage:
    python evaluate_local.py results/haiku_local.jsonl data/longmemeval_s_cleaned.json
    python evaluate_local.py results/haiku_local.jsonl data/longmemeval_s_cleaned.json --model qwen2.5:7b
    python evaluate_local.py results/haiku_local.jsonl data/longmemeval_s_cleaned.json --limit 50
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
WORKERS = 4  # local model is single-threaded on GPU, but overlap helps


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    """Exact same prompts as the official LongMemEval evaluate_qa.py."""
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        else:
            raise NotImplementedError(f"Unknown task: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        return template.format(question, answer, response)


def judge_one(model: str, prompt: str, attempts: int = 2) -> str:
    """Ask the local LLM to judge yes/no."""
    for i in range(attempts):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10},
                },
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
        except Exception:
            time.sleep(1)
    return ""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("hyp_file", help="JSONL file with hypothesis answers")
    ap.add_argument("ref_file", help="JSON file with reference questions/answers")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for judging")
    ap.add_argument("--limit", type=int, help="Only judge first N entries")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-question output")
    args = ap.parse_args()

    # Load data
    try:
        hypotheses = [json.loads(line) for line in open(args.hyp_file, encoding="utf-8")]
    except Exception:
        hypotheses = json.load(open(args.hyp_file, encoding="utf-8"))

    try:
        references = json.load(open(args.ref_file, encoding="utf-8"))
    except Exception:
        references = [json.loads(line) for line in open(args.ref_file, encoding="utf-8")]

    qid2qdata = {e["question_id"]: e for e in references}
    qid2qtype = {e["question_id"]: e["question_type"] for e in references}
    qtypes = set(qid2qtype.values())
    qtype2acc = {t: [] for t in qtypes}

    if args.limit:
        hypotheses = hypotheses[:args.limit]

    result_file = args.hyp_file + f".eval-results-local-{args.model.replace(':', '-')}"

    t0 = time.time()
    done = 0
    logs = []

    with open(result_file, "w", encoding="utf-8") as out_f:
        for entry in hypotheses:
            qid = entry["question_id"]
            if qid not in qid2qtype:
                continue

            qtype = qid2qtype[qid]
            q = qid2qdata[qid]["question"]
            ans = qid2qdata[qid]["answer"]
            hyp = entry.get("hypothesis", "")

            prompt = get_anscheck_prompt(qtype, q, ans, hyp,
                                        abstention="_abs" in qid)
            eval_response = judge_one(args.model, prompt)
            label = "yes" in eval_response.lower()

            entry["autoeval_label"] = {"model": args.model, "label": label}
            logs.append(entry)
            print(json.dumps(entry), file=out_f)
            qtype2acc[qtype].append(1 if label else 0)

            done += 1
            if not args.quiet and done % 25 == 0:
                rate = done / max(time.time() - t0, 0.001)
                print(f"  {done}/{len(hypotheses)}  {rate:.1f}/s", flush=True)

    overall = round(np.mean([1 if x["autoeval_label"]["label"] else 0
                            for x in logs]).item(), 4)
    print(f"\nAccuracy: {overall}")
    for k, v in sorted(qtype2acc.items()):
        if v:
            print(f"\t{k}: {round(np.mean(v), 4)} ({len(v)})")

    print(f"\nSaved to {result_file}")
    print(f"Judge model: {args.model} (local)")
    print(f"NOTE: absolute scores differ from GPT-4o-mini. Use deltas only.")


if __name__ == "__main__":
    main()
