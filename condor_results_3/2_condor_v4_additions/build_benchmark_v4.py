"""Build the v4 additions benchmark. Run once on darmok (needs internet), in the v3 folder.

    python build_benchmark_v4.py --v3 benchmark_prompts_v3.csv

Creates benchmark_v4.csv. Each row lists which models must answer it (column `models`):

  experiment            rows   models
  coder3b_coding         378   coder_3B                        (2x2 size x specialization)
  gsm8k                  100   small_1.5B, large_3B, math_1.5B (second specialization pair)
  mmlu_stem_math         ~400  math_1.5B                       (math specialist on STEM QA)
  mmlu_shuffled         1000   small_1.5B, large_3B, coder_1.5B (answer-order check)
  summarizer_xsum        200   summ_1.5B                       (community summarization fine-tune)

MBPP+ and MMLU-Pro rows reuse the v3 prompt ids, so results merge with the v3 run.
"""
import argparse
import hashlib
import json
import random
import re

import numpy as np
import pandas as pd

SEED = 42
STEM = ["math", "physics", "chemistry", "engineering"]
LETTERS = "ABCDEFGHIJ"


def mmlu_prompt(question, options):
    # Identical template to the original notebook / v3.
    options_text = "\n".join(f"{chr(65 + i)}. {o.strip()}" for i, o in enumerate(options))
    return f"""Answer the following multiple-choice question.

Your entire response must be exactly one capital letter from A through J.
Do not provide an explanation or any other text.

Question:
{question}

Options:
{options_text}
"""


def gsm8k_prompt(question):
    return f"""Solve the following math problem. Show your reasoning briefly, then end with
"The final answer is <number>."

Problem:
{question}
"""


def gsm8k_answer(solution):
    return solution.split("####")[-1].strip().replace(",", "")


def split_5050(df, by, seed):
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.random(len(df)), index=df.index)
    rank = r.groupby(df[by]).rank(method="first")
    size = r.groupby(df[by]).transform("size")
    return np.where(rank <= np.ceil(size / 2), "dev", "test")


def build(v3_path, n_gsm8k, out):
    from datasets import load_dataset

    v3 = pd.read_csv(v3_path, dtype={"prompt_id": str})
    rows = []

    # 1. Coder-3B on all MBPP+ problems (same prompts and ids as v3).
    for _, r in v3[v3["task"] == "coding"].iterrows():
        rows.append({**r.to_dict(), "experiment": "coder3b_coding", "models": "coder_3B"})

    # 2. Math specialist on the STEM subjects of the v3 MMLU-Pro questions (same prompts and ids).
    for _, r in v3[(v3["task"] == "factual_qa") & v3["category"].isin(STEM)].iterrows():
        rows.append({**r.to_dict(), "experiment": "mmlu_stem_math", "models": "math_1.5B"})

    # 2b. Community summarization fine-tune on the same XSum articles (same prompts and ids).
    for _, r in v3[v3["task"] == "summarization"].iterrows():
        rows.append({**r.to_dict(), "experiment": "summarizer_xsum", "models": "summ_1.5B"})

    # 3. Same 1,000 MMLU-Pro questions with the options shuffled (answer-order check).
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_id = {f"mmlu_{ex['question_id']}": ex for ex in mmlu}
    for _, r in v3[v3["task"] == "factual_qa"].iterrows():
        ex = by_id[r["prompt_id"]]
        opts = list(ex["options"])
        perm = list(range(len(opts)))
        random.Random(f"{SEED}-{r['prompt_id']}").shuffle(perm)
        new_opts = [opts[i] for i in perm]
        old_idx = LETTERS.index(ex["answer"])
        new_ref = LETTERS[perm.index(old_idx)]
        rows.append({**r.to_dict(),
                     "prompt_id": r["prompt_id"] + "_shuf", "task": "factual_qa_shuffled",
                     "prompt": mmlu_prompt(ex["question"], new_opts), "reference": new_ref,
                     "original_reference": ex["answer"], "permutation": json.dumps(perm),
                     "experiment": "mmlu_shuffled", "models": "small_1.5B;large_3B;coder_1.5B"})

    # 4. GSM8K: grade-school math word problems (scored on the final number).
    gsm = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SEED).select(range(n_gsm8k))
    g_rows = []
    for k, ex in enumerate(gsm):
        g_rows.append({"prompt_id": f"gsm8k_{k:04d}", "task": "math", "prompt": gsm8k_prompt(ex["question"]),
                       "reference": gsm8k_answer(ex["answer"]), "category": "math_word_problem",
                       "n_options": np.nan, "in_pilot": False,
                       "experiment": "gsm8k", "models": "small_1.5B;large_3B;math_1.5B"})
    g = pd.DataFrame(g_rows)
    g["split"] = split_5050(g, "task", SEED)
    rows += g.to_dict("records")

    df = pd.DataFrame(rows)
    df["prompt_words"] = df["prompt"].str.split().str.len()
    key = "|".join(sorted(df["prompt_id"] + "::" + df["experiment"]))
    assert df.groupby("experiment")["prompt_id"].apply(lambda s: s.is_unique).all(), "duplicate prompt ids"
    df.to_csv(out, index=False)
    summary = df.groupby("experiment").agg(rows=("prompt_id", "count"), models=("models", "first"))
    summary["runs"] = summary["rows"] * summary["models"].str.count(";").add(1)
    print(summary.to_string())
    print(f"Total runs: {summary['runs'].sum()} | hash {hashlib.md5(key.encode()).hexdigest()[:12]}")
    shuf = df[df["experiment"] == "mmlu_shuffled"]
    print(f"Shuffled QA: answer letter changed for {(shuf['reference'] != shuf['original_reference']).mean():.1%} of questions")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default="benchmark_prompts_v3.csv")
    ap.add_argument("--gsm8k", type=int, default=100)
    ap.add_argument("--out", default="benchmark_v4.csv")
    a = ap.parse_args()
    build(a.v3, a.gsm8k, a.out)
