"""Build the larger benchmark for the v3 experiment. Run once on darmok (needs internet).

    python build_benchmark.py                      # 1000 MMLU-Pro, 200 XSum, all MBPP+
    python build_benchmark.py --mmlu 500 --xsum 150

Prompt templates and sampling are IDENTICAL to the original notebook, so the
first 100 prompts of each task are exactly the original 300 (flagged in_pilot=True).
Every prompt is assigned to a 'dev' or 'test' split (50/50 within each task, and
within each MMLU-Pro subject): routing rules are fitted on dev and evaluated on test.
"""
import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd

SEED = 42


def mmlu_prompt(ex):
    options_text = "\n".join(
        f"{chr(65 + i)}. {option.strip()}" for i, option in enumerate(ex["options"])
    )
    return f"""Answer the following multiple-choice question.

Your entire response must be exactly one capital letter from A through J.
Do not provide an explanation or any other text.

Question:
{ex["question"]}

Options:
{options_text}
"""


def xsum_prompt(ex):
    return f"""Summarize the following article in exactly one sentence.
Preserve the most important factual information and do not add information
that is not present in the article.

Article:
{ex["document"]}
"""


def mbpp_prompt(problem):
    return f"""Write the complete Python solution for the following task.

Return only valid Python code.
Include the complete function definition.
Do not use Markdown code fences and do not include an explanation.

{problem["prompt"]}
"""


def build(n_mmlu, n_xsum, n_mbpp, out):
    from datasets import load_dataset
    from evalplus.data import get_mbpp_plus

    rows = []

    # MMLU-Pro: same shuffle as the original, so the first 100 are the original 100.
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=SEED).select(range(n_mmlu))
    for k, ex in enumerate(mmlu):
        rows.append({
            "prompt_id": f'mmlu_{ex["question_id"]}', "task": "factual_qa",
            "prompt": mmlu_prompt(ex), "reference": ex["answer"],
            "category": ex["category"], "n_options": len(ex["options"]), "in_pilot": k < 100,
        })

    # XSum: same filter and shuffle as the original.
    xsum = load_dataset("EdinburghNLP/xsum", split="test")
    xsum = xsum.filter(lambda x: 250 <= len(x["document"].split()) <= 600)
    xsum = xsum.shuffle(seed=SEED).select(range(n_xsum))
    for k, ex in enumerate(xsum):
        rows.append({
            "prompt_id": f'xsum_{ex["id"]}', "task": "summarization",
            "prompt": xsum_prompt(ex), "reference": ex["summary"],
            "category": "summarization", "n_options": np.nan, "in_pilot": k < 100,
        })

    # MBPP+: the original 100 (same seeded sample, same order), then the rest.
    mbpp = get_mbpp_plus()
    task_ids = sorted(mbpp.keys())
    original = random.Random(SEED).sample(task_ids, 100)
    rest = sorted(set(task_ids) - set(original))
    random.Random(SEED + 1).shuffle(rest)
    selected = original + rest
    if n_mbpp:
        selected = selected[:n_mbpp]
    for k, task_id in enumerate(selected):
        rows.append({
            "prompt_id": task_id, "task": "coding",
            "prompt": mbpp_prompt(mbpp[task_id]), "reference": task_id,
            "category": "coding", "n_options": np.nan, "in_pilot": k < 100,
        })

    df = pd.DataFrame(rows)
    assert df["prompt_id"].is_unique, "duplicate prompt ids"
    df["prompt_words"] = df["prompt"].str.split().str.len()

    # 50/50 dev/test split within each task (and each MMLU-Pro subject).
    rng = np.random.default_rng(SEED)
    df["_r"] = rng.random(len(df))
    stratum = df["task"] + "|" + df["category"].astype(str)
    rank = df.groupby(stratum)["_r"].rank(method="first")
    size = df.groupby(stratum)["_r"].transform("size")
    df["split"] = np.where(rank <= np.ceil(size / 2), "dev", "test")
    df = df.drop(columns="_r")

    df.to_csv(out, index=False)
    key = "|".join(sorted(df["prompt_id"].astype(str) + "::" + df["task"]))
    manifest = {
        "seed": SEED,
        "benchmark_hash": hashlib.md5(key.encode()).hexdigest()[:12],
        "counts": df["task"].value_counts().to_dict(),
        "split_counts": df.groupby(["task", "split"]).size().rename("n").reset_index().to_dict("records"),
        "pilot_prompts": int(df["in_pilot"].sum()),
    }
    with open(out.replace(".csv", "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return df


def check_against_original(df, original_csv):
    """The first 100 prompts of each task must equal the original benchmark exactly."""
    orig = pd.read_csv(original_csv)
    new = df[df["in_pilot"]].set_index("prompt_id")
    missing = set(orig["prompt_id"]) - set(new.index)
    assert not missing, f"{len(missing)} original prompts missing, e.g. {list(missing)[:3]}"
    diff = [pid for pid, p in zip(orig["prompt_id"], orig["prompt"]) if new.loc[pid, "prompt"] != p]
    assert not diff, f"{len(diff)} original prompts changed, e.g. {diff[:3]}"
    print(f"OK: all {len(orig)} original prompts are included unchanged.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu", type=int, default=1000)
    ap.add_argument("--xsum", type=int, default=200)
    ap.add_argument("--mbpp", type=int, default=0, help="0 = all MBPP+ problems")
    ap.add_argument("--out", default="benchmark_prompts_v3.csv")
    ap.add_argument("--original", default="benchmark_prompts.csv",
                    help="original 300-prompt file to check against (skipped if missing)")
    a = ap.parse_args()
    df = build(a.mmlu, a.xsum, a.mbpp, a.out)
    import os
    if os.path.exists(a.original):
        check_against_original(df, a.original)
