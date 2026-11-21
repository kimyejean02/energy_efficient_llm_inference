# Download models and perform inference with task-specific token limits
import os
import time
import threading
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# NVML power telemetry is optional: inference still runs if the cluster driver
# does not expose board-power readings to the job.
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    POWER_AVAILABLE = True
    print("NVML power monitoring enabled.")
except Exception as exc:
    pynvml = None
    gpu_handle = None
    POWER_AVAILABLE = False
    print("WARNING: NVML power monitoring unavailable")

MODELS = {
    "small_1.5B": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "tasks": {"factual_qa", "summarization", "coding"},
    },
    "large_3B": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "tasks": {"factual_qa", "summarization", "coding"},
    },
    "coder_1.5B": {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "tasks": {"coding"},
    },
    "summarizer_1.5B": {
        "model_id": "agentlans/Qwen2.5-1.5B-Instruct-Summarizer",
        "tasks": {"summarization"},
    },
}

TASK_MAX_TOKENS = {
    "factual_qa": 4,
    "summarization": 128,
    "coding": 256,
}

POWER_SAMPLE_INTERVAL = 0.05  # seconds

# Keep prompt metadata attached to every inference.
benchmark_df = pd.read_csv("benchmark_prompts.csv")

required_columns = {"prompt_id", "task", "prompt"}
missing_columns = required_columns - set(benchmark_df.columns)

if missing_columns:
    raise ValueError(
        f"benchmark_prompts.csv is missing columns: "
        f"{sorted(missing_columns)}"
    )

benchmark_df = benchmark_df.reset_index(drop=True)
benchmark_df["query_number"] = benchmark_df.index + 1

QUERIES = benchmark_df[
    ["query_number", "prompt_id", "task", "prompt"]
].to_dict("records")


def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Measure latency and GPU-board energy.
def measure_query(model, tokenizer, query, max_new_tokens=128):
    messages = [
        {
            "role": "system",
            "content": "Answer the user's question clearly and concisely."
        },
        {
            "role": "user",
            "content": query
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)
    power_samples = []
    stop_sampling = threading.Event()

    def sample_power():
        while not stop_sampling.is_set():
            try:
                watts = (
                    pynvml.nvmlDeviceGetPowerUsage(
                        gpu_handle
                    )
                    / 1000.0
                )
                power_samples.append(
                    (time.perf_counter(), watts)
                )
            except Exception:
                break
            stop_sampling.wait(
                POWER_SAMPLE_INTERVAL
            )
    sampler = None
    if POWER_AVAILABLE:
        sampler = threading.Thread(
            target=sample_power,
            daemon=True
        )
        sampler.start()
    synchronize_gpu()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    synchronize_gpu()
    elapsed = time.perf_counter() - start
    if sampler is not None:
        stop_sampling.set()
        sampler.join(timeout=2)
    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
    # Integrate sampled power over time.
    energy_j = None
    avg_power_w = None
    if len(power_samples) >= 2:
        energy_j = 0.0
        for j in range(1, len(power_samples)):
            t0, p0 = power_samples[j - 1]
            t1, p1 = power_samples[j]
            energy_j += (
                (p0 + p1) / 2
                * (t1 - t0)
            )
        measured_duration = (
            power_samples[-1][0]
            - power_samples[0][0]
        )
        if measured_duration > 0:
            avg_power_w = (
                energy_j
                / measured_duration
            )
        else:
            energy_j = None
    return {
        "answer": answer,
        "latency_s": elapsed,
        "generated_tokens": len(generated_tokens),
        "tokens_per_s": (
            len(generated_tokens) / elapsed
            if elapsed
            else None
        ),
        "gpu_energy_j": energy_j,
        "avg_gpu_power_w": avg_power_w,
        "power_samples": len(power_samples),
    }

results = []

for model_name, model_config in MODELS.items():

    model_id = model_config["model_id"]
    allowed_tasks = model_config["tasks"]

    print(f"\n{'=' * 65}")
    print(f"Loading {model_name}: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()

    # Warm-up query
    _ = measure_query(
        model,
        tokenizer,
        "Hello! Please answer in one sentence.",
        max_new_tokens=16
    )

    for i, item in enumerate(QUERIES, start=1):

        prompt_id = item["prompt_id"]
        task = item["task"]
        query = item["prompt"]
        query_number = item["query_number"]

        # Skip tasks not assigned to this model.
        if task not in allowed_tasks:
            continue

        # Use the summarizer's documented input format.
        if model_name == "summarizer_1.5B":
            query = f"Summarize:\n{query}"

        token_limit = TASK_MAX_TOKENS[task]

        print(
            f"\n{model_name} | "
            f"Query {i}/{len(QUERIES)} | "
            f"{task} | {prompt_id}"
        )

        measurement = measure_query(
            model,
            tokenizer,
            query,
            max_new_tokens=token_limit
        )

        results.append({
            "model_name": model_name,
            "model_id": model_id,
            "query_number": query_number,
            "prompt_id": prompt_id,
            "task": task,
            "query": item["prompt"],
            "token_limit": token_limit,
            **measurement,
        })

    del model, tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================================================================
# Save and verify inference results
# ================================================================
results_df = pd.DataFrame(results)

expected_runs_by_model = {
    name: int(
        benchmark_df["task"].isin(config["tasks"]).sum()
    )
    for name, config in MODELS.items()
}

expected_total_runs = sum(expected_runs_by_model.values())

assert len(results_df) == expected_total_runs, (
    f"Expected {expected_total_runs} runs, "
    f"got {len(results_df)}"
)

runs_per_model = results_df["model_name"].value_counts().to_dict()

for model_name, expected_count in expected_runs_by_model.items():
    actual_count = runs_per_model.get(model_name, 0)

    assert actual_count == expected_count, (
        f"{model_name} has {actual_count} runs; "
        f"expected {expected_count}"
    )

results_df.to_csv(
    "llm_energy_results.csv",
    index=False
)
print("\nExperiment complete.")
print(
    f"Total inference runs: "
    f"{len(results_df)}"
)
print(
    results_df["model_name"]
    .value_counts()
)
print(
    "\nSaved results to "
    "llm_energy_results.csv"
)

# Repeat MMLU-Pro inference for more stable GPU energy measurements
# 100 prompts x 5 repeats x each configured model
# Accuracy remains based on the 100 unique MMLU questions.
# These repeats are only for stabilizing the energy estimate.
MMLU_REPEATS = 5

mmlu_queries = (
    benchmark_df[
        benchmark_df["task"] == "factual_qa"
    ][
        ["prompt_id", "prompt"]
    ]
    .to_dict("records")
)

assert len(mmlu_queries) == 100, (
    f"Expected 100 MMLU prompts, got {len(mmlu_queries)}"
)

mmlu_repeat_results = []
mmlu_model_names = [
    name
    for name, config in MODELS.items()
    if "factual_qa" in config["tasks"]
]

for model_name in mmlu_model_names:
    model_id = MODELS[model_name]["model_id"]

    print(
        f"\nLoading {model_name} for repeated MMLU "
        f"energy measurements: {model_id}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )
    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    # Warm-up is not included in the repeated measurements.
    _ = measure_query(
        model,
        tokenizer,
        "Answer with exactly one capital letter: A",
        max_new_tokens=4
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for prompt_number, item in enumerate(
        mmlu_queries,
        start=1
    ):
        for repeat in range(
            1,
            MMLU_REPEATS + 1
        ):
            measurement = measure_query(
                model,
                tokenizer,
                item["prompt"],
                max_new_tokens=4
            )
            mmlu_repeat_results.append({
                "model_name": model_name,
                "model_id": model_id,
                "prompt_number": prompt_number,
                "prompt_id": item["prompt_id"],
                "repeat": repeat,
                **measurement,
            })
            energy_display = (
                f"{measurement['gpu_energy_j']:.3f} J"
                if measurement["gpu_energy_j"] is not None
                else "NA"
            )
            print(
                f"{model_name} | "
                f"MMLU {prompt_number}/100 | "
                f"repeat {repeat}/{MMLU_REPEATS} | "
                f"{energy_display}"
            )
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================================================================
# Save and summarize repeated MMLU energy measurements
# ================================================================
mmlu_repeat_df = pd.DataFrame(
    mmlu_repeat_results
)
expected_repeat_measurements = (
    100 * MMLU_REPEATS * len(mmlu_model_names)
)

assert len(mmlu_repeat_df) == expected_repeat_measurements, (
    f"Expected {expected_repeat_measurements} repeated MMLU measurements, "
    f"got {len(mmlu_repeat_df)}"
)
mmlu_repeat_df.to_csv(
    "mmlu_energy_repeats.csv",
    index=False
)

# Average the five repeats for each prompt first.
mmlu_per_prompt_energy = (
    mmlu_repeat_df
    .groupby(
        [
            "model_name",
            "prompt_id"
        ],
        as_index=False
    )
    .agg(
        mean_energy_j=(
            "gpu_energy_j",
            "mean"
        ),
        std_energy_j=(
            "gpu_energy_j",
            "std"
        ),
        mean_latency_s=(
            "latency_s",
            "mean"
        ),
        mean_generated_tokens=(
            "generated_tokens",
            "mean"
        ),
        measurements=(
            "gpu_energy_j",
            "count"
        )
    )
)
mmlu_per_prompt_energy.to_csv(
    "mmlu_energy_per_prompt.csv",
    index=False
)

# Then summarize across the 100 unique prompts for each model.
mmlu_energy_summary = (
    mmlu_per_prompt_energy
    .groupby(
        "model_name",
        as_index=False
    )
    .agg(
        average_energy_j=(
            "mean_energy_j",
            "mean"
        ),
        std_across_prompts_j=(
            "mean_energy_j",
            "std"
        ),
        average_latency_s=(
            "mean_latency_s",
            "mean"
        ),
        prompts=(
            "prompt_id",
            "count"
        )
    )
)

mmlu_energy_summary.to_csv(
    "mmlu_energy_summary.csv",
    index=False
)
print(
    "\nRepeated MMLU energy measurement complete."
)
print(
    f"Total repeated measurements: "
    f"{len(mmlu_repeat_df)}"
)
print((mmlu_energy_summary).to_string(index=False) if hasattr((mmlu_energy_summary), "to_string") else (mmlu_energy_summary))
print(
    "\nSaved raw repeats to mmlu_energy_repeats.csv"
)
print(
    "Saved per-prompt averages to mmlu_energy_per_prompt.csv"
)
print(
    "Saved model summary to mmlu_energy_summary.csv"
)

# Optional response grading. Set RUN_GRADING=1 in the job environment to run it.
if os.environ.get("RUN_GRADING", "1") == "1":
    # Grade generated responses
    # MMLU-Pro -> exact-match accuracy
    # XSum -> ROUGE-L
    # MBPP+ -> pass/fail using EvalPlus tests
    import re
    import pandas as pd
    from rouge_score import rouge_scorer

    # ================================================================
    # Reload files and refuse to grade stale/mismatched output
    # ================================================================
    results_df = pd.read_csv(
        "llm_energy_results.csv"
    )
    benchmark_df = pd.read_csv(
        "benchmark_prompts.csv"
    )
    benchmark_df = benchmark_df.reset_index(
        drop=True
    )
    benchmark_df["query_number"] = (
        benchmark_df.index + 1
    )

    EXPECTED_PROMPTS = len(benchmark_df)

    expected_runs_by_model = {
        name: int(
            benchmark_df["task"].isin(config["tasks"]).sum()
        )
        for name, config in MODELS.items()
    }

    EXPECTED_RUNS = sum(expected_runs_by_model.values())

    assert len(benchmark_df) == EXPECTED_PROMPTS, (
        f"Expected {EXPECTED_PROMPTS} benchmark prompts, "
        f"got {len(benchmark_df)}. "
        "Rerun the benchmark-loading cell."
    )
    assert len(results_df) == EXPECTED_RUNS, (
        f"Expected {EXPECTED_RUNS} inference runs, "
        f"got {len(results_df)}. "
        "Rerun the mass-data inference cell before grading."
    )

    for model_name, model_rows in results_df.groupby("model_name"):
        expected_count = expected_runs_by_model[model_name]

        assert len(model_rows) == expected_count, (
            f"{model_name} has {len(model_rows)} runs; "
            f"expected {expected_count}."
        )

    # Check that the prompt IDs/tasks saved during inference still match
    # the benchmark row assigned to that query number.
    validation_df = results_df.merge(
        benchmark_df[
            ["query_number", "prompt_id", "task"]
        ],
        on="query_number",
        how="left",
        suffixes=("_result", "_benchmark")
    )
    assert (
        validation_df["prompt_id_result"].astype(str)
        ==
        validation_df["prompt_id_benchmark"].astype(str)
    ).all(), (
        "Prompt IDs in inference results do not match "
        "benchmark_prompts.csv. Do not grade stale results."
    )
    assert (
        validation_df["task_result"].astype(str)
        ==
        validation_df["task_benchmark"].astype(str)
    ).all(), (
        "Task labels in inference results do not match "
        "benchmark_prompts.csv. Do not grade stale results."
    )

    # Merge references onto the verified inference rows.
    graded_df = results_df.merge(
        benchmark_df[
            ["query_number", "reference"]
        ],
        on="query_number",
        how="left"
    )
    graded_df["prediction"] = None
    graded_df["quality_score"] = pd.NA
    graded_df["evaluation_metric"] = None

    # ================================================================
    # 1. MMLU-Pro: exact-match multiple-choice accuracy
    # ================================================================
    def extract_mc_answer(text):
        text = str(text).strip().upper()
        match = re.search(
            r"\b([A-J])\b",
            text
        )
        if match:
            return match.group(1)
        return None
    qa_mask = (
        graded_df["task"]
        == "factual_qa"
    )
    graded_df.loc[
        qa_mask,
        "prediction"
    ] = (
        graded_df.loc[
            qa_mask,
            "answer"
        ]
        .apply(extract_mc_answer)
    )
    graded_df.loc[
        qa_mask,
        "quality_score"
    ] = (
        graded_df.loc[
            qa_mask,
            "prediction"
        ].astype(str)
        ==
        graded_df.loc[
            qa_mask,
            "reference"
        ].astype(str)
    ).astype(float)
    graded_df.loc[
        qa_mask,
        "evaluation_metric"
    ] = "accuracy"

    # ================================================================
    # 2. XSum: ROUGE-L against the human reference summary
    # ================================================================
    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    def clean_summary(text):
        text = str(text)

        text = re.sub(
            r"</?summary>",
            "",
            text,
            flags=re.IGNORECASE
        )

        return text.strip()

    def get_rouge_l(
        generated,
        reference
    ):
        score = rouge.score(
            str(reference),
            str(generated)
        )
        return score["rougeL"].fmeasure
    sum_mask = (
        graded_df["task"]
        == "summarization"
    )
    graded_df.loc[
        sum_mask,
        "quality_score"
    ] = (
        graded_df.loc[
            sum_mask
        ].apply(
            lambda row: get_rouge_l(
                clean_summary(row["answer"]),
                row["reference"]
            ),
            axis=1
        )
    )
    graded_df.loc[
        sum_mask,
        "evaluation_metric"
    ] = "ROUGE-L"

    def clean_code(text):
        """Remove only surrounding Markdown code fences from generated code."""
        text = str(text).strip()
        if text.startswith("```python"):
            text = text[len("```python"):].lstrip()
        elif text.startswith("```"):
            text = text[3:].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
        return text

    # ================================================================
    # 3. MBPP+: functional correctness on the selected 100 problems
    # ================================================================
    from evalplus.data import get_mbpp_plus
    from evalplus.evaluate import (
        get_groundtruth,
        check_correctness
    )
    from evalplus.eval import PASS
    from evalplus.eval._special_oracle import (
        MBPP_OUTPUT_NOT_NONE_TASKS
    )
    
    coding_mask = (
        graded_df["task"]
        == "coding"
    )
    coding_rows = graded_df[
        coding_mask
    ].copy()

    if len(coding_rows) > 0:
        all_mbpp = get_mbpp_plus()
        selected_task_ids = set(
            coding_rows[
                "prompt_id"
            ].astype(str)
        )
        selected_problems = {
            task_id: problem
            for task_id, problem
            in all_mbpp.items()
            if str(task_id)
            in selected_task_ids
        }
        assert (
            len(selected_problems)
            == 100
        ), (
            f"Expected 100 selected MBPP+ problems, "
            f"got {len(selected_problems)}."
        )

        # Fresh cache name for the 100-task, seed-42 selection.
        expected_output = get_groundtruth(
            selected_problems,
            "mbpp_plus_100_seed42",
            MBPP_OUTPUT_NOT_NONE_TASKS
        )

        for idx, row in coding_rows.iterrows():
            task_id = str(
                row["prompt_id"]
            )

            if task_id not in selected_problems:
                graded_df.loc[
                    idx,
                    "prediction"
                ] = "evaluation_error"
                graded_df.loc[
                    idx,
                    "quality_score"
                ] = 0.0
                graded_df.loc[
                    idx,
                    "evaluation_metric"
                ] = "MBPP+ pass@1"
                continue
            solution = clean_code(
                row["answer"]
            )

            try:
                result = check_correctness(
                    dataset="mbpp",
                    completion_id=0,
                    problem=selected_problems[
                        task_id
                    ],
                    solution=solution,
                    expected_output=expected_output[
                        task_id
                    ],
                    base_only=False,
                    fast_check=False
                )
                base_status = (
                    result["base"][0]
                )
                plus_status = (
                    result["plus"][0]
                )
                passed = (
                    base_status == PASS
                    and plus_status == PASS
                )
                graded_df.loc[
                    idx,
                    "prediction"
                ] = (
                    "pass"
                    if passed
                    else "fail"
                )
                graded_df.loc[
                    idx,
                    "quality_score"
                ] = float(passed)
                graded_df.loc[
                    idx,
                    "evaluation_metric"
                ] = "MBPP+ pass@1"

            except Exception as e:
                graded_df.loc[
                    idx,
                    "prediction"
                ] = "evaluation_error"
                graded_df.loc[
                    idx,
                    "quality_score"
                ] = 0.0
                graded_df.loc[
                    idx,
                    "evaluation_metric"
                ] = "MBPP+ pass@1"
                print(
                    f"Evaluation error for "
                    f"{row['model_name']} "
                    f"{task_id}: {e}"
                )

    # ================================================================
    # Save and summarize
    # ================================================================
    graded_df["quality_score"] = pd.to_numeric(
        graded_df["quality_score"],
        errors="coerce"
    )
    graded_df.to_csv(
        "llm_energy_results_graded.csv",
        index=False
    )
    print(
        "\nAverage quality, energy, "
        "and latency by model and task:"
    )
    summary = (
        graded_df
        .groupby(
            [
                "model_name",
                "task",
                "evaluation_metric"
            ],
            dropna=False
        )
        .agg(
            average_quality=(
                "quality_score",
                "mean"
            ),
            average_energy_j=(
                "gpu_energy_j",
                "mean"
            ),
            average_latency_s=(
                "latency_s",
                "mean"
            ),
            average_generated_tokens=(
                "generated_tokens",
                "mean"
            ),
            runs=(
                "query_number",
                "count"
            )
        )
        .reset_index()
    )
    print((summary).to_string(index=False) if hasattr((summary), "to_string") else (summary))
    summary.to_csv(
        "llm_energy_summary.csv",
        index=False
    )
    print(
        "\nSaved graded results to "
        "llm_energy_results_graded.csv"
    )
    print(
        "Saved summary to "
        "llm_energy_summary.csv"
    )
