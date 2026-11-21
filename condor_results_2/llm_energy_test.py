#!/usr/bin/env python3
"""Small GPU smoke test for the EEC LLM energy experiment."""

import argparse
import gc
import os
import sys
import threading
import time
import traceback

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "small_1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "large_3B": "Qwen/Qwen2.5-3B-Instruct",
}
SYSTEM_PROMPT = "Answer the user's question clearly and concisely."
MAX_NEW_TOKENS = 128
MAX_INPUT_TOKENS = 1024
POWER_SAMPLE_INTERVAL = 0.05


def init_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return pynvml, handle, str(name), None
    except Exception as exc:
        return None, None, None, str(exc)


def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_query(model, tokenizer, query, pynvml, gpu_handle):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to("cuda")

    power_samples = []
    stop_sampling = threading.Event()

    def sample_power():
        while not stop_sampling.is_set():
            try:
                watts = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
                power_samples.append((time.perf_counter(), watts))
            except Exception:
                break
            stop_sampling.wait(POWER_SAMPLE_INTERVAL)

    synchronize_gpu()
    start = time.perf_counter()
    sampler = None
    if pynvml is not None and gpu_handle is not None:
        sampler = threading.Thread(target=sample_power, daemon=True)
        sampler.start()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    synchronize_gpu()
    elapsed = time.perf_counter() - start
    if sampler is not None:
        stop_sampling.set()
        sampler.join(timeout=2)

    prompt_tokens = inputs["input_ids"].shape[1]
    generated = output[0][prompt_tokens:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    energy_j = None
    avg_power_w = None
    measured_duration = None
    if len(power_samples) >= 2:
        energy_j = 0.0
        for (t0, p0), (t1, p1) in zip(power_samples, power_samples[1:]):
            energy_j += (p0 + p1) * 0.5 * (t1 - t0)
        measured_duration = power_samples[-1][0] - power_samples[0][0]
        if measured_duration > 0:
            avg_power_w = energy_j / measured_duration
        else:
            energy_j = None

    token_count = int(generated.numel())
    return {
        "answer": answer,
        "latency_s": elapsed,
        "generated_tokens": token_count,
        "tokens_per_s": token_count / elapsed if elapsed else None,
        "gpu_energy_j": energy_j,
        "avg_gpu_power_w": avg_power_w,
        "power_samples": len(power_samples),
        "power_measurement_duration_s": measured_duration,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="wildchat_benchmark.csv")
    parser.add_argument("--output", default="llm_energy_results.csv")
    parser.add_argument("--sample-per-category", type=int, default=1)
    args = parser.parse_args()

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available!")

    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU compute capability:",
      torch.cuda.get_device_capability(0))
    print("Supported architectures:",
      torch.cuda.get_arch_list())

    # Actually execute a small GPU operation
    test = torch.randn(100, 100, device="cuda")
    result = test @ test
    torch.cuda.synchronize()

    print("CUDA operation successful!")

    print("=== GPU environment ===", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible to this job; refusing CPU fallback.")

    print(f"CUDA runtime reported by PyTorch: {torch.version.cuda}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB", flush=True)

    pynvml, gpu_handle, nvml_name, nvml_error = init_nvml()
    if pynvml is not None:
        print(f"NVML initialized; device: {nvml_name}", flush=True)
    else:
        print(f"WARNING: NVML power sampling unavailable: {nvml_error}", flush=True)

    benchmark = pd.read_csv(args.input)
    required = {"prompt", "category"}
    missing = required - set(benchmark.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    benchmark = benchmark.dropna(subset=["prompt", "category"]).copy()
    benchmark["prompt"] = benchmark["prompt"].astype(str)
    benchmark["category"] = benchmark["category"].astype(str)
    if benchmark.empty:
        raise ValueError("No rows with non-empty prompt and category were found.")

    # Deterministic smoke test: one prompt per category by default.
    benchmark = (
        benchmark.groupby("category", group_keys=False)
        .sample(n=args.sample_per_category, random_state=42)
        .reset_index(drop=True)
    )
    if "prompt_id" not in benchmark.columns:
        benchmark["prompt_id"] = [f"prompt_{i:04d}" for i in range(len(benchmark))]

    # Create the output file up front so Condor can transfer it even if a
    # model download or initialization fails before the first inference.
    pd.DataFrame(columns=[
        "prompt_id", "query_number", "category", "query", "model_name",
        "model_id", "system_prompt", "max_new_tokens", "answer",
        "latency_s", "generated_tokens", "tokens_per_s", "gpu_energy_j",
        "avg_gpu_power_w", "power_samples", "power_measurement_duration_s",
    ]).to_csv(args.output, index=False)

    print(f"Loaded {len(benchmark)} prompts across {benchmark['category'].nunique()} categories.", flush=True)
    print(benchmark["category"].value_counts().to_string(), flush=True)

    results = []
    failures = []
    for model_name, model_id in MODELS.items():
        print(f"\n{'=' * 65}\nLoading {model_name}: {model_id}", flush=True)
        model = None
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to("cuda")
            model.eval()

            # Warm-up is deliberately excluded from the recorded benchmark rows.
            _ = measure_query(
                model, tokenizer,
                "Hello! Please answer in one sentence.",
                pynvml, gpu_handle,
            )
            torch.cuda.empty_cache()

            for query_number, row in enumerate(benchmark.itertuples(index=False), start=1):
                prompt_id = row.prompt_id
                query = row.prompt
                category = row.category
                print(f"\n{model_name} | {query_number}/{len(benchmark)} | {category}", flush=True)

                measurement = measure_query(model, tokenizer, query, pynvml, gpu_handle)
                print(f"Answer: {measurement['answer']}", flush=True)
                print(
                    f"Latency: {measurement['latency_s']:.3f}s; "
                    f"tokens: {measurement['generated_tokens']}; "
                    f"tokens/s: {measurement['tokens_per_s']:.2f}",
                    flush=True,
                )
                print(f"GPU energy: {measurement['gpu_energy_j']} J", flush=True)

                results.append({
                    "prompt_id": prompt_id,
                    "query_number": query_number,
                    "category": category,
                    "query": query,
                    "model_name": model_name,
                    "model_id": model_id,
                    "system_prompt": SYSTEM_PROMPT,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    **measurement,
                })
                pd.DataFrame(results).to_csv(args.output, index=False)

        except Exception as exc:
            failures.append((model_name, repr(exc)))
            print(f"ERROR while testing {model_name}: {exc}", flush=True)
            traceback.print_exc()
            if results:
                pd.DataFrame(results).to_csv(args.output, index=False)
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    if pynvml is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    if results:
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"Saved {len(results)} result rows to {args.output}", flush=True)
    else:
        print("No inference result rows were produced.", flush=True)

    if failures:
        print("Model failures:", failures, flush=True)
        return 2
    print("GPU smoke test completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
