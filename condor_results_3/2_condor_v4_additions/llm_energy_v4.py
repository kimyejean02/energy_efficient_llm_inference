"""Energy-efficient LLM inference: v4 additions (Coder-3B, math specialist, GSM8K, shuffled QA).

Stages (set STAGES to run a subset, e.g. STAGES=grade,router):
  infer    every model answers every prompt (energy, latency, tokens, confidence)
  repeats  5x repeated MMLU energy on the original 100 questions (+ block-level energy)
  grade    accuracy / ROUGE-L / MBPP+ pass@1 for every run
  router   fit routing policies on the dev split, evaluate on the test split (CPU only)
  live     actually run the fitted cascade router on the test split and measure it

Environment variables:
  BENCHMARK   benchmark CSV (default benchmark_prompts_v3.csv, from build_benchmark.py)
  OUT_DIR     where result files go (default results)
  CKPT_DIR    where resumable checkpoints go (default OUT_DIR; point it at scratch)
  SMOKE_TEST  1 = 2 prompts per task per split, 2 repeats: a quick end-to-end test
  STAGES      comma-separated subset of the stages above
"""
import gc
import hashlib
import json
import os
import random
import re
import threading
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- configuration
SEED = 42
BENCHMARK = os.environ.get("BENCHMARK", "benchmark_v4.csv")
OUT_DIR = os.environ.get("OUT_DIR", "results")
CKPT_DIR = os.environ.get("CKPT_DIR", OUT_DIR)
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
STAGES = [s.strip() for s in os.environ.get("STAGES", "infer,grade").split(",") if s.strip()]
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

MODELS = {
    "small_1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "large_3B": "Qwen/Qwen2.5-3B-Instruct",
    "coder_1.5B": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "coder_3B": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "math_1.5B": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # Community fine-tune (not from Qwen); runs last so a problem with it can't block the others.
    "summ_1.5B": "agentlans/Qwen2.5-1.5B-Instruct-Summarizer",
}
FALLBACK_CHAT_TEMPLATE_FROM = "Qwen/Qwen2.5-1.5B-Instruct"
LARGE = "large_3B"
TASKS = ["factual_qa", "summarization", "coding"]
TASK_MAX_TOKENS = {"factual_qa": 4, "factual_qa_shuffled": 4, "summarization": 128,
                   "coding": 256, "math": 512}   # QA/summ/coding same as the paper
SYSTEM_PROMPT = "Answer the user's question clearly and concisely."       # same as the paper
# Input format documented on the agentlans Summarizer model card ("Summarize:\n{text}").
# Same prefix the collaborator's run used, so overlapping XSum outputs can be compared directly.
MODEL_PROMPT_PREFIX = {"summ_1.5B": "Summarize:\n"}
# Qwen2.5-Math always reasons before answering, so 4 tokens can't hold its answer. It gets room
# to reason on QA, and its letter is read from \boxed{X} / "the answer is X" (see extract_letter_reasoned).
MODEL_TOKEN_OVERRIDE = {("math_1.5B", "factual_qa"): int(os.environ.get("MATH_QA_TOKENS", "512"))}


def token_limit(name, task):
    return MODEL_TOKEN_OVERRIDE.get((name, task), TASK_MAX_TOKENS[task])
POWER_SAMPLE_INTERVAL = 0.05
MAX_INPUT_TOKENS = 1024        # inputs are no longer truncated; longer prompts are only flagged
MMLU_REPEATS = 2 if SMOKE_TEST else 5
CHECKPOINT_EVERY = 25
# Router quality target: within this much of the 3B model's dev-split quality (metric units).
TOLERANCE = {"factual_qa": 0.02, "summarization": 0.01, "coding": 0.02}


def out(name):
    return os.path.join(OUT_DIR, name)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- GPU power (NVML)
def _to_str(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def find_assigned_gpu():
    """NVML ignores CUDA_VISIBLE_DEVICES, so match the GPU Condor assigned by UUID/index."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    n = pynvml.nvmlDeviceGetCount()
    devices = []
    for i in range(n):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        devices.append((i, h, _to_str(pynvml.nvmlDeviceGetUUID(h))))
    log(f"CUDA_VISIBLE_DEVICES={visible!r} | _CONDOR_AssignedGPUs="
        f"{os.environ.get('_CONDOR_AssignedGPUs', '')!r} | NVML sees {n} GPU(s)")
    for i, _, uuid in devices:
        log(f"  NVML index {i}: {uuid}")
    if visible.startswith(("GPU-", "MIG-")):
        matches = [d for d in devices if d[2].startswith(visible)]
        if len(matches) == 1:
            i, h, uuid = matches[0]
            return h, f"NVML index {i} ({uuid}), matched by UUID"
    elif visible.isdigit() and int(visible) < n:
        i, h, uuid = devices[int(visible)]
        return h, f"NVML index {i} ({uuid}), matched by index"
    if n == 1:
        return devices[0][1], f"NVML index 0 ({devices[0][2]}), only GPU on the machine"
    raise RuntimeError(f"Could not tell which of {n} GPUs is assigned (CUDA_VISIBLE_DEVICES={visible!r}).")


try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle, MONITORED_GPU = find_assigned_gpu()
    POWER_AVAILABLE = True
    log(f"NVML power monitoring enabled on: {MONITORED_GPU}")
except Exception as exc:
    gpu_handle, MONITORED_GPU, POWER_AVAILABLE = None, None, False
    log(f"WARNING: NVML power monitoring unavailable: {exc}")

GPU_STAGES = {"infer"}
if (torch.cuda.is_available() and not POWER_AVAILABLE and GPU_STAGES & set(STAGES)
        and os.environ.get("REQUIRE_POWER", "1") == "1"):
    raise SystemExit("Stopping: GPU power can't be measured (set REQUIRE_POWER=0 to run anyway).")

ENERGY_COUNTER_AVAILABLE = False
if POWER_AVAILABLE:
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
        ENERGY_COUNTER_AVAILABLE = True
    except Exception:
        pass
log(f"Hardware energy counter available: {ENERGY_COUNTER_AVAILABLE}")


def read_energy_counter_j():
    if not ENERGY_COUNTER_AVAILABLE:
        return None
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle) / 1000.0


def read_power_w():
    return pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0


def gpu_temp_c():
    try:
        return pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return None


def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_idle_power(seconds=10.0):
    """Median board power with the model loaded but idle (for idle-power subtraction)."""
    if not POWER_AVAILABLE:
        return None
    synchronize_gpu()
    samples, end = [], time.perf_counter() + seconds
    while time.perf_counter() < end:
        samples.append(read_power_w())
        time.sleep(0.1)
    return float(np.median(samples))


class PowerLogger:
    """Background power sampling over a whole block of requests (block-level energy)."""

    def __init__(self, interval=POWER_SAMPLE_INTERVAL):
        self.interval, self.samples = interval, []
        self._stop, self._thread = threading.Event(), None

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.perf_counter(), read_power_w()))
            except Exception:
                break
            self._stop.wait(self.interval)

    def start(self):
        synchronize_gpu()
        self.e0 = read_energy_counter_j()
        if POWER_AVAILABLE:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self.t0 = time.perf_counter()
        return self

    def stop(self):
        synchronize_gpu()
        self.t1 = time.perf_counter()
        e1 = read_energy_counter_j()
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
        energy = None
        if len(self.samples) >= 2:
            energy = sum((self.samples[j - 1][1] + self.samples[j][1]) / 2
                         * (self.samples[j][0] - self.samples[j - 1][0]) for j in range(1, len(self.samples)))
        return {"block_seconds": self.t1 - self.t0, "block_energy_j": energy,
                "block_energy_counter_j": (e1 - self.e0) if e1 is not None and self.e0 is not None else None,
                "block_power_samples": len(self.samples)}


# ---------------------------------------------------------------- models
def load_model(name):
    model_id = MODELS[name]
    log(f"\n{'=' * 65}\nLoading {name}: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if not getattr(tok, "chat_template", None):
        tok.chat_template = AutoTokenizer.from_pretrained(FALLBACK_CHAT_TEMPLATE_FROM).chat_template
        log(f"{name}: no chat template in its tokenizer; using the {FALLBACK_CHAT_TEMPLATE_FROM} template")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    if POWER_AVAILABLE and torch.cuda.is_available():
        used = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1024**3
        alloc = torch.cuda.memory_allocated() / 1024**3
        log(f"GPU check ({name}): NVML used {used:.2f} GB on monitored GPU, PyTorch allocated "
            f"{alloc:.2f} GB -> {'OK' if used >= 0.9 * alloc else 'MISMATCH: wrong GPU?'}")
    for _ in range(3):                                   # warm-up, not recorded
        measure_query(model, tok, "Hello! Please answer in one sentence.", 16)
    return model, tok


def measure_query(model, tok, query, max_new_tokens):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    n_in = int(enc["input_ids"].shape[1])

    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                samples.append((time.perf_counter(), read_power_w()))
            except Exception:
                break
            stop.wait(POWER_SAMPLE_INTERVAL)

    sampler = None
    if POWER_AVAILABLE:
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
    synchronize_gpu()
    e0 = read_energy_counter_j()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tok.eos_token_id,
                              return_dict_in_generate=True, output_scores=True)
    synchronize_gpu()
    elapsed = time.perf_counter() - t0
    e1 = read_energy_counter_j()
    if sampler is not None:
        stop.set()
        sampler.join(timeout=2)

    gen = out_.sequences[0][n_in:]
    answer = tok.decode(gen, skip_special_tokens=True)
    # Token log-probabilities of the generated answer: the router's confidence signal.
    lp = model.compute_transition_scores(out_.sequences, out_.scores, normalize_logits=True)[0]
    lp = np.asarray(lp.float().cpu().numpy() if hasattr(lp, "float") else lp, dtype=float)
    lp = lp[np.isfinite(lp)]

    e_trap = None
    if len(samples) >= 2:
        e_trap = sum((samples[j - 1][1] + samples[j][1]) / 2 * (samples[j][0] - samples[j - 1][0])
                     for j in range(1, len(samples)))
    mean_p = float(np.mean([w for _, w in samples])) if samples else None
    return {
        "answer": answer,
        "input_tokens": n_in,
        "input_over_limit": n_in > MAX_INPUT_TOKENS,
        "generated_tokens": int(len(gen)),
        "hit_token_limit": int(len(gen)) >= max_new_tokens,
        "latency_s": elapsed,
        "tokens_per_s": len(gen) / elapsed if elapsed else None,
        "gpu_energy_j": e_trap,                                   # trapezoid, eq. (1)
        "gpu_energy_avgp_latency_j": mean_p * elapsed if mean_p is not None else None,   # eq. (2)
        "gpu_energy_counter_j": (e1 - e0) if e0 is not None and e1 is not None else None,
        "avg_gpu_power_w": mean_p,
        "power_samples": len(samples),
        "mean_logprob": float(lp.mean()) if len(lp) else float("nan"),
        "first_logprob": float(lp[0]) if len(lp) else float("nan"),
        "gpu_temp_c": gpu_temp_c() if POWER_AVAILABLE else None,
    }



# ---------------------------------------------------------------- graders (from v3)
LETTERS = "ABCDEFGHIJ"


def extract_letter(text, n_options=10):
    """Strict: the response starts with a valid option letter (e.g. 'B', 'B.', '(B)')."""
    valid = LETTERS[: int(n_options) if pd.notna(n_options) else 10]
    t = str(text).strip()
    m = re.match(r"^\(?([A-J])(?:[\).:]|\s|$)", t)
    if m and m.group(1) in valid and not re.match(r"^I\s+[a-z]", t):   # "I think..." is not option I
        return m.group(1)
    m = re.search(r"answer(?: is)?\s*[:\-]?\s*\(?([A-J])\b", t, re.IGNORECASE)
    return m.group(1).upper() if m and m.group(1).upper() in valid else None


def extract_letter_reasoned(text, n_options=10):
    """For answers with reasoning: \\boxed{X}, else the last 'answer is X', else None."""
    valid = LETTERS[: int(n_options) if pd.notna(n_options) else 10]
    t = str(text)
    for pat in (r"\\boxed\{\s*\(?([A-J])\)?\s*\}", r"answer(?: is)?\s*[:\-]?\s*\(?([A-J])\b"):
        m = re.findall(pat, t)
        if m and m[-1] in valid:
            return m[-1]
    return None


def extract_letter_legacy(text):
    """The grader used for the paper's results (first standalone capital letter)."""
    m = re.search(r"\b([A-J])\b", str(text).strip().upper())
    return m.group(1) if m else None


def clean_code(text):
    """Remove only surrounding Markdown code fences (same as the paper)."""
    text = str(text).strip()
    if text.startswith("```python"):
        text = text[len("```python"):].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text


# ---------------------------------------------------------------- benchmark
bench = pd.read_csv(BENCHMARK, dtype={"prompt_id": str})
need = {"prompt_id", "task", "prompt", "reference", "split", "experiment", "models"}
assert need <= set(bench.columns), f"{BENCHMARK} is missing {need - set(bench.columns)}; run build_benchmark_v4.py"
assert bench["prompt_id"].is_unique
if SMOKE_TEST:
    bench = bench.groupby("experiment", sort=False).head(2).reset_index(drop=True)
    log(f"SMOKE TEST: {len(bench)} prompts")
BENCH_HASH = hashlib.md5("|".join(sorted(bench["prompt_id"] + "::" + bench["experiment"])).encode()).hexdigest()[:12]
bench["model_list"] = bench["models"].str.split(";")
unknown = set(m for ms in bench["model_list"] for m in ms) - set(MODELS)
assert not unknown, f"unknown models in benchmark: {unknown}"
log(f"Benchmark {BENCHMARK}: {len(bench)} prompts, hash {BENCH_HASH}")
log(bench.groupby(["experiment", "models"]).size().rename("prompts").to_string())


SKIPPED = set()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- stage: infer
def stage_infer():
    ckpt = os.path.join(CKPT_DIR, "infer_checkpoint.csv")
    rows, done = [], set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt, dtype={"prompt_id": str})
        if len(prev) and (prev["benchmark_hash"] == BENCH_HASH).all():
            # Drop runs made with an outdated token limit (e.g. math_1.5B QA cut off at 4 tokens).
            stale = prev.apply(lambda r: int(r["token_limit"]) != token_limit(r["model_name"], r["task"]), axis=1)
            if stale.any():
                log(f"Re-running {int(stale.sum())} checkpointed runs whose token limit changed: "
                    f"{prev.loc[stale].groupby(['model_name', 'task']).size().to_dict()}")
                prev = prev.loc[~stale]
            rows = prev.to_dict("records")
            done = set(zip(prev["model_name"], prev["prompt_id"]))
            log(f"Resuming inference: {len(done)} runs already in {ckpt}")
    for name in MODELS:
        items = bench[bench["model_list"].map(lambda ms: name in ms)][["prompt_id", "task", "prompt"]].to_dict("records")
        if not items:
            continue
        order = random.Random(f"{SEED}-v4-{name}").sample(items, len(items))   # randomized order
        todo = [q for q in order if (name, q["prompt_id"]) not in done]
        if not todo:
            log(f"{name}: all {len(items)} runs already done")
            continue
        try:
            model, tok = load_model(name)
        except Exception as e:                      # e.g. wrong model id or unsupported format
            log(f"SKIPPED {name}: could not load {MODELS[name]}: {type(e).__name__}: {e}")
            SKIPPED.add(name)
            free_gpu()
            continue
        idle = measure_idle_power(10)
        log(f"Idle power ({name}): {idle} W | {len(todo)} prompts to run")
        run_order = {q["prompt_id"]: k for k, q in enumerate(order, 1)}
        start = len(rows)
        run_prompts(model, tok, name, todo, rows, ckpt, done, {"idle_power_w": idle})
        for r in rows[start:]:
            r["run_order"] = run_order[r["prompt_id"]]
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        del model, tok            # release this model before the next one is loaded
        free_gpu()
    df = pd.DataFrame(rows)
    for _, r in bench.iterrows():
        got = set(df.loc[df["prompt_id"] == r["prompt_id"], "model_name"]) if len(df) else set()
        want = set(r["model_list"]) - SKIPPED
        assert got >= want, f"{r['prompt_id']}: ran {got}, expected {want}"
    if SKIPPED:
        log(f"WARNING: models skipped because they failed to load: {sorted(SKIPPED)}")
    assert not df.duplicated(["model_name", "prompt_id"]).any()
    df.to_csv(out("infer_results_v4.csv"), index=False)
    log(f"Saved {len(df)} runs to {out('infer_results_v4.csv')}")


def run_prompts(model, tok, name, items, rows, ckpt=None, done=None, extra=None):
    done = done if done is not None else set()
    todo = [q for q in items if (name, q["prompt_id"]) not in done]
    prefix = MODEL_PROMPT_PREFIX.get(name, "")
    for k, q in enumerate(todo, 1):
        lim = token_limit(name, q["task"])
        m = measure_query(model, tok, prefix + q["prompt"], lim)
        rows.append({"model_name": name, "prompt_id": q["prompt_id"], "task": q["task"],
                     "token_limit": lim, "benchmark_hash": BENCH_HASH,
                     "prompt_prefix": prefix,
                     **(extra or {}), **m})
        if k % CHECKPOINT_EVERY == 0 or k == len(todo):
            if ckpt:
                pd.DataFrame(rows).to_csv(ckpt, index=False)
            log(f"{name} | {k}/{len(todo)} | {q['task']} | {q['prompt_id']} | in={m['input_tokens']} "
                f"out={m['generated_tokens']} | {m['latency_s']:.3f}s | "
                f"{m['gpu_energy_avgp_latency_j'] or float('nan'):.1f} J | {m['answer'][:60]!r}")
    return rows


# ---------------------------------------------------------------- stage: grade
def extract_number(text):
    """Final numeric answer: \\boxed{...}, then 'final answer is ...', then the last number."""
    t = str(text).replace(",", "")
    for pat in (r"\\boxed\{([^{}]*)\}", r"final answer is[:\s]*\$?\s*(-?[\d.]+)", r"answer[:\s]*\$?\s*(-?[\d.]+)"):
        m = re.findall(pat, t, re.IGNORECASE)
        if m:
            nums = re.findall(r"-?\d+(?:\.\d+)?", m[-1])
            if nums:
                return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    return nums[-1] if nums else None


def same_number(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


_EVALPLUS = {}


def grade(df):
    from rouge_score import rouge_scorer  # noqa: F401  (kept for parity with v3 environment check)
    cols = ["prompt_id", "reference", "category", "split", "experiment"] + \
           [c for c in ("n_options", "original_reference") if c in bench]
    g = df.merge(bench[cols], on="prompt_id", how="left", validate="many_to_one")
    assert g["reference"].notna().all()
    g["quality_score"], g["prediction"] = np.nan, None

    qa = g["task"].isin(["factual_qa", "factual_qa_shuffled"])
    nopt = g["n_options"] if "n_options" in g else pd.Series(10, index=g.index)
    g.loc[qa, "prediction"] = [extract_letter_reasoned(a, n) if int(lim) > 4 else extract_letter(a, n)
                               for a, n, lim in zip(g.loc[qa, "answer"], nopt[qa], g.loc[qa, "token_limit"])]
    g.loc[qa, "quality_score"] = (g.loc[qa, "prediction"].astype(str) == g.loc[qa, "reference"].astype(str)).astype(float)
    g.loc[qa, "format_ok"] = g.loc[qa, "prediction"].notna()

    sm = g["task"] == "summarization"
    if sm.any():
        from rouge_score import rouge_scorer as _rs
        scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
        clean = g.loc[sm, "answer"].astype(str).str.replace(r"</?[A-Za-z_]+>", " ", regex=True).str.strip()
        g.loc[sm, "quality_score"] = [scorer.score(str(r), a)["rougeL"].fmeasure for r, a in zip(g.loc[sm, "reference"], clean)]
        g.loc[sm, "rougeL_raw"] = [scorer.score(str(r), str(a))["rougeL"].fmeasure for r, a in zip(g.loc[sm, "reference"], g.loc[sm, "answer"])]
        g.loc[sm, "format_ok"] = ~g.loc[sm, "answer"].astype(str).str.contains(r"</?[A-Za-z_]+>", regex=True)

    mt = g["task"] == "math"
    g.loc[mt, "prediction"] = g.loc[mt, "answer"].map(extract_number)
    g.loc[mt, "quality_score"] = [float(same_number(p, r)) for p, r in zip(g.loc[mt, "prediction"], g.loc[mt, "reference"])]
    g.loc[mt, "format_ok"] = g.loc[mt, "answer"].astype(str).str.contains(r"final answer is|\\boxed", case=False, regex=True)

    cd = g["task"] == "coding"
    if cd.any():
        from evalplus.data import get_mbpp_plus
        from evalplus.eval import PASS
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        from evalplus.evaluate import check_correctness, get_groundtruth
        ids = sorted(bench.loc[bench["task"] == "coding", "prompt_id"])
        key = hashlib.md5("|".join(ids).encode()).hexdigest()[:10]
        if key not in _EVALPLUS:
            allp = get_mbpp_plus()
            probs = {t: allp[t] for t in ids}
            _EVALPLUS[key] = (probs, get_groundtruth(probs, f"mbpp_plus_v3_{key}", MBPP_OUTPUT_NOT_NONE_TASKS))
        probs, expected = _EVALPLUS[key]
        for idx in g.index[cd]:
            tid = g.at[idx, "prompt_id"]
            try:
                res = check_correctness(dataset="mbpp", completion_id=0, problem=probs[tid],
                                        solution=clean_code(g.at[idx, "answer"]),
                                        expected_output=expected[tid], base_only=False, fast_check=False)
                ok = res["base"][0] == PASS and res["plus"][0] == PASS
                g.at[idx, "prediction"], g.at[idx, "quality_score"] = ("pass" if ok else "fail"), float(ok)
            except Exception as e:
                g.at[idx, "prediction"], g.at[idx, "quality_score"] = "evaluation_error", 0.0
                log(f"Evaluation error {g.at[idx, 'model_name']} {tid}: {e}")
    return g


def stage_grade():
    df = pd.read_csv(out("infer_results_v4.csv"), dtype={"prompt_id": str})
    assert (df["benchmark_hash"] == BENCH_HASH).all(), "infer_results_v4.csv is from a different benchmark"
    g = grade(df)
    g.to_csv(out("infer_results_v4_graded.csv"), index=False)
    summ = g.groupby(["experiment", "model_name", "task"]).agg(
        n=("prompt_id", "count"), quality=("quality_score", "mean"), format_ok=("format_ok", "mean"),
        energy_avgp_j=("gpu_energy_avgp_latency_j", "mean"), energy_counter_j=("gpu_energy_counter_j", "mean"),
        latency_s=("latency_s", "mean"), input_tokens=("input_tokens", "mean"),
        generated_tokens=("generated_tokens", "mean"), hit_limit_rate=("hit_token_limit", "mean"),
        idle_power_w=("idle_power_w", "first")).reset_index()
    summ.to_csv(out("summary_v4.csv"), index=False)
    log("\n" + summ.to_string(index=False))
    shuf = g[g["task"] == "factual_qa_shuffled"]
    if len(shuf):
        top = shuf.groupby("model_name")["prediction"].agg(lambda s: s.value_counts(normalize=True).head(2).round(3).to_dict())
        log("Most common answer letters with shuffled options:\n" + top.to_string())


# ---------------------------------------------------------------- main
STAGE_FUNCS = {"infer": stage_infer, "grade": stage_grade}
for st in STAGES:
    log(f"\n######## STAGE: {st} ########")
    t = time.time()
    STAGE_FUNCS[st]()
    log(f"######## {st} done in {(time.time() - t) / 60:.1f} min ########")
log("\nAll requested stages finished.")
