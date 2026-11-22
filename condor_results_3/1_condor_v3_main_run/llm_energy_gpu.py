"""Energy-efficient LLM inference via adaptive model selection: v3 experiment.

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
BENCHMARK = os.environ.get("BENCHMARK", "benchmark_prompts_v3.csv")
OUT_DIR = os.environ.get("OUT_DIR", "results")
CKPT_DIR = os.environ.get("CKPT_DIR", OUT_DIR)
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
STAGES = [s.strip() for s in os.environ.get("STAGES", "infer,repeats,grade,router,live").split(",") if s.strip()]
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

MODELS = {
    "small_1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "large_3B": "Qwen/Qwen2.5-3B-Instruct",
    "coder_1.5B": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
}
LARGE = "large_3B"
TASKS = ["factual_qa", "summarization", "coding"]
TASK_MAX_TOKENS = {"factual_qa": 4, "summarization": 128, "coding": 256}   # same as the paper
SYSTEM_PROMPT = "Answer the user's question clearly and concisely."       # same as the paper
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

GPU_STAGES = {"infer", "repeats", "live"}
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
log(f"Hardware energy counter available: {ENERGY_COUNTER_AVAILABLE} (Volta and newer GPUs only)")


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


def unload(model, tok):
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


# ---------------------------------------------------------------- benchmark
bench = pd.read_csv(BENCHMARK)
need = {"prompt_id", "task", "prompt", "reference", "split"}
assert need <= set(bench.columns), f"{BENCHMARK} is missing {need - set(bench.columns)}; run build_benchmark.py"
bench["prompt_id"] = bench["prompt_id"].astype(str)
if "category" not in bench:
    bench["category"] = bench["task"]
if "in_pilot" not in bench:
    bench["in_pilot"] = True
if SMOKE_TEST:
    bench = bench.groupby(["task", "split"], sort=False).head(2).reset_index(drop=True)
    log(f"SMOKE TEST: {len(bench)} prompts")
BENCH_HASH = hashlib.md5("|".join(sorted(bench["prompt_id"] + "::" + bench["task"])).encode()).hexdigest()[:12]
log(f"Benchmark {BENCHMARK}: {len(bench)} prompts, hash {BENCH_HASH}")
log(bench.groupby(["task", "split"]).size().unstack().to_string())


def run_prompts(model, tok, name, items, rows, ckpt=None, done=None, extra=None):
    """Run items (list of dicts with prompt_id/task/prompt) in the given order."""
    done = done if done is not None else set()
    todo = [q for q in items if (name, q["prompt_id"]) not in done]
    for k, q in enumerate(todo, 1):
        m = measure_query(model, tok, q["prompt"], TASK_MAX_TOKENS[q["task"]])
        rows.append({"model_name": name, "prompt_id": q["prompt_id"], "task": q["task"],
                     "token_limit": TASK_MAX_TOKENS[q["task"]], "benchmark_hash": BENCH_HASH,
                     **(extra or {}), **m})
        if k % CHECKPOINT_EVERY == 0 or k == len(todo):
            if ckpt:
                pd.DataFrame(rows).to_csv(ckpt, index=False)
            log(f"{name} | {k}/{len(todo)} | {q['task']} | {q['prompt_id']} | in={m['input_tokens']} "
                f"out={m['generated_tokens']} | {m['latency_s']:.3f}s | "
                f"{m['gpu_energy_avgp_latency_j'] or float('nan'):.1f} J | {m['answer'][:60]!r}")
    return rows


# ---------------------------------------------------------------- stage: infer
def stage_infer():
    ckpt = os.path.join(CKPT_DIR, "infer_checkpoint.csv")
    rows, done = [], set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt, dtype={"prompt_id": str})
        if len(prev) and (prev["benchmark_hash"] == BENCH_HASH).all():
            rows = prev.to_dict("records")
            done = set(zip(prev["model_name"], prev["prompt_id"]))
            log(f"Resuming inference: {len(done)} runs already in {ckpt}")
    items = bench[["prompt_id", "task", "prompt"]].to_dict("records")
    for name in MODELS:
        order = random.Random(f"{SEED}-{name}").sample(items, len(items))   # randomized task order
        if all((name, q["prompt_id"]) in done for q in order):
            continue
        model, tok = load_model(name)
        idle = measure_idle_power(10)
        log(f"Idle power ({name}): {idle} W")
        run_order = {q["prompt_id"]: k for k, q in enumerate(order, 1)}
        start = len(rows)
        run_prompts(model, tok, name, order, rows, ckpt, done, {"idle_power_w": idle})
        for r in rows[start:]:
            r["run_order"] = run_order[r["prompt_id"]]
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        unload(model, tok)
    df = pd.DataFrame(rows)
    for name in MODELS:
        got = df.loc[df["model_name"] == name, "prompt_id"]
        assert got.is_unique and set(got) == set(bench["prompt_id"]), f"{name}: incomplete inference"
    df.to_csv(out("infer_results.csv"), index=False)
    log(f"Saved {len(df)} runs to {out('infer_results.csv')}")


# ---------------------------------------------------------------- stage: repeats
def stage_repeats():
    pool = bench[(bench["task"] == "factual_qa") & bench["in_pilot"].astype(bool)]
    if pool.empty:
        pool = bench[bench["task"] == "factual_qa"]
    items = pool[["prompt_id", "task", "prompt"]].to_dict("records")
    rows, blocks = [], []
    for name in MODELS:
        model, tok = load_model(name)
        idle = measure_idle_power(10)
        logger = PowerLogger().start()
        for q in items:
            for rep in range(1, MMLU_REPEATS + 1):
                m = measure_query(model, tok, q["prompt"], TASK_MAX_TOKENS["factual_qa"])
                rows.append({"model_name": name, "prompt_id": q["prompt_id"], "repeat": rep,
                             "idle_power_w": idle, **m})
        b = logger.stop()
        n = len(items) * MMLU_REPEATS
        blocks.append({"model_name": name, "queries": n, "idle_power_w": idle, **b,
                       "block_energy_per_query_j": b["block_energy_j"] / n if b["block_energy_j"] else None,
                       "block_counter_per_query_j": b["block_energy_counter_j"] / n if b["block_energy_counter_j"] else None})
        log(f"{name}: block-level MMLU energy per query {blocks[-1]['block_energy_per_query_j']} J")
        pd.DataFrame(rows).to_csv(os.path.join(CKPT_DIR, "repeats_checkpoint.csv"), index=False)
        unload(model, tok)
    rep = pd.DataFrame(rows)
    rep.to_csv(out("mmlu_energy_repeats.csv"), index=False)
    pd.DataFrame(blocks).to_csv(out("mmlu_energy_block.csv"), index=False)
    per = rep.groupby(["model_name", "prompt_id"], as_index=False).agg(
        trap_j=("gpu_energy_j", "mean"), avgp_j=("gpu_energy_avgp_latency_j", "mean"),
        sd_trap_j=("gpu_energy_j", "std"), latency_s=("latency_s", "mean"), input_tokens=("input_tokens", "first"))
    per.to_csv(out("mmlu_energy_per_prompt.csv"), index=False)
    log(per.groupby("model_name")[["trap_j", "avgp_j", "latency_s"]].mean().to_string())


# ---------------------------------------------------------------- stage: grade
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


_EVALPLUS = {}


def grade(df):
    """Add quality_score (and helpers) to a dataframe of runs. Executes generated code."""
    from rouge_score import rouge_scorer
    cols = ["prompt_id", "reference", "category", "split"] + (["n_options"] if "n_options" in bench else [])
    g = df.merge(bench[cols], on="prompt_id", how="left", validate="many_to_one")
    assert g["reference"].notna().all()
    g["quality_score"] = np.nan
    g["prediction"] = None

    qa = g["task"] == "factual_qa"
    nopt = g["n_options"] if "n_options" in g else pd.Series(10, index=g.index)
    g.loc[qa, "prediction"] = [extract_letter(a, n) for a, n in zip(g.loc[qa, "answer"], nopt[qa])]
    g.loc[qa, "quality_score"] = (g.loc[qa, "prediction"].astype(str) == g.loc[qa, "reference"].astype(str)).astype(float)
    g.loc[qa, "quality_legacy"] = (g.loc[qa, "answer"].map(extract_letter_legacy).astype(str)
                                   == g.loc[qa, "reference"].astype(str)).astype(float)

    sm = g["task"] == "summarization"
    if sm.any():
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        g.loc[sm, "quality_score"] = [scorer.score(str(r), str(a))["rougeL"].fmeasure
                                      for r, a in zip(g.loc[sm, "reference"], g.loc[sm, "answer"])]

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
    df = pd.read_csv(out("infer_results.csv"), dtype={"prompt_id": str})
    assert (df["benchmark_hash"] == BENCH_HASH).all(), "infer_results.csv is from a different benchmark"
    g = grade(df)
    g.to_csv(out("infer_results_graded.csv"), index=False)
    rng = np.random.default_rng(SEED)

    def ci(x):
        x = np.asarray(x, float)
        m = rng.choice(x, (5000, len(x))).mean(1)
        return f"[{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"

    summ = g.groupby(["model_name", "task"]).agg(
        n=("prompt_id", "count"), quality=("quality_score", "mean"),
        quality_ci=("quality_score", ci), quality_legacy=("quality_legacy", "mean"),
        energy_avgp_j=("gpu_energy_avgp_latency_j", "mean"), energy_trap_j=("gpu_energy_j", "mean"),
        energy_counter_j=("gpu_energy_counter_j", "mean"), latency_s=("latency_s", "mean"),
        input_tokens=("input_tokens", "mean"), generated_tokens=("generated_tokens", "mean"),
        hit_limit_rate=("hit_token_limit", "mean"), avg_power_w=("avg_gpu_power_w", "mean"),
        idle_power_w=("idle_power_w", "first")).reset_index()
    summ.to_csv(out("summary_by_model_task.csv"), index=False)
    log("\n" + summ.to_string(index=False))


# ---------------------------------------------------------------- stage: router (CPU)
def detect_task(prompt):
    """Rule-based task detector used by the router (runs on CPU before any model).

    It reads the instruction at the start of the prompt, where users usually state the task.
    """
    head = prompt[:300].lower()
    if "summarize" in head or "summary" in head:
        return "summarization"
    if "python" in head or "write a function" in head:
        return "coding"
    if "multiple-choice" in head or re.search(r"\n[a-j]\. ", prompt.lower()):
        return "factual_qa"
    return "unknown"


def add_router_columns(g):
    g = g.copy()
    g["energy"] = g["gpu_energy_avgp_latency_j"]
    g["energy_dynamic"] = g["energy"] - g["idle_power_w"].fillna(0) * g["latency_s"]
    # Confidence: probability of the answer letter for QA; mean token log-prob otherwise.
    g["confidence"] = np.where(g["task"] == "factual_qa", g["first_logprob"], g["mean_logprob"])
    return g


def tables(g, task, split):
    d = g[(g["task"] == task) & (g["split"] == split)]
    piv = lambda c: d.pivot(index="prompt_id", columns="model_name", values=c)
    return piv("quality_score"), piv("energy"), piv("confidence"), \
        d.drop_duplicates("prompt_id").set_index("prompt_id")["category"]


def apply_policy(spec, Q, E, C):
    """Returns per-prompt (quality, energy, escalated) for one task's policy spec."""
    if spec["type"] == "single":
        m = spec["model"]
        return Q[m], E[m], pd.Series(False, index=Q.index)
    if spec["type"] == "cascade":
        f = spec["first"]
        esc = ~(C[f] >= spec["tau"])          # missing confidence -> escalate (same as live)
        return Q[f].where(~esc, Q[LARGE]), E[f] + esc * E[LARGE], esc
    if spec["type"] == "by_category":
        q, e = pd.Series(index=Q.index, dtype=float), pd.Series(index=Q.index, dtype=float)
        for pid in Q.index:
            m = spec["map"].get(spec["_cat"][pid], LARGE)
            q[pid], e[pid] = Q.at[pid, m], E.at[pid, m]
        return q, e, pd.Series(False, index=Q.index)
    raise ValueError(spec)


def fit_task(g, task, allow_cascade=True):
    """Cheapest policy whose DEV quality is within TOLERANCE of the 3B model's dev quality."""
    Q, E, C, _ = tables(g, task, "dev")
    target = Q[LARGE].mean() - TOLERANCE[task]
    cands = [{"type": "single", "model": m} for m in Q.columns]
    if allow_cascade:
        for f in Q.columns:
            if f == LARGE:
                continue
            taus = np.unique(np.quantile(C[f].dropna(), np.linspace(0, 1, 41)))
            cands += [{"type": "cascade", "first": f, "tau": float(t)} for t in taus]
    best = None
    for s in cands:
        q, e, _ = apply_policy(s, Q, E, C)
        if q.mean() >= target and (best is None or e.mean() < best[1] - 1e-9
                                   or (abs(e.mean() - best[1]) < 1e-9 and q.mean() > best[2])):
            best = (s, e.mean(), q.mean())
    return best[0] if best else {"type": "single", "model": LARGE}


def fit_by_category(g):
    Q, E, C, cat = tables(g, "factual_qa", "dev")
    mapping = {}
    for c in sorted(cat.unique()):
        ids = cat.index[cat == c]
        target = Q.loc[ids, LARGE].mean() - TOLERANCE["factual_qa"]
        ok = [m for m in Q.columns if Q.loc[ids, m].mean() >= target]
        mapping[c] = min(ok, key=lambda m: E.loc[ids, m].mean())
    return {"type": "by_category", "map": mapping}


def evaluate(g, policy, split):
    """policy: task -> spec. Returns per-task rows and per-prompt results."""
    rows, per = [], []
    for task in TASKS:
        Q, E, C, cat = tables(g, task, split)
        if Q.empty:
            continue
        spec = dict(policy[task])
        if spec["type"] == "by_category":
            spec["_cat"] = cat
        q, e, esc = apply_policy(spec, Q, E, C)
        rows.append({"task": task, "n": len(q), "quality": q.mean(), "energy_mean_j": e.mean(),
                     "energy_total_kj": e.sum() / 1000, "escalation_rate": esc.mean()})
        per.append(pd.DataFrame({"task": task, "quality": q, "energy": e, "escalated": esc}))
    return pd.DataFrame(rows), pd.concat(per)


def oracle(g, split):
    rows = []
    for task in TASKS:
        Q, E, _, _ = tables(g, task, split)
        if Q.empty:
            continue
        best_q = Q.max(axis=1)
        e = pd.Series([E.loc[p, [m for m in Q.columns if Q.at[p, m] >= best_q[p] - 1e-12]].min()
                       for p in Q.index], index=Q.index)
        rows.append({"task": task, "n": len(Q), "quality": best_q.mean(), "energy_mean_j": e.mean(),
                     "energy_total_kj": e.sum() / 1000, "escalation_rate": np.nan})
    return pd.DataFrame(rows)


def stage_router():
    from scipy import stats
    g = add_router_columns(pd.read_csv(out("infer_results_graded.csv"), dtype={"prompt_id": str}))

    # 1. Task detection accuracy and cost (CPU).
    t0 = time.perf_counter()
    det = bench["prompt"].map(detect_task)
    det_us = (time.perf_counter() - t0) / len(bench) * 1e6
    det_acc = float((det == bench["task"]).mean())
    log(f"Task detector accuracy {det_acc:.3f}, {det_us:.1f} microseconds per prompt (CPU)")

    # 2. Fit policies on dev.
    policies = {f"always {m}": {t: {"type": "single", "model": m} for t in TASKS} for m in MODELS}
    policies["task-aware (one model per task)"] = {t: fit_task(g, t, allow_cascade=False) for t in TASKS}
    policies["cascade (small first, escalate to 3B)"] = {t: fit_task(g, t, allow_cascade=True) for t in TASKS}
    pol_cat = dict(policies["task-aware (one model per task)"])
    pol_cat["factual_qa"] = fit_by_category(g)
    policies["task-aware + QA by subject"] = pol_cat
    with open(out("router_policies.json"), "w") as f:
        json.dump(policies, f, indent=2)

    # 3. Evaluate every policy on dev and test.
    allrows, per_test = [], {}
    for split in ("dev", "test"):
        base = None
        for name, pol in policies.items():
            r, per = evaluate(g, pol, split)
            if split == "test":
                per_test[name] = per
            r.insert(0, "policy", name)
            r.insert(1, "split", split)
            allrows.append(r)
        o = oracle(g, split)
        o.insert(0, "policy", "oracle (upper bound)")
        o.insert(1, "split", split)
        allrows.append(o)
    ev = pd.concat(allrows, ignore_index=True)
    tot = ev.groupby(["policy", "split"], sort=False)["energy_total_kj"].sum().rename("workload_energy_kj").reset_index()
    ref = tot[tot["policy"] == f"always {LARGE}"].set_index("split")["workload_energy_kj"]
    tot["saving_vs_always_3B_pct"] = 100 * (1 - tot["workload_energy_kj"] / tot["split"].map(ref))
    ev = ev.merge(tot, on=["policy", "split"])
    ev.to_csv(out("router_eval.csv"), index=False)
    log("\n" + ev[ev["split"] == "test"].to_string(index=False))

    # 4. Bootstrap CI of the test-split energy saving, and paired quality tests vs always-3B.
    rng = np.random.default_rng(SEED)
    base = per_test[f"always {LARGE}"]
    sig = []
    for name, per in per_test.items():
        if name == f"always {LARGE}":
            continue
        savings = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(np.where(per["task"].values == t)[0], (per["task"] == t).sum())
                                  for t in per["task"].unique()])
            savings.append(100 * (1 - per["energy"].values[idx].sum() / base["energy"].values[idx].sum()))
        row = {"policy": name, "saving_pct_ci95": f"[{np.percentile(savings, 2.5):.1f}, {np.percentile(savings, 97.5):.1f}]"}
        for t in per["task"].unique():
            a, b = per.loc[per["task"] == t, "quality"], base.loc[base["task"] == t, "quality"]
            if t == "summarization":
                p = stats.wilcoxon(a, b).pvalue if (a != b).any() else 1.0
            else:
                lo, hi = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
                p = stats.binomtest(lo, lo + hi, 0.5).pvalue if lo + hi else 1.0
            row[f"p_quality_vs_3B_{t}"] = p
        sig.append(row)
    pd.DataFrame(sig).to_csv(out("router_significance.csv"), index=False)
    log("\n" + pd.DataFrame(sig).to_string(index=False))

    # 5. Trade-off curves on test: sweep the escalation threshold.
    sweep = []
    for task in TASKS:
        Q, E, C, _ = tables(g, task, "test")
        if Q.empty:
            continue
        for f in [m for m in Q.columns if m != LARGE]:
            for qtl in np.linspace(0, 1, 41):
                tau = float(np.quantile(C[f].dropna(), qtl)) if qtl < 1 else np.inf
                q, e, esc = apply_policy({"type": "cascade", "first": f, "tau": tau}, Q, E, C)
                sweep.append({"task": task, "first_model": f, "threshold_quantile": qtl, "tau": tau,
                              "escalation_rate": esc.mean(), "quality": q.mean(), "energy_mean_j": e.mean(),
                              "large_quality": Q[LARGE].mean(), "large_energy_j": E[LARGE].mean()})
    pd.DataFrame(sweep).to_csv(out("router_sweep_test.csv"), index=False)
    with open(out("router_meta.json"), "w") as f:
        json.dump({"task_detector_accuracy": det_acc, "task_detector_us_per_prompt": det_us,
                   "tolerance": TOLERANCE, "energy_column": "gpu_energy_avgp_latency_j"}, f, indent=2)


# ---------------------------------------------------------------- stage: live routed run
def stage_live():
    """Run the dev-fitted cascade router on the test split for real, and measure it."""
    with open(out("router_policies.json")) as f:
        policy = json.load(f)["cascade (small first, escalate to 3B)"]
    test = bench[bench["split"] == "test"].copy()
    test["detected"] = test["prompt"].map(detect_task)
    test["route_task"] = np.where(test["detected"] == "unknown", test["task"], test["detected"])
    test["stage1_model"] = [policy[t]["model"] if policy[t]["type"] == "single" else policy[t]["first"]
                            for t in test["route_task"]]
    rows, blocks = [], []

    def run_block(name, items, stage):
        model, tok = load_model(name)
        idle = measure_idle_power(10)
        items = random.Random(f"{SEED}-live-{stage}-{name}").sample(items, len(items))
        logger = PowerLogger().start()
        start = len(rows)
        run_prompts(model, tok, name, items, rows, extra={"stage": stage, "idle_power_w": idle})
        b = logger.stop()
        blocks.append({"stage": stage, "model_name": name, "queries": len(items), **b})
        unload(model, tok)
        return rows[start:]

    # Stage 1: every test prompt goes to its task's first-stage model.
    stage1 = {}
    for name, grp in test.groupby("stage1_model"):
        for r in run_block(name, grp[["prompt_id", "task", "prompt"]].to_dict("records"), 1):
            stage1[r["prompt_id"]] = r
    # Escalate low-confidence answers to the 3B model.
    esc_ids = []
    for _, q in test.iterrows():
        spec = policy[q["route_task"]]
        r = stage1[q["prompt_id"]]
        conf = r["first_logprob"] if q["task"] == "factual_qa" else r["mean_logprob"]
        if spec["type"] == "cascade" and not (conf >= spec["tau"]):
            esc_ids.append(q["prompt_id"])
    log(f"Escalating {len(esc_ids)} of {len(test)} test prompts to {LARGE}")
    if esc_ids:
        run_block(LARGE, test[test["prompt_id"].isin(esc_ids)][["prompt_id", "task", "prompt"]].to_dict("records"), 2)

    live = grade(pd.DataFrame(rows))
    live.to_csv(out("live_routed_runs.csv"), index=False)
    pd.DataFrame(blocks).to_csv(out("live_routed_blocks.csv"), index=False)

    # Final answer per prompt = stage 2 if escalated, else stage 1. Energy = both stages.
    final = live.sort_values("stage").groupby("prompt_id").tail(1).set_index("prompt_id")
    energy = live.groupby("prompt_id")["gpu_energy_avgp_latency_j"].sum()
    # Compare with the matrix: same model + prompt should give the same answer (greedy decoding).
    mat = pd.read_csv(out("infer_results_graded.csv"), dtype={"prompt_id": str})
    m = live.merge(mat[["model_name", "prompt_id", "answer", "gpu_energy_avgp_latency_j"]],
                   on=["model_name", "prompt_id"], suffixes=("", "_matrix"))
    large_test = mat[(mat["model_name"] == LARGE) & mat["prompt_id"].isin(test["prompt_id"])].set_index("prompt_id")
    rows_s = []
    for task in TASKS:
        ids = test.loc[test["task"] == task, "prompt_id"]
        if ids.empty:
            continue
        rows_s.append({"task": task, "n": len(ids), "escalated": int(ids.isin(esc_ids).sum()),
                       "live_quality": final.loc[ids, "quality_score"].mean(),
                       "live_energy_total_kj": energy[ids].sum() / 1000,
                       "always_3B_quality": large_test.loc[ids, "quality_score"].mean(),
                       "always_3B_energy_total_kj": large_test.loc[ids, "gpu_energy_avgp_latency_j"].sum() / 1000})
    s = pd.DataFrame(rows_s)
    s.loc[len(s)] = {"task": "ALL", "n": s["n"].sum(), "escalated": s["escalated"].sum(),
                     "live_quality": np.nan, "live_energy_total_kj": s["live_energy_total_kj"].sum(),
                     "always_3B_quality": np.nan, "always_3B_energy_total_kj": s["always_3B_energy_total_kj"].sum()}
    s["saving_vs_always_3B_pct"] = 100 * (1 - s["live_energy_total_kj"] / s["always_3B_energy_total_kj"])
    s.to_csv(out("live_routed_summary.csv"), index=False)
    log("\n" + s.to_string(index=False))
    log(f"Live answers identical to matrix answers: {(m['answer'] == m['answer_matrix']).mean():.3f} "
        f"| live/matrix energy ratio: {m['gpu_energy_avgp_latency_j'].sum() / m['gpu_energy_avgp_latency_j_matrix'].sum():.3f}")
    log(f"Task detector routed {(test['detected'] == test['task']).mean():.3f} of test prompts to the right task")


# ---------------------------------------------------------------- main
STAGE_FUNCS = {"infer": stage_infer, "repeats": stage_repeats, "grade": stage_grade,
               "router": stage_router, "live": stage_live}
for st in STAGES:
    log(f"\n######## STAGE: {st} ########")
    t = time.time()
    STAGE_FUNCS[st]()
    log(f"######## {st} done in {(time.time() - t) / 60:.1f} min ########")
log("\nAll requested stages finished.")
