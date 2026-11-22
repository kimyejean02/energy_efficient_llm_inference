# v3 experiment on UTCS Condor: larger benchmark + real router

## What this run adds

| # | Addition | Where |
|---|---|---|
| 2 | **Larger benchmark:** 1,000 MMLU-Pro, 200 XSum, all 378 MBPP+. Same prompt templates and sampling as before, so the original 300 prompts are included unchanged (`in_pilot = True`). | `build_benchmark.py` |
| 1 | **Real router:** every prompt is in a `dev` or `test` half. Routing policies are fitted on dev and scored on test, including a **cascade** (cheap model first, escalate to the 3B when its confidence is low). The fitted cascade is then **actually run live** on the test half. | `llm_energy_gpu.py` (stages `router`, `live`) |
| 4 | **Measurement:** input tokens, idle power (for idle-subtracted energy), GPU temperature, hardware energy counter when the GPU has one, randomized prompt order, resumable checkpoints. | `llm_energy_gpu.py` |

Generation settings (system prompt, token limits, greedy decoding, FP16) are unchanged from the paper.
One difference: prompts are no longer truncated at 1,024 tokens (longer ones are only flagged), and
the multiple-choice grader no longer reads "I think..." as option I. The paper's grader is also
reported as `quality_legacy` so you can compare.

## Files

| File | Purpose |
|---|---|
| `build_benchmark.py` | Run once on darmok to create `benchmark_prompts_v3.csv` (needs internet). |
| `benchmark_prompts.csv` | Your original 300 prompts; the builder checks they're all included unchanged. |
| `llm_energy_gpu.py` | The experiment (stages: infer, repeats, grade, router, live). |
| `llm_energy_gpu.sh` | Job wrapper: environment, GPU info, packages results into a tarball even if the job crashes. |
| `llm_energy_gpu_smoke.sub` | Short test of every stage (~10 min). |
| `llm_energy_gpu.sub` | Full run (~2–2.5 hours on a GTX 1080 Ti). |

## Steps

### 1. Copy files to darmok (from your Mac)
```bash
scp condor_v3.zip ishaj@darmok.cs.utexas.edu:~/
```
On darmok:
```bash
mkdir -p ~/llm_energy_v3 && cd ~/llm_energy_v3
unzip ~/condor_v3.zip
```

### 2. Install packages and build the benchmark (on darmok)
```bash
source /lusr/opt/miniconda/bin/activate pytorch-cuda
python -m pip install --user datasets scipy
export HF_HOME=/scratch/cluster/$USER/huggingface
python build_benchmark.py
```
It should end with `OK: all 300 original prompts are included unchanged.` and create
`benchmark_prompts_v3.csv` (1,578 prompts). If the XSum download fails with an error about
dataset scripts, run `python -m pip install --user "datasets<4"` and try again.

### 3. Check your scratch path and email
- `llm_energy_gpu.sh`: `SCRATCH=` line (same as before).
- Both `.sub` files: `notify_user` and `+Group`.

### 4. Smoke test, then the full run
```bash
condor_submit llm_energy_gpu_smoke.sub
tail -f llm_energy_v3_smoke.out
```
Check that it ends with `All requested stages finished.`, the `GPU check` lines say `OK`, and the
live stage prints `Live answers identical to matrix answers: 1.000`. Then:
```bash
condor_submit llm_energy_gpu.sub
tail -f llm_energy_v3.out
```

### 5. Get the results
```bash
tar -xzf results_full.tar.gz    # creates results/
```
A copy is also saved to `$SCRATCH/llm_energy_v3/runs/<timestamp>_full/`.

## If the job is cut off or too long

Checkpoints are kept in `$SCRATCH/llm_energy_v3/ckpt_full`. Resubmitting the same job resumes
inference where it stopped. To split the run into two shorter jobs, add one of these lines to
`llm_energy_gpu.sub` and submit twice:
```
environment = "STAGES=infer,repeats"      # first job (~1.5 h)
environment = "STAGES=grade,router,live"  # second job (~40 min); needs the first job's results/
```
For the second job, extract the first job's tarball in the same folder and add `results` to
`transfer_input_files`.

## Output files (in `results/`)

| File | Contents |
|---|---|
| `infer_results_graded.csv` | Every model x prompt run: energy (3 estimators), latency, input/output tokens, confidence, quality, split |
| `summary_by_model_task.csv` | Quality (with 95% CI), energy, latency, tokens, idle power by model and task |
| `mmlu_energy_*.csv` | Repeated MMLU energy on the original 100 questions, plus block-level energy |
| `router_policies.json` | Routing policies fitted on the dev split |
| `router_eval.csv` | Every policy on dev and test: quality, energy, escalation rate, % saving vs always-3B |
| `router_significance.csv` | 95% CI of the energy saving and paired quality tests vs always-3B (test split) |
| `router_sweep_test.csv` | Trade-off curve: quality and energy as the escalation threshold varies |
| `live_routed_summary.csv` | The cascade router actually run on the test split: measured energy, quality, saving |
| `live_routed_runs.csv`, `live_routed_blocks.csv` | Per-request and per-block detail for the live run |
