# v4 additions: more specialized models

This run adds **only new measurements**. They merge with your v3 results (same prompt ids), and the
combined router analysis is done afterwards on the merged data.

| Experiment | Model(s) | Prompts | Question it answers |
|---|---|---|---|
| `coder3b_coding` | Qwen2.5-Coder-3B-Instruct | 378 MBPP+ | 2x2 size x specialization: does specialization still help at 3B? |
| `gsm8k` | Qwen2.5-1.5B, Qwen2.5-3B, **Qwen2.5-Math-1.5B-Instruct** | 100 GSM8K | Second specialization pair: math |
| `mmlu_stem_math` | Qwen2.5-Math-1.5B-Instruct | ~400 MMLU-Pro (math, physics, chemistry, engineering) | Can the math specialist help on STEM questions (and be routed to)? |
| `summarizer_xsum` | agentlans/Qwen2.5-1.5B-Instruct-Summarizer | 200 XSum | Does a community summarization fine-tune beat the general model? |
| `mmlu_shuffled` | Qwen2.5-1.5B, 3B, Coder-1.5B | 1,000 MMLU-Pro, options shuffled | Is the coder's "J" habit about knowledge or answer position? |

Same settings as v3 (system prompt, greedy decoding, FP16, token limits). GSM8K allows up to 512
output tokens because the models work through the problem step by step. Summaries are scored with
ROUGE-L after removing wrapper tags such as `<text>...</text>` (the raw score is kept as `rougeL_raw`).
If a model fails to load (e.g. the community summarizer), the job skips it and finishes the rest.

**Estimated time on the GTX 1080 Ti: about 2 hours** (GSM8K is the slowest part, ~70 min).

## Steps (on darmok, in your v3 folder)

### 1. Copy and unzip (from your Mac)
```bash
cd ~/Downloads
scp condor_v4.zip ishaj@darmok.cs.utexas.edu:~/
```
On darmok:
```bash
cd ~/llm_energy_v3
unzip -o ~/condor_v4.zip
```
(It must be the v3 folder: the builder reads `benchmark_prompts_v3.csv` from there.)

### 2. Build the v4 benchmark
```bash
source /lusr/opt/miniconda/bin/activate pytorch-cuda
export HF_HOME=/scratch/cluster/ishaj/huggingface
python build_benchmark_v4.py
```
It prints a table of experiments (5 rows) and creates `benchmark_v4.csv`.

### 3. Download and check all models
```bash
python check_models.py
```
- It should end with `All models downloaded.`
- Read the **summarizer's model card** it prints. If it says to use a special prompt
  (e.g. starting with "Summarize:"), tell Claude before submitting.
- If it says "adapter only", run `python -m pip install --user peft`.
- If the summarizer fails to download, the model name may be wrong; you can still run
  everything else (the job will skip it).

### 4. Smoke test, then the full run
```bash
condor_submit llm_energy_v4_smoke.sub
tail -f llm_energy_v4_smoke.out          # Ctrl+C stops watching, not the job
```
It should end with `All requested stages finished.` and print a summary table. Then:
```bash
condor_submit llm_energy_v4.sub
```

### 5. Get the results (on your Mac)
The job saves `results_v4_full.tar.gz` and a backup in
`/scratch/cluster/ishaj/llm_energy_v4/runs/<timestamp>_full/`. On your Mac:
```bash
scp ishaj@darmok.cs.utexas.edu:/scratch/cluster/ishaj/llm_energy_v4/runs/<timestamp>_full/results_v4_full.tar.gz ~/Downloads/
```
(Find `<timestamp>` with `ls /scratch/cluster/ishaj/llm_energy_v4/runs/` on darmok.)
