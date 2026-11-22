# Energy-Efficient LLM Inference via Adaptive Model Selection — project files

Everything used to produce the results in `5_paper/Energy_Efficient_LLM_Inference_Paper_v3`.

## Folders

| Folder | What it is |
|---|---|
| `1_condor_v3_main_run/` | Main experiment on the UTCS Condor cluster (GTX 1080 Ti): 1.5B, 3B and Coder-1.5B models on 1,578 prompts (MMLU-Pro, XSum, MBPP+). Stages: inference, 5x repeated QA energy, grading, router fitting, live router run. |
| `2_condor_v4_additions/` | Second run: Coder-3B, Math-1.5B, Summarizer-1.5B, GSM8K (100 problems), and shuffled-option MMLU-Pro. |
| `3_results/v3/` | Output of run 1 (`infer_results_graded.csv` is the main per-run file). |
| `3_results/v4/` | Output of run 2 (`infer_results_v4_graded.csv`), plus the re-fitted router (`router4_eval.csv`, `router4_policies.json`). |
| `4_analysis/` | Scripts that compute the paper's numbers and figures from the result CSVs. |
| `5_paper/` | Paper (.docx and .pdf), the build script, the paper text (`content_v4.py`), the IEEE template (`tpl/`), and the figures. |

## How to reproduce on Condor (darmok)

Environment: `source /lusr/opt/miniconda/bin/activate pytorch-cuda` and
`export HF_HOME=/scratch/cluster/<user>/huggingface`. Needs `rouge-score` and `evalplus`.

1. **Run 1:** in `1_condor_v3_main_run/`, run `python build_benchmark.py` (creates
   `benchmark_prompts_v3.csv`), then `condor_submit llm_energy_gpu.sub`. See its README.
2. **Run 2:** copy `2_condor_v4_additions/` into the same folder (it reads
   `benchmark_prompts_v3.csv`), run `python build_benchmark_v4.py` (expected hash
   `f942bcd5be49`), `python check_models.py`, then `condor_submit llm_energy_v4.sub`.
   A crashed job resumes from its checkpoint in scratch when resubmitted.

Before submitting, set `+Group` in the `.sub` files to your account type (`GRAD` or `UNDER`)
and `notify_user` to your email.

## How to reproduce the analysis and paper

The analysis scripts were run in our workspace and use absolute paths
(`/home/claude/v3res/results/...`, `/home/claude/v4res/results_v4/...`). Point those paths
at `3_results/v3/` and `3_results/v4/` before running.

- `an4.py`: shared loading/helper code for the v4 comparisons (paired McNemar test, pivots).
- `router4.py`: re-fits the routing policies with all six models, with and without GSM8K.
- `figs_v3.py`: Figs. 1, 2 and 4 of the paper (quality vs energy, prompt length, router sweep).
- `figs_v4.py`: Fig. 3 (specialists vs general models).
- Paper: in `5_paper/`, run `python build_paper_v4.py` (needs `lxml`; figures are read from
  `analysis/`, so copy `figures/*.png` into an `analysis/` folder next to the script first).

## Notes

- Benchmark prompt CSVs are not included; the build scripts regenerate them from
  Hugging Face datasets with a fixed seed (42).
- Energy numbers use `gpu_energy_avgp_latency_j` (mean power × latency), GPU board only.
- Run 2 may have used a different physical GPU than run 1; the same questions measured
  3–6% higher in energy there.
- The collaborator's independent pilot run (used for the replication check) is not included here.
