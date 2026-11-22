#!/bin/bash
# Usage (from the .sub files):
#   arguments = llm_energy_gpu.sh          -> full v3 run
#   arguments = llm_energy_gpu.sh smoke    -> short end-to-end test of every stage
# Optional: set STAGES in the .sub file's `environment` line to run only some stages.
set -euo pipefail

source /lusr/opt/miniconda/bin/activate pytorch-cuda

# ---- Change this to your own scratch directory ----
SCRATCH="/scratch/cluster/ishaj"
# ---------------------------------------------------

export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export BENCHMARK="benchmark_prompts_v3.csv"
export OUT_DIR="results"

MODE="full"
if [[ "${1:-}" == "smoke" ]]; then
    MODE="smoke"
    export SMOKE_TEST=1
fi
# Checkpoints live on scratch, so a resubmitted job resumes where the last one stopped.
export CKPT_DIR="$SCRATCH/llm_energy_v3/ckpt_$MODE"
mkdir -p "$CKPT_DIR" "$OUT_DIR"
echo "=== MODE: $MODE | STAGES: ${STAGES:-all} | checkpoints: $CKPT_DIR ==="

# Package results on exit (even after a crash) and keep a copy on scratch.
TARBALL="results_${MODE}.tar.gz"
BACKUP="$SCRATCH/llm_energy_v3/runs/$(date +%Y%m%d_%H%M%S)_$MODE"
package_results() {
    tar -czf "$TARBALL" "$OUT_DIR" 2>/dev/null || tar -czf "$TARBALL" --files-from /dev/null
    mkdir -p "$BACKUP" && cp -rf "$OUT_DIR" "$TARBALL" "$BACKUP"/ 2>/dev/null || true
    echo "Results packaged in $TARBALL (backup: $BACKUP)"
}
trap package_results EXIT

python - <<'PY'
import importlib.util, sys
required = {"rouge_score": "rouge-score", "evalplus": "evalplus", "pynvml": "nvidia-ml-py", "scipy": "scipy"}
missing = [p for m, p in required.items() if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("Missing packages: " + " ".join(missing) + "\nInstall on darmok: python -m pip install --user " + " ".join(missing))
PY

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,power.draw,power.limit,memory.total --format=csv || true
python -c "import torch, transformers; print('PyTorch', torch.__version__, '| Transformers', transformers.__version__, '| CUDA', torch.cuda.is_available())"

python -u llm_energy_gpu.py
