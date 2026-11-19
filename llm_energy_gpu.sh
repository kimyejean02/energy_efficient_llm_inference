#!/bin/bash
set -euo pipefail

# Activate the UTCS-provided PyTorch/CUDA environment.
source /lusr/opt/miniconda/bin/activate pytorch-cuda

# Store Hugging Face models in scratch space
export HF_HOME="/scratch/cluster/ishaj/huggingface" # Change directory
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$HF_HUB_CACHE"

# Enable the response-quality grading section in llm_energy_gpu.py.
export RUN_GRADING=1

# Fail early with a clear message if grading dependencies are unavailable.
python - <<'PY'
import importlib.util
import sys

required = {
    "rouge_score": "rouge-score",
    "evalplus": "evalplus",
}
missing = [package for module, package in required.items()
           if importlib.util.find_spec(module) is None]
if missing:
    print("Missing grading dependencies: " + ", ".join(missing), file=sys.stderr)
    print("Install them in the pytorch-cuda environment before submitting:", file=sys.stderr)
    print("python -m pip install --user " + " ".join(missing), file=sys.stderr)
    sys.exit(1)
PY

echo "Python: $(which python)"
python --version

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -u llm_energy_gpu.py
