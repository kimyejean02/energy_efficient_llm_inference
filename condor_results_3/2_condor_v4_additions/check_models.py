"""Run on darmok BEFORE submitting: downloads every model and reports problems early.

    export HF_HOME=/scratch/cluster/$USER/huggingface
    python check_models.py
"""
import json
import os

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "agentlans/Qwen2.5-1.5B-Instruct-Summarizer",
]

problems = []
for mid in MODELS:
    print(f"\n=== {mid}")
    try:
        path = snapshot_download(mid)
    except Exception as e:
        print(f"  FAILED to download: {type(e).__name__}: {e}")
        problems.append(mid)
        continue
    files = sorted(os.listdir(path))
    full = any(f.endswith(".safetensors") and not f.startswith("adapter") for f in files) or "pytorch_model.bin" in files
    adapter = "adapter_config.json" in files
    print(f"  files: {', '.join(files)}")
    print(f"  full weights: {full} | LoRA adapter: {adapter}")
    if adapter and not full:
        print("  NOTE: adapter only -> install peft:  python -m pip install --user peft")
    tc = os.path.join(path, "tokenizer_config.json")
    if os.path.exists(tc):
        has_template = "chat_template" in json.load(open(tc))
        print(f"  chat template in tokenizer: {has_template}"
              + ("" if has_template else "  (the experiment will use the Qwen2.5 template)"))

# Show the summarizer's own usage notes, so you can check its expected prompt format.
try:
    readme = hf_hub_download("agentlans/Qwen2.5-1.5B-Instruct-Summarizer", "README.md")
    print("\n=== Summarizer model card (first 60 lines) ===")
    print("".join(open(readme, encoding="utf-8").readlines()[:60]))
except Exception as e:
    print(f"\nCould not read the summarizer's README: {e}")

print("\nAll models downloaded." if not problems else f"\nPROBLEMS with: {problems}")
