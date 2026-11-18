# Running on Condor:
1. SSH into darmok.cs.utexas.edu.
2. Copy the following files: benchmark_prompts.csv, llm_energy_gpu.py, llm_energy_gpu.sh, llm_energy_gpu.sub.
3. Edit llm_energy_gpu.sub to send notification to your email
4. source /lusr/opt/miniconda/bin/activate pytorch-cuda
5. python -m pip install --user transformers rouge-score evalplus nvidia-ml-py
6. condor_submit llm_energy_gpu.sub
