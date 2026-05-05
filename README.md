prune-icl
Utilities for pruning causal language models, running in-context-learning evaluations, curating result ledgers, and producing summary plots. The repository is organized around three stages:
Prune a base model with a supported pruning method.
Evaluate dense or pruned models on `lm-evaluation-harness` tasks and the custom synthetic linear ICL task.
Curate and plot the collected JSONL results.
---
Repository layout
```text
configs/models/            Model YAML configs used by the runner and eval scripts
manifests/                 CSV manifests of experiment runs
scripts/                   Quick local shell wrappers for one-off prune/eval runs
slurm/                     Slurm job scripts for pruning/evaluation on HPC
src/runner.py              Manifest-driven experiment runner
src/eval/                  Evaluation wrappers and result parsers
src/prune/                 Method registry and backend mapping
src/utils/                 Environment, model, manifest, and checkpoint helpers
curate_results.py          Selects the best result per manifest run_id
plot_curated_results.py    Generates CSV summaries and plots from curated results
requirements.txt           Python dependencies
```
---
Requirements
System requirements
Linux or HPC environment with Bash.
Conda or another Python environment manager.
Python 3.10 is the expected version.
NVIDIA GPU with CUDA-compatible PyTorch for model pruning/evaluation.
Git, because setup requires external pruning/evaluation repositories.
Hugging Face account/token with access to gated Llama models such as `meta-llama/Llama-3.1-8B` and/or `meta-llama/Llama-3.2-3B`.
Sufficient disk space for Hugging Face caches, external repos, evaluation artifacts, and pruned checkpoints.
Python requirements
The repository includes `requirements.txt`:
```text
torch==2.10.0
transformers==4.45
pyyaml
huggingface_hub
datasets
accelerate
sentencepiece
protobuf
scipy
numpy
pandas
matplotlib
tqdm
evaluate
jsonlines
```
If your cluster does not provide the pinned PyTorch build, install the CUDA/PyTorch version recommended for your machine, then install the rest of the requirements.
External repositories
The setup script expects these external tools under `external/`:
`RIA`
`lm-evaluation-harness`
`wanda`
`sparsegpt`
`AlphaPruning`
The code and Slurm scripts currently use a few different external directory names, so verify them before running:
`setup_repo.sh` clones RIA to `external/ria_core`, but `scripts/prune_once.sh` expects `external/ria`.
`setup_repo.sh` clones Wanda to `external/wanda_official`, but `slurm/general.sbatch` expects `external/wanda`.
You can either edit the scripts to match your directory names or create symlinks, for example:
```bash
ln -s external/ria_core external/ria
ln -s external/wanda_official external/wanda
```
---
Installation
From the repository root:
```bash
cd prune-icl-main

conda create -y -n pruneicl python=3.10
conda activate pruneicl

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```
Install `lm-evaluation-harness` in editable mode after cloning it:
```bash
mkdir -p external

git clone https://github.com/EleutherAI/lm-evaluation-harness.git external/lm-evaluation-harness
python -m pip install -e external/lm-evaluation-harness
```
Log in to Hugging Face:
```bash
huggingface-cli login
```
Create the expected output directories:
```bash
mkdir -p logs \
  artifacts/eval_jsonl \
  artifacts/summaries \
  artifacts/plots_phase1 \
  runs/phase1/pruned \
  runs/phase1/meta \
  runs/phase1/eval
```
Optional: bootstrap script
The repository includes `setup_repo.sh`, but check it before running. It currently computes `REPO_ROOT` as the parent of the script directory, which is appropriate only if the script is inside a subdirectory. Since this copy places `setup_repo.sh` at the repository root, edit the `REPO_ROOT` assignment if needed.
After confirming paths:
```bash
ENV_NAME=pruneicl PYTHON_VERSION=3.10 bash setup_repo.sh
```
---
Path configuration before running
Several scripts contain hard-coded HPC paths. Update them for your machine before launching jobs.
Files to edit
```text
scripts/prune_once.sh
scripts/eval_pruned.sh
slurm/general.sbatch
slurm/run_eval.sh
slurm/run_all_with_1.sh
slurm/run_manifest_eval.sbatch
src/utils/checkpoint_resolver.py
```
Important variables
`ROOT`: repository root.
`ENV_PY`: Python executable inside your conda environment.
`ENV_LMEVAL`: `lm-eval` executable inside your conda environment.
`CODE_DIR` or `RIA_DIR`: external pruning repo path.
`OUT_DIR`: parent directory where pruned checkpoints are saved.
`PRUNED_MODEL_ROOT` in `src/utils/checkpoint_resolver.py`: parent directory used by `src.runner` to locate pruned checkpoints.
Slurm partition/node settings such as `#SBATCH -p`, `#SBATCH -w`, `--mem`, and `--gpus`.
The manifest-driven runner expects pruned checkpoints under this pattern:
```text
<PRUNED_MODEL_ROOT>/<method>_<sparsity>/<model_key>_<sparsity>_<timestamp>/
```
For example:
```text
/data/ankritgupta/iclprune/wanda/wanda_30/llama32_3b_30_20260430_123456/
```
Each resolved checkpoint must contain at least:
```text
config.json
model.safetensors.index.json
```
---
Sanity checks
Run these after installation:
```bash
conda activate pruneicl

python -m src.utils.check_env
python -m src.utils.check_manifest
```
Optional dense model smoke test:
```bash
python -m src.eval.smoke_dense_model --config configs/models/llama32_3b.yaml
```
To inspect a manifest run without executing it:
```bash
python -m src.runner \
  --run_id phase1__llama32_3b__dense__sp0__schuniform__calnone__bbh_zs__seed13 \
  --manifest manifests/full-manifest.csv \
  --dry_run
```
---
Pruning
Option A: quick one-off pruning wrapper
Edit `ROOT`, `RIA_DIR`, and `ENV_PY` inside `scripts/prune_once.sh`, then run:
```bash
conda activate pruneicl

PRUNE_METHOD=magnitude \
SPARSITY=0.5 \
bash scripts/prune_once.sh meta-llama/Llama-3.1-8B
```
Notes:
`SPARSITY` in `scripts/prune_once.sh` is a ratio, so `0.5` means 50% sparsity.
The script writes the latest checkpoint path to:
```text
runs/phase1/meta/last_pruned_model.txt
```
Option B: Slurm pruning job
Edit the Slurm header and paths in `slurm/general.sbatch`, then submit:
```bash
sbatch slurm/general.sbatch wanda 30 meta-llama/Llama-3.1-8B llama31_8b
```
Arguments:
```text
1. PRUNE_METHOD       Example: wanda, sparsegpt, magnitude
2. SPARSITY_PERCENT   Example: 30, 50, 70, 90
3. MODEL_NAME         Hugging Face model ID
4. MODEL_KEY          Local model key used by configs/manifests
```
The Slurm script converts `SPARSITY_PERCENT=30` to `--sparsity_ratio 0.3` and saves the model under:
```text
<OUT_DIR>/<PRUNE_METHOD>_<SPARSITY_PERCENT>/<MODEL_KEY>_<SPARSITY_PERCENT>_<TIMESTAMP>/
```
Supported manifest methods are defined in `src/prune/registry.py`:
```text
dense, magnitude, wanda, wandaplus, ria, sparsegpt, wanda_owl
```
---
Evaluation
Evaluation results are appended to:
```text
artifacts/eval_jsonl/results.jsonl
```
Option A: evaluate the last pruned model with the quick wrapper
Edit `ROOT` and `ENV_LMEVAL` in `scripts/eval_pruned.sh`, then run:
```bash
conda activate pruneicl

TASKS=mmlu \
LIMIT=20 \
BATCH=auto \
bash scripts/eval_pruned.sh
```
By default, the script reads the checkpoint path from:
```text
runs/phase1/meta/last_pruned_model.txt
```
To evaluate a specific checkpoint:
```bash
TASKS=mmlu LIMIT=200 bash scripts/eval_pruned.sh /path/to/pruned/checkpoint
```
The quick wrapper runs both:
zero-shot evaluation with `--num_fewshot 0`
five-shot evaluation with `--num_fewshot 5`
Option B: run one manifest entry locally
Dense run example:
```bash
python -m src.runner \
  --run_id phase1__llama32_3b__dense__sp0__schuniform__calnone__bbh_zs__seed13 \
  --manifest manifests/full-manifest.csv
```
Pruned run example:
```bash
python -m src.runner \
  --run_id phase1__llama32_3b__wanda__sp30__schuniform__calA__bbh_zs__seed13 \
  --manifest manifests/full-manifest.csv
```
For pruned runs, `src.runner` automatically resolves the checkpoint using `src/utils/checkpoint_resolver.py`. If it cannot find the checkpoint, update `PRUNED_MODEL_ROOT` or move/symlink the checkpoint into the expected folder pattern.
Option C: run selected manifest entries with Slurm
Submit one or more run IDs:
```bash
sbatch slurm/run_eval.sh \
  phase1__llama32_3b__dense__sp0__schuniform__calnone__bbh_zs__seed13 \
  phase1__llama32_3b__wanda__sp30__schuniform__calA__bbh_zs__seed13
```
Run every row whose manifest `enabled` value is `1`:
```bash
sbatch slurm/run_all_with_1.sh 1
```
Manual lm-evaluation-harness wrapper
You can call the project wrapper directly:
```bash
python -m src.eval.run_lm_eval \
  --config configs/models/llama32_3b.yaml \
  --task mmlu \
  --num_fewshot 5 \
  --limit 20 \
  --output_json artifacts/manual/mmlu_5shot
```
Evaluate a local pruned checkpoint:
```bash
python -m src.eval.run_lm_eval \
  --config configs/models/llama32_3b.yaml \
  --checkpoint_path /path/to/pruned/checkpoint \
  --task mmlu \
  --num_fewshot 5 \
  --limit 20 \
  --output_json artifacts/manual/pruned_mmlu_5shot
```
Synthetic linear ICL evaluation
Dense synthetic run:
```bash
python -m src.eval.run_synth_icl \
  --config configs/models/llama32_3b.yaml \
  --num_fewshot 8 \
  --limit 200 \
  --seed 13 \
  --output_json artifacts/manual/synth_dense_8shot.json
```
Pruned synthetic run:
```bash
python -m src.eval.run_synth_icl \
  --config configs/models/llama32_3b.yaml \
  --checkpoint_path /path/to/pruned/checkpoint \
  --num_fewshot 8 \
  --limit 200 \
  --seed 13 \
  --output_json artifacts/manual/synth_pruned_8shot.json
```
If you run synthetic evaluation manually and want to append it to the shared JSONL ledger, parse it with a matching manifest `run_id`:
```bash
python -m src.eval.parse_synth_eval_result \
  --run_id <RUN_ID_FROM_MANIFEST> \
  --input_json artifacts/manual/synth_pruned_8shot.json \
  --output_jsonl artifacts/eval_jsonl/results.jsonl \
  --manifest manifests/full-manifest.csv
```
---
Curating results
After multiple evaluations, curate the shared JSONL ledger into one best row per manifest run ID:
```bash
python curate_results.py \
  --manifest manifests/full-manifest.csv \
  --results artifacts/eval_jsonl/results.jsonl \
  --best-output artifacts/eval_jsonl/curated_best.jsonl \
  --dupes-output artifacts/eval_jsonl/duplicate_resolution.json
```
Outputs:
```text
artifacts/eval_jsonl/curated_best.jsonl
artifacts/eval_jsonl/duplicate_resolution.json
```
Curation rules implemented in `curate_results.py`:
Phase 0: choose the best `metric_value`; break ties by latest timestamp.
Phase 1 BBH COT tasks: prefer `limit=50` where available.
Phase 1 non-BBH tasks: prefer `limit=200` where available.
Duplicate candidates are documented in `duplicate_resolution.json`.
---
Plotting
Generate plots and summary CSVs from the curated file:
```bash
python plot_curated_results.py \
  --input artifacts/eval_jsonl/curated_best.jsonl \
  --outdir artifacts/plots_phase1
```
The plotting script creates files such as:
```text
tidy_results.csv
dense_baselines.csv
retention_results.csv
fewshot_gain_results.csv
best_method_by_task_sparsity.csv
raw_* plots
retention_* plots
fewshot_gain_* plots
synthetic_icl_* plots
heatmap_avg_retention_fewshot.png
```
Typical workflow:
```bash
python curate_results.py \
  --manifest manifests/full-manifest.csv \
  --results artifacts/eval_jsonl/results.jsonl \
  --best-output artifacts/eval_jsonl/curated_best.jsonl \
  --dupes-output artifacts/eval_jsonl/duplicate_resolution.json

python plot_curated_results.py \
  --input artifacts/eval_jsonl/curated_best.jsonl \
  --outdir artifacts/plots_phase1
```
---
End-to-end example
```bash
# 1. Activate environment
conda activate pruneicl

# 2. Check environment and manifest
python -m src.utils.check_env
python -m src.utils.check_manifest

# 3. Prune a model
sbatch slurm/general.sbatch wanda 30 meta-llama/Llama-3.1-8B llama31_8b

# 4. Run an evaluation once the checkpoint exists
python -m src.runner \
  --run_id phase1__llama32_3b__wanda__sp30__schuniform__calA__bbh_zs__seed13 \
  --manifest manifests/full-manifest.csv

# 5. Curate results
python curate_results.py \
  --manifest manifests/full-manifest.csv \
  --results artifacts/eval_jsonl/results.jsonl \
  --best-output artifacts/eval_jsonl/curated_best.jsonl \
  --dupes-output artifacts/eval_jsonl/duplicate_resolution.json

# 6. Create plots
python plot_curated_results.py \
  --input artifacts/eval_jsonl/curated_best.jsonl \
  --outdir artifacts/plots_phase1
```
---
Troubleshooting
`FileNotFoundError: Missing checkpoint parent directory`
`src.runner` could not find the pruned model directory expected by `src/utils/checkpoint_resolver.py`. Update `PRUNED_MODEL_ROOT` or symlink your checkpoint directory into the expected layout:
```text
<PRUNED_MODEL_ROOT>/<method>_<sparsity>/<model_key>_<sparsity>_<timestamp>/
```
`lm_eval: command not found`
Install the harness in the active environment:
```bash
python -m pip install -e external/lm-evaluation-harness
which lm_eval
```
Hugging Face gated model errors
Make sure your account has accepted the model license and that you are logged in:
```bash
huggingface-cli login
```
The file `src/utils/check_llama_access.py` contains a token placeholder. Do not commit real tokens into source files.
External repo path errors
Check whether your external directories match what the scripts expect:
```bash
ls external
```
Either edit the script paths or create symlinks:
```bash
ln -s external/ria_core external/ria
ln -s external/wanda_official external/wanda
```
Slurm job runs in the wrong environment
Check the `ENV_NAME`, `ENV_PY`, and `source /apps/anaconda3/etc/profile.d/conda.sh` lines in the Slurm scripts. These are cluster-specific and may need to be replaced for your environment.
---
Main artifacts
Raw/parsed evaluation ledger: `artifacts/eval_jsonl/results.jsonl`
Curated best results: `artifacts/eval_jsonl/curated_best.jsonl`
Duplicate-resolution report: `artifacts/eval_jsonl/duplicate_resolution.json`
Plot outputs: `artifacts/plots_phase1/`
Per-run outputs: `artifacts/<phase>/<run_id>/`
