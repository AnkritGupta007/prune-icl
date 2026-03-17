#!/bin/bash -l
#SBATCH -p Contributors
#SBATCH -w GPU50
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/ankritgupta/projects/prune-icl/logs/pruneicl_mmlu_dense_smoke_%j.out
#SBATCH -e /home/a/ankritgupta/projects/prune-icl/logs/pruneicl_mmlu_dense_smoke_%j.err

# Make conda commands available in the batch shell.
source /apps/anaconda3/etc/profile.d/conda.sh

# Activate the project environment for shell consistency.
conda activate pruneicl

cd /home/a/ankritgupta/projects/prune-icl

# Use the exact env interpreter to avoid path ambiguity.
ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl/bin/python

echo "which python: $(which python)"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "ENV_PY: $ENV_PY"

# Run a small dense MMLU smoke test through the harness wrapper.
$ENV_PY -m src.eval.run_lm_eval \
  --config configs/models/llama31_8b.yaml \
  --task mmlu \
  --num_fewshot 5 \
  --limit 20 \
  --output_json artifacts/summaries/mmlu_dense_smoke.json