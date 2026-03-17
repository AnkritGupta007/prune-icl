#!/bin/bash -l
#SBATCH -p Contributors
#SBATCH -w GPU50
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH -o /home/a/ankritgupta/projects/prune-icl/logs/pruneicl_dense_smoke_%j.out
#SBATCH -e /home/a/ankritgupta/projects/prune-icl/logs/pruneicl_dense_smoke_%j.err

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate pruneicl

cd /home/a/ankritgupta/projects/prune-icl

ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl/bin/python

echo "which python: $(which python)"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "ENV_PY: $ENV_PY"
echo "HF_TOKEN_SET: ${HF_TOKEN:+yes}"

# Pass token explicitly to the Python process.
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

$ENV_PY -m src.eval.smoke_dense_model --config configs/models/llama31_8b.yaml