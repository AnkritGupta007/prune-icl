#!/bin/bash -l
#SBATCH -p Contributors
#SBATCH -w GPU48
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH -o /home/a/ankritgupta/projects/prune-icl/artifacts/logs/slurm/pruneicl_runner_%j.out
#SBATCH -e /home/a/ankritgupta/projects/prune-icl/artifacts/logs/slurm/pruneicl_runner_%j.err

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate pruneicl

cd /home/a/ankritgupta/projects/prune-icl

ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl/bin/python

echo "hostname: $(hostname)"
echo "which python: $(which python)"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "ENV_PY: $ENV_PY"

$ENV_PY - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY

$ENV_PY -m src.runner --run_id "$RUN_ID"