#!/bin/bash -l
#SBATCH -p Contributors
#SBATCH -w GPU50
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH -o /home/a/ankritgupta/projects/prune-icl/logs/pruneicl_check_env_%j.out
#SBATCH -e /home/a/ankritgupta/projects/prune-icl/logs/pruneicl_check_env_%j.err

# Make conda commands available in the batch shell.
source /apps/anaconda3/etc/profile.d/conda.sh

# Activate the environment, even though we will still call its python explicitly.
conda activate pruneicl

cd /home/a/ankritgupta/projects/prune-icl

# Explicit path to the intended environment interpreter.
ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl/bin/python

echo "which python: $(which python)"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "ENV_PY: $ENV_PY"

# Run the check script with the environment's Python directly.
$ENV_PY -m src.utils.check_env