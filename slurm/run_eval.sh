#!/bin/bash -l
#SBATCH -J eval_run
#SBATCH -p Contributors
#SBATCH -w GPU50
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH -o /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.out
#SBATCH -e /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.err

set -euo pipefail
set -x

if [ $# -lt 1 ]; then
  echo "Usage: sbatch run_eval.sh <RUN_ID>"
  exit 1
fi

RUN_ID="$1"
MANIFEST=/manifests/full-manifest.csv

ROOT=/home/a/ankritgupta/projects/prune-icl
ENV_NAME=pruneicl-eval
ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl-eval/bin/python

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${ROOT}"

echo "JOB STARTED"
echo "RUN_ID=${RUN_ID}"
echo "MANIFEST=${MANIFEST}"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
date
nvidia-smi

"${ENV_PY}" -V
"${ENV_PY}" -c "import transformers, accelerate, torch; print('transformers:', transformers.__version__); print('accelerate:', accelerate.__version__); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

"${ENV_PY}" -m src.runner \
  --run_id "${RUN_ID}" \
  --manifest "${MANIFEST}"

echo "DONE"
date