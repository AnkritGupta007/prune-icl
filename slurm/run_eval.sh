#!/bin/bash -l
#SBATCH -J eval_run
#SBATCH -p Contributors
#SBATCH -w GPU1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH -o /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.out
#SBATCH -e /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.err

set -euo pipefail
set -x

if [ $# -lt 1 ]; then
  echo "Usage: sbatch run_eval.sh <RUN_ID1> [RUN_ID2] [RUN_ID3] ..."
  exit 1
fi

ROOT=/home/a/ankritgupta/projects/prune-icl
MANIFEST="${ROOT}/manifests/full-manifest.csv"

ENV_NAME=pruneicl-eval
ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl-eval/bin/python

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${ROOT}"

echo "JOB STARTED"
echo "MANIFEST=${MANIFEST}"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "RUN COUNT=$#"
date
nvidia-smi

"${ENV_PY}" -V
"${ENV_PY}" -c "import transformers, accelerate, torch; print('transformers:', transformers.__version__); print('accelerate:', accelerate.__version__); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

for RUN_ID in "$@"; do
  echo "========================================"
  echo "STARTING RUN_ID=${RUN_ID}"
  date

  "${ENV_PY}" -m src.runner \
    --run_id "${RUN_ID}" \
    --manifest "${MANIFEST}"

  echo "FINISHED RUN_ID=${RUN_ID}"
  date
done

echo "ALL RUNS DONE"
date