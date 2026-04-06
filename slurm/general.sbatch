#!/bin/bash -l
#SBATCH -J prune_run
#SBATCH -p Contributors
#SBATCH -w GPU1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH -o /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.out
#SBATCH -e /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.err

set -euo pipefail
set -x

if [ $# -lt 2 ]; then
  echo "Usage: sbatch run_prune.sh <PRUNE_METHOD> <SPARSITY_PERCENT> [MODEL_NAME] [MODEL_KEY]"
  echo "Example: sbatch -J wanda30 run_prune.sh wanda 30 meta-llama/Llama-3.1-8B llama31_8b"
  exit 1
fi

PRUNE_METHOD="$1"                  # e.g. wanda, sparsegpt
SPARSITY_PERCENT="$2"              # e.g. 30, 50, 70
MODEL_NAME="${3:-meta-llama/Llama-3.1-8B}"
MODEL_KEY="${4:-llama31_8b}"

ROOT=/home/a/ankritgupta/projects/prune-icl
CODE_DIR="${ROOT}/external/wanda"
ENV_NAME=pruneicl
ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl/bin/python
OUT_DIR=/data/ankritgupta/iclprune/wanda

mkdir -p "${ROOT}/logs" "${OUT_DIR}"

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Convert percent like 30 -> 0.3
SPARSITY_RATIO=$(python - <<PY
sp = float("${SPARSITY_PERCENT}")
print(sp / 100.0)
PY
)

STAMP=$(date +%Y%m%d_%H%M%S)
SAVE_PARENT="${OUT_DIR}/${PRUNE_METHOD}_${SPARSITY_PERCENT}"
SAVE_DIR="${SAVE_PARENT}/${MODEL_KEY}_${SPARSITY_PERCENT}_${STAMP}"

mkdir -p "${SAVE_PARENT}"

cd "${CODE_DIR}"

echo "JOB STARTED"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "PRUNE_METHOD=${PRUNE_METHOD}"
echo "SPARSITY_PERCENT=${SPARSITY_PERCENT}"
echo "SPARSITY_RATIO=${SPARSITY_RATIO}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_KEY=${MODEL_KEY}"
echo "SAVE_DIR=${SAVE_DIR}"
date
nvidia-smi
"${ENV_PY}" -V
"${ENV_PY}" -c "import torch; print('cuda:', torch.cuda.is_available()); print('count:', torch.cuda.device_count()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

"${ENV_PY}" main.py \
  --model "${MODEL_NAME}" \
  --prune_method "${PRUNE_METHOD}" \
  --sparsity_ratio "${SPARSITY_RATIO}" \
  --sparsity_type unstructured \
  --save "${ROOT}/results" \
  --save_model "${SAVE_DIR}"

echo "DONE"
echo "Saved pruned model to ${SAVE_DIR}"
date