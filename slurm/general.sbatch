#!/bin/bash -l
#SBATCH -J sparsegpt_30
#SBATCH -p Contributors
#SBATCH -w GPU1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH -o /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.out
#SBATCH -e /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.err

set -euo pipefail
set -x

ROOT=/home/a/ankritgupta/projects/prune-icl
CODE_DIR=${ROOT}/external/wanda
ENV_NAME=pruneicl
MODEL_DIR="Nothing"
OUT_DIR=/data/ankritgupta/iclprune/wanda

mkdir -p "${ROOT}/logs" "${OUT_DIR}"

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl/bin/python

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

STAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR=${OUT_DIR}/sparsegpt_30/llama31_8b_30_${STAMP}

cd "${CODE_DIR}"

echo "JOB STARTED"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "MODEL_DIR=${MODEL_DIR}"
echo "SAVE_DIR=${SAVE_DIR}"
date
nvidia-smi
"${ENV_PY}" -V
"${ENV_PY}" -c "import torch; print('cuda:', torch.cuda.is_available()); print('count:', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${ENV_PY}" main.py \
  --model meta-llama/Llama-3.1-8B \
  --prune_method sparsegpt \
  --sparsity_ratio 0.3 \
  --sparsity_type unstructured \
  --save "${ROOT}"/results \
  --save_model "${SAVE_DIR}" 

echo "DONE"
echo "Saved pruned model to ${SAVE_DIR}"
