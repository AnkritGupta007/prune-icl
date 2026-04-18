#!/bin/bash
set -euo pipefail

ROOT=/home/a/ankritgupta/projects/prune-icl
RIA_DIR=${ROOT}/external/ria
ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl/bin/python

export HF_HOME=${ROOT}/.hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets

MODEL="${1:-meta-llama/Llama-3.1-8B}"
PRUNE_METHOD="${PRUNE_METHOD:-magnitude}"   # first run must be magnitude
SPARSITY="${SPARSITY:-0.5}"
MODEL_TAG="${MODEL##*/}"
STAMP=$(date +%Y%m%d_%H%M%S)

SAVE_MODEL="${ROOT}/runs/phase1/pruned/${MODEL_TAG}_${PRUNE_METHOD}_s${SPARSITY}_${STAMP}"
mkdir -p "${ROOT}/runs/phase1/pruned"
mkdir -p "${ROOT}/runs/phase1/meta"

cd "${RIA_DIR}"

echo "MODEL=${MODEL}"
echo "PRUNE_METHOD=${PRUNE_METHOD}"
echo "SPARSITY=${SPARSITY}"
echo "SAVE_MODEL=${SAVE_MODEL}"

"${ENV_PY}" main.py \
  --model "${MODEL}" \
  --prune_method "${PRUNE_METHOD}" \
  --sparsity_ratio "${SPARSITY}" \
  --sparsity_type unstructured \
  --cache_dir "${HF_HOME}/model_cache" \
  --eval_dataset wikitext2 \
  --save \
  --save_model "${SAVE_MODEL}"

echo "${SAVE_MODEL}" > "${ROOT}/runs/phase1/meta/last_pruned_model.txt"
echo "Pruned model saved to: ${SAVE_MODEL}"