#!/bin/bash
set -euo pipefail

ROOT=/home/a/ankritgupta/projects/prune-icl
ENV_LMEVAL=/home/a/ankritgupta/.conda/envs/pruneicl/bin/lm-eval

export HF_HOME=${ROOT}/.hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets

PRUNED_MODEL="${1:-$(cat ${ROOT}/runs/phase1/meta/last_pruned_model.txt)}"
TASKS="${TASKS:-mmlu}"   # replace with your exact previously working task/group string if needed
LIMIT="${LIMIT:-20}"
BATCH="${BATCH:-auto}"

STAMP=$(date +%Y%m%d_%H%M%S)
MODEL_TAG="${PRUNED_MODEL##*/}"
OUT_BASE="${ROOT}/runs/phase1/eval/${MODEL_TAG}_${STAMP}"

mkdir -p "${OUT_BASE}"
cd "${ROOT}"

echo "PRUNED_MODEL=${PRUNED_MODEL}"
echo "TASKS=${TASKS}"
echo "LIMIT=${LIMIT}"
echo "OUT_BASE=${OUT_BASE}"

"${ENV_LMEVAL}" run \
  --model hf \
  --model_args pretrained="${PRUNED_MODEL}",dtype=float16 \
  --tasks "${TASKS}" \
  --num_fewshot 0 \
  --limit "${LIMIT}" \
  --batch_size "${BATCH}" \
  --device cuda:0 \
  --output_path "${OUT_BASE}/zero"

"${ENV_LMEVAL}" run \
  --model hf \
  --model_args pretrained="${PRUNED_MODEL}",dtype=float16 \
  --tasks "${TASKS}" \
  --num_fewshot 5 \
  --limit "${LIMIT}" \
  --batch_size "${BATCH}" \
  --device cuda:0 \
  --output_path "${OUT_BASE}/five"

echo "${OUT_BASE}" > "${ROOT}/runs/phase1/meta/last_pruned_eval_dir.txt"
echo "Pruned eval finished: ${OUT_BASE}"