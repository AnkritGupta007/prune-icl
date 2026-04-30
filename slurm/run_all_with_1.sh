#!/bin/bash -l
#SBATCH -J eval_enabled
#SBATCH -p Contributors
#SBATCH -w GPU54
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH -o /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.out
#SBATCH -e /home/a/ankritgupta/projects/prune-icl/logs/%x_%j.err

set -euo pipefail
set -x

ROOT=/home/a/ankritgupta/projects/prune-icl
MANIFEST="${ROOT}/manifests/full-manifest.csv"
ENABLED_VALUE="${1:-1}"

ENV_NAME=pruneicl-eval
ENV_PY=/home/a/ankritgupta/.conda/envs/pruneicl-eval/bin/python

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${ROOT}"

echo "JOB STARTED"
echo "MANIFEST=${MANIFEST}"
echo "ENABLED_VALUE=${ENABLED_VALUE}"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
date
nvidia-smi

"${ENV_PY}" -V
"${ENV_PY}" -c "import transformers, accelerate, torch; print('transformers:', transformers.__version__); print('accelerate:', accelerate.__version__); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu_count:', torch.cuda.device_count()); print('gpu0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

mapfile -t RUN_IDS < <("${ENV_PY}" - <<PY
import pandas as pd

manifest = "${MANIFEST}"
enabled_value = int("${ENABLED_VALUE}")

df = pd.read_csv(manifest)

if "enabled" not in df.columns:
    raise ValueError("Manifest must contain an 'enabled' column")
if "run_id" not in df.columns:
    raise ValueError("Manifest must contain a 'run_id' column")

selected = (
    df[df["enabled"] == enabled_value]["run_id"]
    .dropna()
    .astype(str)
    .tolist()
)

for rid in selected:
    print(rid)
PY
)

echo "TOTAL RUNS FOR enabled=${ENABLED_VALUE}: ${#RUN_IDS[@]}"

if [ "${#RUN_IDS[@]}" -eq 0 ]; then
  echo "No runs found for enabled=${ENABLED_VALUE}"
  exit 0
fi

for RUN_ID in "${RUN_IDS[@]}"; do
  echo "========================================"
  echo "STARTING RUN_ID=${RUN_ID}"
  date

  if "${ENV_PY}" -m src.runner \
      --run_id "${RUN_ID}" \
      --manifest "${MANIFEST}"; then
    echo "FINISHED RUN_ID=${RUN_ID}"
  else
    echo "FAILED RUN_ID=${RUN_ID}"
  fi

  date
done

echo "ALL RUNS FOR enabled=${ENABLED_VALUE} DONE"
date