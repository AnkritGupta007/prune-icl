#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Project bootstrap script for prune-icl on GAIVI / similar HPC
#
# What this script does:
# 1. Verifies it is being run from the repo root
# 2. Creates or reuses the conda environment
# 3. Installs Python requirements
# 4. Clones external repositories into external/
# 5. Optionally checks out pinned commits
# 6. Creates standard output/log/artifact directories
# 7. Prints next validation commands
# ============================================================

# ---------- User-configurable defaults ----------
ENV_NAME="${ENV_NAME:-prune-icl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${REPO_ROOT}/external"

# Pinned commits already used in your project
RIA_COMMIT="7bc55b06924659ea977f7804b773ce3ef0d55c2f"
HARNESS_COMMIT="d800e04dcb1ce96791d8b2926cf0cc7703d58457"
WANDA_COMMIT="8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6"
SPARSEGPT_COMMIT="147d2159dc4f3e9f73e47b32c04d7b3708f44436"
ALPHAPRUNING_COMMIT="5f0e9845549ec8ee6dc395f1410566f26cc9e54e"

# ---------- Helper functions ----------
msg() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: Required command not found: $1"
    exit 1
  }
}

clone_or_update_repo() {
  local repo_url="$1"
  local repo_dir="$2"
  local repo_commit="${3:-}"

  if [[ ! -d "${repo_dir}/.git" ]]; then
    echo "[clone] ${repo_url} -> ${repo_dir}"
    git clone "${repo_url}" "${repo_dir}"
  else
    echo "[exists] ${repo_dir}"
  fi

  if [[ -n "${repo_commit}" ]]; then
    echo "[checkout] ${repo_dir} @ ${repo_commit}"
    git -C "${repo_dir}" fetch --all --tags
    git -C "${repo_dir}" checkout "${repo_commit}"
  fi
}

# ---------- Sanity checks ----------
msg "Checking prerequisites"

need_cmd git
need_cmd bash

if [[ ! -f "${REPO_ROOT}/README.md" && ! -d "${REPO_ROOT}/src" ]]; then
  echo "ERROR: Could not verify repo root at: ${REPO_ROOT}"
  echo "Please run this script from inside the project repo."
  exit 1
fi

# ---------- Load conda ----------
msg "Loading conda"

if [[ -f /apps/anaconda3/etc/profile.d/conda.sh ]]; then
  # GAIVI / USF style path
  source /apps/anaconda3/etc/profile.d/conda.sh
else
  echo "ERROR: Could not find conda.sh at /apps/anaconda3/etc/profile.d/conda.sh"
  echo "Edit scripts/setup_repo.sh if your conda installation lives elsewhere."
  exit 1
fi

need_cmd conda

# ---------- Create or reuse conda env ----------
msg "Creating or reusing conda environment: ${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[env exists] ${ENV_NAME}"
else
  echo "[env create] conda create -n ${ENV_NAME} python=${PYTHON_VERSION}"
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

ENV_PY="$(conda run -n "${ENV_NAME}" which python | tail -n 1 || true)"
echo "[python] ${ENV_PY:-unknown}"

# ---------- Upgrade pip tooling ----------
msg "Upgrading pip/setuptools/wheel"

python -m pip install --upgrade pip setuptools wheel

# ---------- Install project requirements ----------
msg "Installing Python requirements"

if [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
  python -m pip install -r "${REPO_ROOT}/requirements.txt"
else
  echo "WARNING: requirements.txt not found. Skipping pip install."
fi

# ---------- Prepare external dir ----------
msg "Preparing external repositories"

mkdir -p "${EXTERNAL_DIR}"

clone_or_update_repo \
  "https://github.com/IST-DASLab/RIA.git" \
  "${EXTERNAL_DIR}/ria_core" \
  "${RIA_COMMIT}"

clone_or_update_repo \
  "https://github.com/EleutherAI/lm-evaluation-harness.git" \
  "${EXTERNAL_DIR}/lm-evaluation-harness" \
  "${HARNESS_COMMIT}"

clone_or_update_repo \
  "https://github.com/locuslab/wanda.git" \
  "${EXTERNAL_DIR}/wanda_official" \
  "${WANDA_COMMIT}"

clone_or_update_repo \
  "https://github.com/IST-DASLab/sparsegpt.git" \
  "${EXTERNAL_DIR}/sparsegpt_official" \
  "${SPARSEGPT_COMMIT}"

clone_or_update_repo \
  "https://github.com/kaiqiangh/AlphaPruning.git" \
  "${EXTERNAL_DIR}/AlphaPruning" \
  "${ALPHAPRUNING_COMMIT}"

# ---------- Install harness editable ----------
msg "Installing lm-evaluation-harness in editable mode"

python -m pip install -e "${EXTERNAL_DIR}/lm-evaluation-harness"

# ---------- Create standard folders ----------
msg "Creating standard project directories"

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/artifacts"
mkdir -p "${REPO_ROOT}/artifacts/summaries"
mkdir -p "${REPO_ROOT}/artifacts/eval_jsonl"
mkdir -p "${REPO_ROOT}/manifests"
mkdir -p "${REPO_ROOT}/scripts"
mkdir -p "${REPO_ROOT}/slurm"

# ---------- Print external repo status ----------
msg "External repository status"

for d in \
  "${EXTERNAL_DIR}/ria_core" \
  "${EXTERNAL_DIR}/lm-evaluation-harness" \
  "${EXTERNAL_DIR}/wanda_official" \
  "${EXTERNAL_DIR}/sparsegpt_official" \
  "${EXTERNAL_DIR}/AlphaPruning"
do
  echo "----- ${d}"
  git -C "${d}" rev-parse HEAD
done

# ---------- Final summary ----------
msg "Bootstrap complete"

echo "Conda environment: ${ENV_NAME}"
echo "Repo root: ${REPO_ROOT}"
echo
echo "Recommended next checks:"
echo "  conda activate ${ENV_NAME}"
echo "  python -m src.utils.check_env"
echo "  python -m src.eval.smoke_dense_model --config configs/models/llama31_8b.yaml"
echo
echo "Recommended Slurm checks:"
echo "  sbatch slurm/check_env.sbatch"
echo "  sbatch slurm/smoke_dense_model.sbatch"
echo
echo "Done."