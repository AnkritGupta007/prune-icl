#!/bin/bash -l
#SBATCH -p Contributors
#SBATCH -w GPU50
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH -o /home/a/ankritgupta/projects/logs/test_modern_gpu_%j.out
#SBATCH -e /home/a/ankritgupta/projects/logs/test_modern_gpu_%j.err

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate prune-icl

python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = x @ y
    print("matrix multiply worked:", z.shape)
PY