"""
Environment sanity check for Slurm jobs.

Purpose:
- confirm which Python executable is being used
- confirm which conda environment is active
- confirm required packages are importable
"""

from __future__ import annotations

import os
import sys

print("python_executable:", sys.executable)
print("conda_default_env:", os.environ.get("CONDA_DEFAULT_ENV"))
print("conda_prefix:", os.environ.get("CONDA_PREFIX"))

# Try importing required libraries one by one so failures are easy to interpret.
import torch
print("torch_version:", torch.__version__)

import transformers
print("transformers_version:", transformers.__version__)

import yaml
print("pyyaml_ok: True")
