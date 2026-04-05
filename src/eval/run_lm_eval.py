"""
Thin wrapper for running lm-evaluation-harness from this project.

Purpose:
- keep benchmark execution inside our repo structure
- control output locations
- make later integration with the main runner easier

Current use:
- small dense MMLU smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from src.utils.model_config import load_model_config
from src.utils.io import ensure_dir


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model YAML config, e.g. configs/models/llama31_8b.yaml",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mmlu",
        help="Harness task name. For now we use mmlu.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples to use.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=20,
        help="Small smoke-test limit. Can be int or float depending on harness behavior.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Where to save the raw harness result JSON.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional local checkpoint directory. If provided, use it instead of hf_model_name."
    )
    args = parser.parse_args()

    # Load model settings from YAML.
    cfg = load_model_config(args.config)
    hf_model_name = cfg["hf_model_name"]

    hf_model_name = cfg["hf_model_name"]
    model_source = args.checkpoint_path if args.checkpoint_path else hf_model_name



    dtype_name = cfg.get("torch_dtype", "bfloat16")

    # Make sure parent output directory exists.
    output_path = Path(args.output_json)
    ensure_dir(str(output_path.parent))

    # Build the harness command.
    #
    # We use:
    # - model hf
    # - pretrained=<hf model name>
    # - dtype=<dtype string from YAML>
    # - device=cuda:0
    #
    # output_path tells harness to save the result JSON directly.
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_source},device_map=auto",
        "--tasks", args.task,
        "--num_fewshot", str(args.num_fewshot),
        "--limit", str(args.limit),
        "--device", "cuda:0",
        "--output_path", str(output_path),
    ]

    print("Running command:")
    print(" ".join(cmd))


    print("Resolved model source:", model_source)
    print("Final lm_eval command:", " ".join(cmd))

    # Run the harness process and stop on failure.
    subprocess.run(cmd, check=True)

    # After harness finishes, print a short confirmation.
    print("\nHarness run completed.")
    print("Raw results saved to:", output_path)


if __name__ == "__main__":
    main()