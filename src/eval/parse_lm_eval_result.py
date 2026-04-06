"""
Parse a raw lm-evaluation-harness JSON result into a flat experiment record.

Purpose:
- turn large harness JSON outputs into compact, project-friendly records
- append those records to a JSONL ledger for later CSV aggregation

Current support:
- MMLU group-level score extraction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.io import get_run_row, ensure_dir

def pick_metric(metric_block: dict) -> tuple[str, float, float | None]:
    """
    Pick the main scalar metric from an lm-eval result block.

    Preference order:
    1. acc,none
    2. exact_match,strict-match
    3. exact_match,flexible-extract
    4. exact_match,none
    5. acc_norm,none
    """
    candidates = [
        "acc,none",
        "exact_match,get-answer",
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc_norm,none",
    ]

    for key in candidates:
        if key in metric_block:
            stderr_key = key.replace(",", "_stderr,")
            metric_name = key.split(",")[0]
            metric_value = float(metric_block[key])
            metric_stderr = (
                float(metric_block[stderr_key]) if stderr_key in metric_block else None
            )
            return metric_name, metric_value, metric_stderr

    available = sorted(metric_block.keys())
    raise KeyError(f"No supported metric key found. Available keys: {available}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                        help="Run ID from the manifest.")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to raw lm-eval JSON output.")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Path to append flat JSONL records.")
    parser.add_argument("--manifest", type=str, default="manifests/full-manifest.csv",
                        help="Path to the experiment manifest.")
    args = parser.parse_args()

    # Read the manifest row so the parsed result keeps run metadata.
    row = get_run_row(args.run_id, args.manifest)

    # Load the raw harness JSON.
    with open(args.input_json, "r") as f:
        raw = json.load(f)

    # For MMLU we use the top-level group score if available.
    # In your current harness output, this is inside raw["groups"]["mmlu"].
    if row["task"] == "mmlu":
        metric_block = raw["groups"]["mmlu"]
    else:
        metric_block = raw["results"][row["task"]]

    metric_name, metric_value, metric_stderr = pick_metric(metric_block)

    record = {
        "run_id": row["run_id"],
        "phase": row["phase"],
        "model": row["model"],
        "method": row["method"],
        "sparsity": int(row["sparsity"]),
        "schedule": row["schedule"],
        "calibration": row["calibration"],
        "task": row["task"],
        "num_fewshot": int(row["num_fewshot"]),
        "seed": int(row["seed"]),

        # Parsed evaluation fields
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_stderr": metric_stderr,
        "sample_len": int(metric_block.get("sample_len", -1)),
        "eval_time_sec": float(raw.get("total_evaluation_time_seconds", -1.0)),

        # Provenance
        "source_json": args.input_json,
        "model_name": raw.get("model_name"),
        "model_dtype": raw.get("config", {}).get("model_dtype"),
        "limit": raw.get("config", {}).get("limit"),
    }

    # Make sure the JSONL directory exists.
    output_path = Path(args.output_jsonl)
    ensure_dir(str(output_path.parent))

    # Append one JSON record per line.
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    print("Parsed record:")
    print(json.dumps(record, indent=2))
    print(f"\nAppended to: {args.output_jsonl}")


if __name__ == "__main__":
    main()