"""
Parse a raw lm-evaluation-harness JSON result into a flat experiment record.

Purpose:
- turn large harness JSON outputs into compact, project-friendly records
- append those records to a JSONL ledger for later CSV aggregation

Current support:
- MMLU group-level score extraction
- task-aware metric selection, including GSM8K flexible extraction preference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, get_run_row


def pick_metric(task_name: str, metric_block: dict[str, Any]) -> tuple[str, str, float, float | None]:
    """
    Pick the main scalar metric from an lm-eval result block.

    Task-aware preference:
    - GSM8K prefers flexible extraction because strict string matching can
      undercount numerically correct answers with formatting differences.
    - Most other tasks prefer stricter/exact metrics first.

    Returns:
        (metric_name, metric_key, metric_value, metric_stderr)
    """
    if task_name == "gsm8k":
        candidates = [
            "exact_match,flexible-extract",
            "exact_match,get-answer",
            "exact_match,strict-match",
            "exact_match,none",
            "acc,none",
            "acc_norm,none",
        ]
    else:
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
            return metric_name, key, metric_value, metric_stderr

    available = sorted(metric_block.keys())
    raise KeyError(
        f"No supported metric key found for task '{task_name}'. "
        f"Available keys: {available}"
    )


def resolve_metric_block(row: dict[str, str], raw: dict[str, Any]) -> dict[str, Any]:
    """
    Select the appropriate metric block from the raw lm-eval output.
    """
    task_name = row["task"]

    # For MMLU we prefer the top-level group score if available.
    if task_name == "mmlu":
        groups = raw.get("groups", {})
        if "mmlu" in groups:
            return groups["mmlu"]

    results = raw.get("results", {})
    if task_name in results:
        return results[task_name]

    available_results = sorted(results.keys())
    available_groups = sorted(raw.get("groups", {}).keys())
    raise KeyError(
        f"Could not find metric block for task '{task_name}'. "
        f"Available results keys: {available_results}. "
        f"Available group keys: {available_groups}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID from the manifest.",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to raw lm-eval JSON output.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to append flat JSONL records.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/full-manifest.csv",
        help="Path to the experiment manifest.",
    )
    args = parser.parse_args()

    # Read the manifest row so the parsed result keeps run metadata.
    row = get_run_row(args.run_id, args.manifest)

    # Load the raw harness JSON.
    with open(args.input_json, "r") as f:
        raw = json.load(f)

    metric_block = resolve_metric_block(row, raw)
    metric_name, metric_key, metric_value, metric_stderr = pick_metric(
        row["task"], metric_block
    )

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
        "metric_key": metric_key,
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