"""
Compute k-shot gain from flat evaluation JSONL records.

Definition:
    k-shot gain = k-shot metric_value - zero-shot metric_value

Current behavior:
- reads artifacts/eval_jsonl/results.jsonl
- matches zero-shot and nonzero-shot records for the same run setting
- appends gain records to a JSONL output
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.utils.io import ensure_dir


def make_key(record: dict) -> tuple:
    """
    Build a grouping key that should match zero-shot and k-shot runs
    belonging to the same experimental condition.
    """
    return (
        record["phase"],
        record["model"],
        record["method"],
        int(record["sparsity"]),
        record["schedule"],
        record["calibration"],
        record["task"],
        int(record["seed"]),
        record["metric_name"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Flat evaluation JSONL ledger.")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Where to append k-shot gain records.")
    args = parser.parse_args()

    # Read all flat eval records.
    records = []
    with open(args.input_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Group records by experimental condition.
    grouped = defaultdict(list)
    for rec in records:
        grouped[make_key(rec)].append(rec)

    # Prepare output path.
    output_path = Path(args.output_jsonl)
    ensure_dir(str(output_path.parent))

    gain_records = []

    for key, group in grouped.items():
        zero = None
        nonzero = []

        for rec in group:
            if int(rec["num_fewshot"]) == 0:
                zero = rec
            else:
                nonzero.append(rec)

        if zero is None:
            continue

        for krec in nonzero:
            gain = float(krec["metric_value"]) - float(zero["metric_value"])

            out = {
                "phase": krec["phase"],
                "model": krec["model"],
                "method": krec["method"],
                "sparsity": int(krec["sparsity"]),
                "schedule": krec["schedule"],
                "calibration": krec["calibration"],
                "task": krec["task"],
                "seed": int(krec["seed"]),
                "metric_name": krec["metric_name"],

                "zero_shot_run_id": zero["run_id"],
                "k_shot_run_id": krec["run_id"],
                "k": int(krec["num_fewshot"]),
                "zero_shot_value": float(zero["metric_value"]),
                "k_shot_value": float(krec["metric_value"]),
                "k_shot_gain": gain,
            }

            gain_records.append(out)

    # Append all gain records.
    with open(args.output_jsonl, "a") as f:
        for rec in gain_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Computed {len(gain_records)} gain records.")
    for rec in gain_records:
        print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()