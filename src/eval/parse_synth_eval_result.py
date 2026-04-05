"""
Parse synthetic linear ICL raw JSON output into one flat JSONL record.

What this file does:
- reads the raw JSON produced by src.eval.run_synth_icl
- infers run metadata from the manifest via run_id
- appends one flat record to the project JSONL ledger

Why we need this:
- synthetic_linear_icl is not produced by lm-eval-harness
- so it needs its own parser into the shared results format
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.io import get_run_row
from src.utils.run_config import row_to_config


def append_jsonl(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    """
    CLI entry point for parsing one synthetic ICL result JSON into JSONL.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                        help="Run identifier from the manifest.")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to raw synthetic eval JSON.")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Path to JSONL ledger to append to.")
    parser.add_argument("--manifest", type=str, default="manifests/full-manifest.csv",
                        help="Manifest CSV path used to recover run metadata.")
    args = parser.parse_args()

    # Load manifest row so we can recover the canonical run metadata.
    row = get_run_row(args.run_id, args.manifest)
    cfg = row_to_config(row)

    # Load raw synthetic eval JSON.
    with open(args.input_json, "r") as f:
        payload = json.load(f)

    # Build one flat record matching the project JSONL style.
    record = {
        "run_id": cfg.run_id,
        "phase": cfg.phase,
        "model": cfg.model,
        "method": cfg.method,
        "sparsity": cfg.sparsity,
        "schedule": cfg.schedule,
        "calibration": cfg.calibration,
        "task": cfg.task,
        "num_fewshot": cfg.num_fewshot,
        "seed": cfg.seed,
        "metric_name": payload.get("metric_name", "acc"),
        "metric_value": payload.get("metric_value"),
        "metric_stderr": payload.get("metric_stderr"),
        "sample_len": payload.get("sample_len"),
        "eval_time_sec": payload.get("eval_time_sec"),
        "source_json": args.input_json,
        "model_name": payload.get("model_name"),
        "model_dtype": payload.get("model_dtype"),
        "limit": payload.get("limit"),
    }

    append_jsonl(args.output_jsonl, record)

    print("Parsed synthetic record:")
    print(json.dumps(record, indent=2))
    print(f"\nAppended to: {args.output_jsonl}")


if __name__ == "__main__":
    main()