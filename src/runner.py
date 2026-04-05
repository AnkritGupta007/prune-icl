"""
Main runner entry point for the pruning + ICL project.

Current supported execution:
- dense MMLU evaluation runs

Current flow:
- resolve run from manifest
- resolve backend and model config
- save run config JSON
- execute lm-eval for supported runs
- parse raw JSON into flat JSONL results

Later:
- pruning execution
- more tasks
- k-shot gain automation hooks
"""

from __future__ import annotations

import sys
import argparse
import json
import subprocess
from pathlib import Path

from src.utils.io import get_run_row, ensure_dir
from src.utils.run_config import row_to_config, config_to_dict
from src.prune.registry import get_backend
from src.utils.checkpoint_resolver import resolve_pruned_checkpoint


RESULTS_JSONL = "artifacts/eval_jsonl/results.jsonl"

def find_latest_lm_eval_json(run_dir: str) -> str:
    """
    Find the newest plausible lm-eval JSON result under a run directory.
    Excludes run_config.json.
    """
    candidates = [
        p for p in Path(run_dir).rglob("*.json")
        if p.name != "run_config.json"
    ]

    if not candidates:
        raise FileNotFoundError(f"No JSON files found under {run_dir}")

    preferred = [p for p in candidates if "result" in p.name.lower()]
    chosen_pool = preferred if preferred else candidates
    chosen_pool = sorted(chosen_pool, key=lambda p: p.stat().st_mtime)

    return str(chosen_pool[-1])

def run_dense_eval(cfg) -> None:
    """
    Execute a dense evaluation run for supported tasks.

    Currently supported:
    - task = mmlu
    """
    ensure_dir(cfg.output_dir)

    # Step 1: run raw lm-eval and save run-specific raw JSON.
    eval_cmd = [
        sys.executable, "-m", "src.eval.run_lm_eval",
        "--config", cfg.model_config_path,
        "--task", cfg.task,
        "--num_fewshot", str(cfg.num_fewshot),
        "--output_json", cfg.output_dir,
    ]

    if getattr(cfg, "limit", None) is not None:
        eval_cmd.extend(["--limit", str(cfg.limit)])

    print("Running eval command:")
    print(" ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True)

    actual_json = find_latest_lm_eval_json(cfg.output_dir)
    print(f"\nResolved raw eval JSON: {actual_json}")

    # Step 2: parse the raw JSON into the flat JSONL ledger.
    parse_cmd = [
        sys.executable, "-m", "src.eval.parse_lm_eval_result",
        "--run_id", cfg.run_id,
        "--input_json", actual_json,
        "--output_jsonl", RESULTS_JSONL,
    ]

    print("\nRunning parse command:")
    print(" ".join(parse_cmd))
    subprocess.run(parse_cmd, check=True)

    
def run_dense_synth_icl(cfg) -> None:
    """
    Run the synthetic linear ICL evaluation and parse the result.

    What this function does:
    - calls the custom paper-aligned synthetic ICL evaluator
    - writes a raw JSON result into the run artifact directory
    - parses that raw JSON into the flat project JSONL ledger

    Why we need this:
    - lm-eval-harness is for benchmark tasks like MMLU
    - synthetic linear ICL is a custom paper-aligned task, so it needs
      its own evaluator and parser
    """
    ensure_dir(cfg.output_dir)

    raw_json_path = cfg.output_dir + "/synth_eval_result.json"

    # Step 1: run the synthetic ICL evaluator and save a raw JSON file.
    eval_cmd = [
        sys.executable, "-m", "src.eval.run_synth_icl",
        "--config", cfg.model_config_path,
        "--num_fewshot", str(cfg.num_fewshot),
        "--seed", str(cfg.seed),
        "--output_json", raw_json_path,
    ]

    if getattr(cfg, "limit", None) is not None:
        eval_cmd.extend(["--limit", str(cfg.limit)])

    print("Running synthetic ICL eval command:")
    print(" ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True)

    # Step 2: parse the raw JSON into the flat JSONL results ledger.
    parse_cmd = [
        sys.executable, "-m", "src.eval.parse_synth_eval_result",
        "--run_id", cfg.run_id,
        "--input_json", raw_json_path,
        "--output_jsonl", RESULTS_JSONL,
    ]

    print("\nRunning synth parse command:")
    print(" ".join(parse_cmd))
    subprocess.run(parse_cmd, check=True)

    

def run_pruned_eval(cfg) -> None:
    ensure_dir(cfg.output_dir)

    eval_cmd = [
        sys.executable, "-m", "src.eval.run_lm_eval",
        "--config", cfg.model_config_path,
        "--checkpoint_path", cfg.checkpoint_path,
        "--task", cfg.task,
        "--num_fewshot", str(cfg.num_fewshot),
        "--output_json", cfg.output_dir,
    ]

    if getattr(cfg, "limit", None) is not None:
        eval_cmd.extend(["--limit", str(cfg.limit)])

    print("Running pruned eval command:")
    print(" ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True)

    actual_json = find_latest_lm_eval_json(cfg.output_dir)
    print(f"\nResolved raw eval JSON: {actual_json}")

    parse_cmd = [
        sys.executable, "-m", "src.eval.parse_lm_eval_result",
        "--run_id", cfg.run_id,
        "--input_json", actual_json,
        "--output_jsonl", RESULTS_JSONL,
    ]

    print("\nRunning parse command:")
    print(" ".join(parse_cmd))
    subprocess.run(parse_cmd, check=True)


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                        help="Unique run identifier from the manifest CSV.")
    parser.add_argument("--manifest", type=str, default="manifests/phase1_minimal.csv",
                        help="Path to the manifest CSV file.")
    parser.add_argument("--dry_run", action="store_true",
                        help="If set, only print resolved config without running anything.")
    args = parser.parse_args()

    # Load the requested manifest row.
    row = get_run_row(args.run_id, args.manifest)

    # Convert it into structured config.
    cfg = row_to_config(row)

    if cfg.method != "dense":
        cfg.checkpoint_path = resolve_pruned_checkpoint(
            model=cfg.model,
            method=cfg.method,
            sparsity=cfg.sparsity,
        )

    # Resolve the pruning backend.
    backend = get_backend(cfg.method)

    # Create artifact directory.
    ensure_dir(cfg.output_dir)

    # Save resolved config JSON.
    payload = config_to_dict(cfg)
    payload["backend"] = backend

    # Save the resolved run configuration for reproducibility.
    with open(cfg.summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    # In dry-run mode, just print the resolved config and stop.
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    # Dense eval path.
    LM_EVAL_TASKS = {"mmlu", "bbh", "gsm8k", "capability_mini"}

    if cfg.method == "dense":
        if cfg.task in LM_EVAL_TASKS:
            run_dense_eval(cfg)
            return
        if cfg.task == "synthetic_linear_icl":
            run_dense_synth_icl(cfg)
            return

    if cfg.method in {"wanda", "sparsegpt", "ria", "wandapp", "magnitude"}:
        if cfg.task in LM_EVAL_TASKS:
            run_pruned_eval(cfg)
            return
        if cfg.task == "synthetic_linear_icl":
            run_pruned_synth_icl(cfg)
            return

    # All non-dense methods remain future work for now.
    raise NotImplementedError(
        f"Execution not yet implemented for method={cfg.method}, task={cfg.task}"
    )



if __name__ == "__main__":
    main()