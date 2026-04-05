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

LM_EVAL_TASKS = {
    "mmlu",
    "mmlu_abstract_algebra",
    "bbh",
    "gsm8k",
    "capability_mini",
}

PRUNED_METHODS = {
    "ria",
    "wanda",
    "wandapp",
    "magnitude",
    "sparsegpt",
}


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


def build_lm_eval_cmd(cfg, checkpoint_path: str | None = None) -> list[str]:
    """
    Build the shared wrapper command for lm-eval tasks.

    If checkpoint_path is provided, evaluate that local pruned checkpoint.
    Otherwise use the dense base model from config.
    """
    cmd = [
        sys.executable, "-m", "src.eval.run_lm_eval",
        "--config", cfg.model_config_path,
        "--task", cfg.task,
        "--num_fewshot", str(cfg.num_fewshot),
        "--output_json", cfg.output_dir,
    ]

    if checkpoint_path is not None:
        cmd.extend(["--checkpoint_path", checkpoint_path])

    if getattr(cfg, "limit", None) is not None:
        cmd.extend(["--limit", str(cfg.limit)])

    return cmd


def run_dense_eval(cfg) -> None:
    """
    Execute a dense lm-eval benchmark run.
    """
    ensure_dir(cfg.output_dir)

    eval_cmd = build_lm_eval_cmd(cfg)

    print("Running dense eval command:")
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


def run_pruned_eval(cfg) -> None:
    """
    Execute a pruned lm-eval benchmark run using a resolved local checkpoint.
    """
    ensure_dir(cfg.output_dir)

    if not getattr(cfg, "checkpoint_path", None):
        raise ValueError(f"Missing checkpoint_path for pruned run: {cfg.run_id}")

    eval_cmd = build_lm_eval_cmd(cfg, checkpoint_path=cfg.checkpoint_path)

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


def run_dense_synth_icl(cfg) -> None:
    """
    Run the custom synthetic linear ICL evaluation for the dense model.
    """
    ensure_dir(cfg.output_dir)

    raw_json_path = cfg.output_dir + "/synth_eval_result.json"

    eval_cmd = [
        sys.executable, "-m", "src.eval.run_synth_icl",
        "--config", cfg.model_config_path,
        "--num_fewshot", str(cfg.num_fewshot),
        "--seed", str(cfg.seed),
        "--output_json", raw_json_path,
    ]

    if getattr(cfg, "limit", None) is not None:
        eval_cmd.extend(["--limit", str(int(cfg.limit))])

    print("Running dense synthetic ICL eval command:")
    print(" ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True)

    parse_cmd = [
        sys.executable, "-m", "src.eval.parse_synth_eval_result",
        "--run_id", cfg.run_id,
        "--input_json", raw_json_path,
        "--output_jsonl", RESULTS_JSONL,
    ]

    print("\nRunning synth parse command:")
    print(" ".join(parse_cmd))
    subprocess.run(parse_cmd, check=True)


def run_pruned_synth_icl(cfg) -> None:
    """
    Run the custom synthetic linear ICL evaluation for a pruned checkpoint.
    """
    ensure_dir(cfg.output_dir)

    if not getattr(cfg, "checkpoint_path", None):
        raise ValueError(f"Missing checkpoint_path for pruned synthetic run: {cfg.run_id}")

    raw_json_path = cfg.output_dir + "/synth_eval_result.json"

    eval_cmd = [
        sys.executable, "-m", "src.eval.run_synth_icl",
        "--config", cfg.model_config_path,
        "--checkpoint_path", cfg.checkpoint_path,
        "--num_fewshot", str(cfg.num_fewshot),
        "--seed", str(cfg.seed),
        "--output_json", raw_json_path,
    ]

    if getattr(cfg, "limit", None) is not None:
        eval_cmd.extend(["--limit", str(int(cfg.limit))])

    print("Running pruned synthetic ICL eval command:")
    print(" ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True)

    parse_cmd = [
        sys.executable, "-m", "src.eval.parse_synth_eval_result",
        "--run_id", cfg.run_id,
        "--input_json", raw_json_path,
        "--output_jsonl", RESULTS_JSONL,
    ]

    print("\nRunning synth parse command:")
    print(" ".join(parse_cmd))
    subprocess.run(parse_cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                        help="Unique run identifier from the manifest CSV.")
    parser.add_argument("--manifest", type=str, default="manifests/full-manifest.csv",
                        help="Path to the manifest CSV file.")
    parser.add_argument("--dry_run", action="store_true",
                        help="If set, only print resolved config without running anything.")
    args = parser.parse_args()

    row = get_run_row(args.run_id, args.manifest)
    cfg = row_to_config(row)

    # Honor enabled flag from manifest.
    if int(cfg.enabled) != 1:
        print(f"Run is disabled in manifest: {cfg.run_id}")
        return

    backend = get_backend(cfg.method)

    # Resolve pruned checkpoint automatically.
    if cfg.method in PRUNED_METHODS:
        cfg.checkpoint_path = resolve_pruned_checkpoint(
            model=cfg.model,
            method=cfg.method,
            sparsity=cfg.sparsity,
        )

    ensure_dir(cfg.output_dir)

    payload = config_to_dict(cfg)
    payload["backend"] = backend

    with open(cfg.summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    # Dense execution
    if cfg.method == "dense":
        if cfg.task in LM_EVAL_TASKS:
            run_dense_eval(cfg)
            print(f"\nCompleted dense eval run: {cfg.run_id}")
            return

        if cfg.task == "synthetic_linear_icl":
            run_dense_synth_icl(cfg)
            print(f"\nCompleted dense synthetic eval run: {cfg.run_id}")
            return

        raise NotImplementedError(f"Dense execution not implemented for task: {cfg.task}")

    # Pruned execution
    if cfg.method in PRUNED_METHODS:
        if cfg.task in LM_EVAL_TASKS:
            run_pruned_eval(cfg)
            print(f"\nCompleted pruned eval run: {cfg.run_id}")
            return

        if cfg.task == "synthetic_linear_icl":
            run_pruned_synth_icl(cfg)
            print(f"\nCompleted pruned synthetic eval run: {cfg.run_id}")
            return

        raise NotImplementedError(
            f"Pruned execution not implemented for method={cfg.method}, task={cfg.task}"
        )

    raise NotImplementedError(
        f"Execution not yet implemented for method={cfg.method}, task={cfg.task}"
    )


if __name__ == "__main__":
    main()