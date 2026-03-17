"""
Main runner entry point for the pruning + ICL project.

Current purpose:
- read one run from the manifest by run_id
- resolve backend and model config path
- create output directory
- save resolved run configuration as JSON
- print config in dry-run mode

Later this same runner will dispatch actual prune/eval stages.
"""

from __future__ import annotations

import argparse
import json

from src.utils.io import get_run_row, ensure_dir
from src.utils.run_config import row_to_config, config_to_dict
from src.prune.registry import get_backend


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

    # Load the row for the requested run_id.
    row = get_run_row(args.run_id, args.manifest)

    # Convert the raw row into a structured config object.
    cfg = row_to_config(row)

    # Resolve which pruning backend this method should use.
    backend = get_backend(cfg.method)

    # Create the run output directory.
    ensure_dir(cfg.output_dir)

    # Build the JSON payload to save.
    payload = config_to_dict(cfg)
    payload["backend"] = backend

    # Save the resolved run configuration for reproducibility.
    with open(cfg.summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    # In dry-run mode, just print the resolved config and stop.
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    # Placeholder for future execution logic.
    print(f"[TODO] execute run: {cfg.run_id}")
    print(f"phase={cfg.phase} method={cfg.method} task={cfg.task} backend={backend}")
    print(f"model_config_path={cfg.model_config_path}")


if __name__ == "__main__":
    main()
