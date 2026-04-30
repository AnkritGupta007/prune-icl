#!/usr/bin/env python3
"""
Curate results from a manifest CSV and a JSONL results file.

Creates TWO output files:

1) curated_best.jsonl
   - one selected best result per manifest run_id

2) duplicate_resolution.json
   - pretty, easy-to-read JSON file
   - includes summary, missing run_ids, and duplicate resolution details

Selection rules:
- Phase 0:
    * no special limit preference
    * choose best by metric_value, then latest timestamp
- Phase 1:
    * if task is BBH zero-shot or few-shot, prefer limit=50
      accepted task spellings:
        - bbh_zeroshot
        - bbh-zeroshot
        - bbh_fewshot
        - bbh-fewshot
        - bbh
    * otherwise prefer limit=200

Tie-breaking:
1. higher metric_value
2. latest timestamp parsed from source_json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TIMESTAMP_RE = re.compile(
    r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)\.json"
)

BBH_TASKS = {
    "bbh_cot_zeroshot",
    "bbh_cot_fewshot",
}


@dataclass
class Candidate:
    raw: Dict[str, Any]
    run_id: str
    metric_value: float
    limit: Optional[float]
    task: str
    timestamp: Optional[datetime]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest CSV")
    parser.add_argument("--results", type=Path, required=True, help="Path to input JSONL results file")
    parser.add_argument("--best-output", type=Path, required=True, help="Path to curated best JSONL")
    parser.add_argument("--dupes-output", type=Path, required=True, help="Path to duplicate-resolution JSON")
    return parser.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_timestamp_from_source_json(source_json: Any) -> Optional[datetime]:
    if not isinstance(source_json, str):
        return None

    match = TIMESTAMP_RE.search(source_json)
    if not match:
        return None

    ts_str = match.group(1)
    for fmt in ("%Y-%m-%dT%H-%M-%S.%f", "%Y-%m-%dT%H-%M-%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            pass
    return None


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    manifest_rows: Dict[str, Dict[str, str]] = {}

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row["run_id"].strip()
            manifest_rows[run_id] = row

    return manifest_rows


def candidate_identity(c: Candidate) -> Tuple[Any, ...]:
    """
    Fields that define whether two duplicate rows are effectively the same.
    If all duplicates share the same identity, we do not emit a duplicate entry.
    """
    return (
        c.metric_value,
        c.limit,
        c.task,
        c.raw.get("source_json"),
        c.raw.get("sample_len"),
        c.raw.get("metric_name"),
    )


def has_meaningful_duplicates(candidates: List[Candidate]) -> bool:
    identities = {candidate_identity(c) for c in candidates}
    return len(identities) > 1

def load_results(results_path: Path) -> Dict[str, List[Candidate]]:
    grouped: Dict[str, List[Candidate]] = defaultdict(list)

    with results_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON on line {line_num}: {e}")
                continue

            run_id = obj.get("run_id")
            if not run_id:
                print(f"[WARN] Skipping line {line_num}: missing run_id")
                continue

            metric_value = parse_float(obj.get("metric_value"))
            if metric_value is None:
                metric_value = float("-inf")

            grouped[run_id].append(
                Candidate(
                    raw=obj,
                    run_id=run_id,
                    metric_value=metric_value,
                    limit=parse_float(obj.get("limit")),
                    task=str(obj.get("task", "")).strip(),
                    timestamp=parse_timestamp_from_source_json(obj.get("source_json")),
                )
            )

    return grouped


def normalize_task(task: str) -> str:
    return task.strip().lower()


def preferred_limit(manifest_row: Dict[str, str]) -> Optional[float]:
    phase = manifest_row.get("phase", "").strip().lower()
    task = normalize_task(manifest_row.get("task", ""))

    if phase == "phase0":
        return None

    if phase == "phase1":
        if task in BBH_TASKS:
            return 50.0
        return 200.0

    return None


def sort_key(candidate: Candidate) -> Tuple[float, float]:
    ts = candidate.timestamp.timestamp() if candidate.timestamp else float("-inf")
    return (candidate.metric_value, ts)


def choose_candidate(
    manifest_row: Dict[str, str],
    candidates: List[Candidate],
) -> Tuple[Candidate, str, List[Candidate]]:
    phase = manifest_row.get("phase", "").strip().lower()
    manifest_task = normalize_task(manifest_row.get("task", ""))

    def finalize(pool: List[Candidate], prefix: str) -> Tuple[Candidate, str, List[Candidate]]:
        ranked = sorted(pool, key=sort_key, reverse=True)
        chosen = ranked[0]

        if len(ranked) == 1:
            return chosen, f"{prefix}; only one candidate", ranked

        top, second = ranked[0], ranked[1]
        if top.metric_value > second.metric_value:
            reason = (
                f"{prefix}; chose higher metric_value "
                f"({top.metric_value:.6f} > {second.metric_value:.6f})"
            )
        else:
            top_ts = top.timestamp.isoformat() if top.timestamp else "unknown"
            second_ts = second.timestamp.isoformat() if second.timestamp else "unknown"
            reason = (
                f"{prefix}; metric tied, chose latest timestamp "
                f"({top_ts} vs {second_ts})"
            )
        return chosen, reason, ranked

    # phase0: no limit preference
    if phase == "phase0":
        return finalize(candidates, "phase0; no limit preference")

    # phase1 BBH
    if phase == "phase1" and manifest_task in BBH_TASKS:
        explicit_bbh = [
            c for c in candidates
            if ("bbh_fewshot" in normalize_task(c.task)) or ("bbh_zeroshot" in normalize_task(c.task))
        ]

        if explicit_bbh:
            explicit_bbh_limit50 = [c for c in explicit_bbh if c.limit == 50.0]
            if explicit_bbh_limit50:
                return finalize(
                    explicit_bbh_limit50,
                    "phase1 BBH; filtered to explicit bbh_fewshot/bbh_zeroshot candidates; preferred limit=50",
                )
            return finalize(
                explicit_bbh,
                "phase1 BBH; filtered to explicit bbh_fewshot/bbh_zeroshot candidates; no limit=50 found",
            )

        # fallback: no explicit task candidates, still keep BBH limit rule
        limit50 = [c for c in candidates if c.limit == 50.0]
        if limit50:
            return finalize(
                limit50,
                "phase1 BBH; no explicit bbh_fewshot/bbh_zeroshot candidates found; fell back to limit=50",
            )
        return finalize(
            candidates,
            "phase1 BBH; no explicit bbh_fewshot/bbh_zeroshot candidates found and no limit=50 found; used all candidates",
        )

    # phase1 non-BBH
    if phase == "phase1":
        limit200 = [c for c in candidates if c.limit == 200.0]
        if limit200:
            return finalize(limit200, "phase1 non-BBH; preferred limit=200")
        return finalize(candidates, "phase1 non-BBH; no limit=200 found; used all candidates")

    # fallback for any other phase
    return finalize(candidates, "default selection")


def candidate_to_summary(c: Candidate) -> Dict[str, Any]:
    return {
        "metric_value": c.metric_value,
        "limit": c.limit,
        "task": c.task,
        "source_json": c.raw.get("source_json"),
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()

    manifest = load_manifest(args.manifest)
    results_by_run = load_results(args.results)

    curated_best_rows: List[Dict[str, Any]] = []
    duplicate_resolution: List[Dict[str, Any]] = []
    missing_run_ids: List[str] = []

    for run_id, manifest_row in manifest.items():
        candidates = results_by_run.get(run_id, [])
        if not candidates:
            missing_run_ids.append(run_id)
            continue

        chosen, reason, ranked_pool = choose_candidate(manifest_row, candidates)

        best_row = dict(chosen.raw)
        best_row["selection_reason"] = reason
        best_row["manifest_phase"] = manifest_row.get("phase")
        best_row["manifest_task"] = manifest_row.get("task")
        best_row["manifest_limit"] = parse_float(manifest_row.get("limit"))
        curated_best_rows.append(best_row)

        if len(candidates) > 1 and has_meaningful_duplicates(candidates):
            duplicate_resolution.append(
                {
                    "run_id": run_id,
                    "num_candidates": len(candidates),
                    "selected_metric_value": chosen.metric_value,
                    "selected_limit": chosen.limit,
                    "selected_source_json": chosen.raw.get("source_json"),
                    "reason": reason,
                    "all_candidates": [
                        candidate_to_summary(c)
                        for c in sorted(candidates, key=sort_key, reverse=True)
                    ],
                }
            )

    curated_best_rows.sort(key=lambda x: x["run_id"])
    duplicate_resolution.sort(key=lambda x: x["run_id"])
    missing_run_ids.sort()

    write_jsonl(args.best_output, curated_best_rows)

    duplicate_payload = {
        "manifest_path": str(args.manifest),
        "results_path": str(args.results),
        "summary": {
            "manifest_run_ids": len(manifest),
            "result_unique_run_ids_seen": len(results_by_run),
            "curated_run_ids_written": len(curated_best_rows),
            "missing_manifest_run_ids": len(missing_run_ids),
            "run_ids_with_duplicates": len(duplicate_resolution),
        },
        "missing_run_ids": missing_run_ids,
        "duplicate_resolution": duplicate_resolution,
    }

    args.dupes_output.parent.mkdir(parents=True, exist_ok=True)
    with args.dupes_output.open("w", encoding="utf-8") as f:
        json.dump(duplicate_payload, f, indent=2)

    print("=" * 80)
    print("CURATION SUMMARY")
    print("=" * 80)
    print(f"Manifest run_ids:            {len(manifest)}")
    print(f"Results unique run_ids:      {len(results_by_run)}")
    print(f"Best rows written:           {len(curated_best_rows)}")
    print(f"Duplicate groups written:    {len(duplicate_resolution)}")
    print(f"Missing manifest run_ids:    {len(missing_run_ids)}")
    print(f"Best output:                 {args.best_output}")
    print(f"Duplicates output:           {args.dupes_output}")

    if missing_run_ids:
        print("\nMISSING RUN IDS:")
        for run_id in missing_run_ids:
            print(f"- {run_id}")


if __name__ == "__main__":
    main()