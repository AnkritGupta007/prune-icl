from __future__ import annotations
import os
import pandas as pd


MANIFEST_PATH = "manifests/phase1_minimal.csv"


def load_manifest(path: str = MANIFEST_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_run_row(run_id: str, path: str = MANIFEST_PATH) -> dict:
    df = load_manifest(path)
    rows = df[df["run_id"] == run_id]
    if len(rows) == 0:
        raise ValueError(f"Run ID not found: {run_id}")
    if len(rows) > 1:
        raise ValueError(f"Duplicate Run ID found: {run_id}")
    return rows.iloc[0].to_dict()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
