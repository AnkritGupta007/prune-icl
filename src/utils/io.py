from __future__ import annotations
import os
import pandas as pd


MANIFEST_PATH = "manifests/phase1_minimal.csv"


def load_manifest(path: str = MANIFEST_PATH) -> pd.DataFrame:
    """
    Load a manifest CSV file into a pandas DataFrame.

    Args:
        path: Path to the manifest CSV file. Defaults to MANIFEST_PATH.

    Returns:
        A pandas DataFrame containing the manifest rows.
    """
    df = pd.read_csv(path)
    return df


def get_run_row(run_id: str, path: str = MANIFEST_PATH) -> dict:
    """
    Retrieve a single manifest row by run_id.

    Args:
        run_id: Unique run identifier from the manifest.
        path: Path to the manifest CSV file. Defaults to MANIFEST_PATH.

    Returns:
        A dictionary containing the manifest row values.

    Raises:
        ValueError: If the run_id is not found or has duplicate entries.
    """
    df = load_manifest(path)
    rows = df[df["run_id"] == run_id]
    if len(rows) == 0:
        raise ValueError(f"Run ID not found: {run_id}")
    if len(rows) > 1:
        raise ValueError(f"Duplicate Run ID found: {run_id}")
    return rows.iloc[0].to_dict()


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory path exists.

    Args:
        path: Path to the directory to be created.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
