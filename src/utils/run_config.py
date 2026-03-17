"""
Run configuration utilities.

Purpose:
- convert one manifest CSV row into a structured config object
- attach derived fields such as the model config path
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from src.utils.model_registry import get_model_config_path


@dataclass
class RunConfig:
    # Fields directly coming from the manifest row.
    run_id: str
    phase: str
    model: str
    method: str
    sparsity: int
    schedule: str
    calibration: str
    task: str
    num_fewshot: int
    seed: int
    enabled: int
    notes: str

    # Derived field resolved from the model registry.
    model_config_path: str

    @property
    def output_dir(self) -> str:
        """Directory where artifacts for this run will be stored."""
        return f"artifacts/{self.phase}/{self.run_id}"

    @property
    def summary_json(self) -> str:
        """Path where the resolved run config will be saved as JSON."""
        return f"{self.output_dir}/run_config.json"


def row_to_config(row: dict) -> RunConfig:
    """
    Convert a manifest row dictionary into a RunConfig object.

    Args:
        row: One manifest row as a dictionary.

    Returns:
        A structured RunConfig object with both raw and derived fields.
    """
    model_key = str(row["model"])
    model_config_path = get_model_config_path(model_key)

    return RunConfig(
        run_id=str(row["run_id"]),
        phase=str(row["phase"]),
        model=model_key,
        method=str(row["method"]),
        sparsity=int(row["sparsity"]),
        schedule=str(row["schedule"]),
        calibration=str(row["calibration"]),
        task=str(row["task"]),
        num_fewshot=int(row["num_fewshot"]),
        seed=int(row["seed"]),
        enabled=int(row["enabled"]),
        notes=str(row["notes"]),
        model_config_path=model_config_path,
    )


def config_to_dict(cfg: RunConfig) -> dict:
    """
    Convert a RunConfig dataclass to a plain dictionary.

    Args:
        cfg: RunConfig object.

    Returns:
        Plain dictionary representation of the config.
    """
    return asdict(cfg)