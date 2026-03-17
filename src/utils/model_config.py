"""
Utilities for loading model configuration files.

This module reads YAML files from configs/models/
and returns a plain Python dictionary that other code can use.
"""

from __future__ import annotations

import yaml


def load_model_config(config_path: str) -> dict:
    """
    Load a YAML model config from disk.

    Args:
        config_path: Path to a YAML file, for example:
            configs/models/llama31_8b.yaml

    Returns:
        A dictionary containing model loading settings.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg
