"""
Model registry for the pruning + ICL project.

Purpose:
- map short model keys from the manifest to actual YAML config paths
- keep model selection centralized and explicit
"""

from __future__ import annotations


# Map short manifest model names to YAML config files.
MODEL_CONFIG_PATHS = {
    "llama31_8b": "configs/models/llama31_8b.yaml",
    "mistral7b_v03": "configs/models/mistral7b_v03.yaml",
}


def get_model_config_path(model_key: str) -> str:
    """
    Return the YAML config path for a manifest model key.

    Args:
        model_key: Short model name used in the manifest,
            for example "llama31_8b".

    Returns:
        Path to the corresponding YAML config file.

    Raises:
        ValueError: if the model key is unknown.
    """
    if model_key not in MODEL_CONFIG_PATHS:
        raise ValueError(f"Unknown model key: {model_key}")
    return MODEL_CONFIG_PATHS[model_key]
