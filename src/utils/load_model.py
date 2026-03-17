"""
Model loading utilities for the pruning + ICL project.

This module centralizes how we load:
- tokenizer
- causal language model

so that all scripts use the same logic.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.model_config import load_model_config


# Map text in YAML to actual torch dtype objects.
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_tokenizer_and_model(config_path: str):
    """
    Load tokenizer and model from a YAML config.

    Args:
        config_path: Path to model YAML config.

    Returns:
        tokenizer, model, cfg
        where cfg is the parsed configuration dictionary.
    """
    cfg = load_model_config(config_path)

    hf_model_name = cfg["hf_model_name"]
    trust_remote_code = bool(cfg.get("trust_remote_code", False))
    dtype_name = cfg.get("torch_dtype", "float16")
    device_map = cfg.get("device_map", "auto")

    # Convert YAML string dtype to torch dtype object.
    torch_dtype = DTYPE_MAP[dtype_name]

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        trust_remote_code=trust_remote_code,
    )

    # Some base models do not define a pad token.
    # For simple causal LM evaluation/generation, using eos as pad is common.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model.
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    # Put model in eval mode since we are only doing inference here.
    model.eval()

    return tokenizer, model, cfg
