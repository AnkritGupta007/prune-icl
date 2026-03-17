"""
Dense model smoke test.

Purpose:
- load the base model from config
- run one short generation
- print useful debug information

This is not a benchmark. It is only a quick environment check.
"""

from __future__ import annotations

import argparse
import torch

from src.utils.load_model import load_tokenizer_and_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/llama31_8b.yaml",
        help="Path to model YAML config.",
    )
    args = parser.parse_args()

    print("Loading model config from:", args.config)

    # Load tokenizer + model from shared utility.
    tokenizer, model, cfg = load_tokenizer_and_model(args.config)

    # Report basic info.
    print("Loaded hf_model_name:", cfg["hf_model_name"])
    print("Requested dtype:", cfg["torch_dtype"])
    print("Model class:", model.__class__.__name__)

    # Use a very small prompt for a quick test.
    prompt = "Q: What is 2 + 2?\nA:"

    # Tokenize and move inputs to the model device.
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Prompt token count:", inputs["input_ids"].shape[1])
    print("Model device:", model.device)

    # Disable gradients because this is inference only.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(cfg.get("max_new_tokens", 32)),
            do_sample=bool(cfg.get("do_sample", False)),
            temperature=float(cfg.get("temperature", 1.0)),
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the full output.
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== Generated text ===")
    print(decoded)
    print("======================\n")


if __name__ == "__main__":
    main()
