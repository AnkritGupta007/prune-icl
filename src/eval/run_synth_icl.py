#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.model_config import load_model_config


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class SynthTaskConfig:
    dim: int = 4
    low: int = -10
    high: int = 10
    query_label_balance: bool = True
    max_sampling_attempts: int = 100000
    store_max_examples: int = 50


@dataclass
class ModelRuntimeConfig:
    model_source: str
    trust_remote_code: bool = False
    torch_dtype: torch.dtype | None = None
    device_map: str | dict[str, Any] | None = "auto"


def build_logger(verbose: bool) -> logging.Logger:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("run_synth_icl")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    key = str(dtype_name).strip().lower()
    if key not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported torch_dtype={dtype_name!r}. "
            f"Supported: {sorted(DTYPE_MAP.keys())}"
        )
    return DTYPE_MAP[key]


def read_model_runtime_config(
    config_path: str,
    checkpoint_path: str | None,
) -> tuple[dict[str, Any], ModelRuntimeConfig, SynthTaskConfig]:
    cfg = load_model_config(config_path)

    if "hf_model_name" not in cfg and checkpoint_path is None:
        raise ValueError(
            "Config must contain 'hf_model_name' unless --checkpoint_path is provided."
        )

    model_source = checkpoint_path or cfg["hf_model_name"]
    runtime_cfg = ModelRuntimeConfig(
        model_source=model_source,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        torch_dtype=parse_dtype(cfg.get("torch_dtype", None)),
        device_map=cfg.get("device_map", "auto"),
    )

    synth_cfg = SynthTaskConfig(
        dim=int(cfg.get("synth_dim", 4)),
        low=int(cfg.get("synth_low", -10)),
        high=int(cfg.get("synth_high", 10)),
        query_label_balance=bool(cfg.get("synth_query_label_balance", True)),
        max_sampling_attempts=int(cfg.get("synth_max_sampling_attempts", 100000)),
        store_max_examples=int(cfg.get("synth_store_max_examples", 50)),
    )

    if synth_cfg.dim <= 0:
        raise ValueError("synth_dim must be positive.")
    if synth_cfg.low > synth_cfg.high:
        raise ValueError("synth_low must be <= synth_high.")
    if synth_cfg.max_sampling_attempts <= 0:
        raise ValueError("synth_max_sampling_attempts must be positive.")
    if synth_cfg.store_max_examples < 0:
        raise ValueError("synth_store_max_examples must be >= 0.")

    return cfg, runtime_cfg, synth_cfg


def load_tokenizer_and_model(runtime_cfg: ModelRuntimeConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        runtime_cfg.model_source,
        trust_remote_code=runtime_cfg.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has no pad_token and no eos_token. Cannot set pad_token safely."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        runtime_cfg.model_source,
        torch_dtype=runtime_cfg.torch_dtype,
        device_map=runtime_cfg.device_map,
        trust_remote_code=runtime_cfg.trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def sample_int_vector(dim: int, low: int, high: int) -> list[int]:
    return [random.randint(low, high) for _ in range(dim)]


def sample_nonzero_separator(dim: int, low: int, high: int) -> list[int]:
    while True:
        w = sample_int_vector(dim=dim, low=low, high=high)
        if any(v != 0 for v in w):
            return w


def linear_score(x: list[int], w: list[int]) -> int:
    return sum(a * b for a, b in zip(x, w))


def classify_binary(x: list[int], w: list[int]) -> int:
    """
    Binary label handling is 0/1 throughout.
    Label rule:
      y = 1 if dot(x, w) >= 0
      y = 0 otherwise
    """
    return 1 if linear_score(x, w) >= 0 else 0


def format_example(x: list[int], y: int) -> str:
    return f"{x} = {y}"


def format_query(x: list[int]) -> str:
    return f"{x} ="


def build_prompt(support: list[tuple[list[int], int]], query_x: list[int]) -> str:
    lines = [format_example(x, y) for x, y in support]
    lines.append(format_query(query_x))
    return "\n".join(lines)


def vector_key(x: list[int]) -> tuple[int, ...]:
    return tuple(x)


def sample_unique_example_for_label(
    w: list[int],
    target_label: int,
    used_vectors: set[tuple[int, ...]],
    dim: int,
    low: int,
    high: int,
    max_attempts: int,
) -> tuple[list[int], int]:
    for _ in range(max_attempts):
        x = sample_int_vector(dim=dim, low=low, high=high)
        key = vector_key(x)
        if key in used_vectors:
            continue
        y = classify_binary(x, w)
        if y == target_label:
            used_vectors.add(key)
            return x, y
    raise RuntimeError(
        f"Failed to sample a unique example for label={target_label} "
        f"within {max_attempts} attempts."
    )


def sample_balanced_support(
    w: list[int],
    n: int,
    dim: int,
    low: int,
    high: int,
    max_attempts: int,
) -> list[tuple[list[int], int]]:
    if n < 0:
        raise ValueError("num_fewshot must be >= 0")

    target_0 = n // 2
    target_1 = n - target_0

    used_vectors: set[tuple[int, ...]] = set()
    support: list[tuple[list[int], int]] = []

    for _ in range(target_0):
        support.append(
            sample_unique_example_for_label(
                w=w,
                target_label=0,
                used_vectors=used_vectors,
                dim=dim,
                low=low,
                high=high,
                max_attempts=max_attempts,
            )
        )

    for _ in range(target_1):
        support.append(
            sample_unique_example_for_label(
                w=w,
                target_label=1,
                used_vectors=used_vectors,
                dim=dim,
                low=low,
                high=high,
                max_attempts=max_attempts,
            )
        )

    random.shuffle(support)
    return support


def sample_query(
    w: list[int],
    support: list[tuple[list[int], int]],
    dim: int,
    low: int,
    high: int,
    max_attempts: int,
    force_balanced_query_label: bool,
    example_index: int,
) -> tuple[list[int], int]:
    used = {vector_key(x) for x, _ in support}

    target_label = None
    if force_balanced_query_label:
        target_label = example_index % 2

    for _ in range(max_attempts):
        x = sample_int_vector(dim=dim, low=low, high=high)
        if vector_key(x) in used:
            continue
        y = classify_binary(x, w)
        if target_label is None or y == target_label:
            return x, y

    raise RuntimeError(
        f"Failed to sample query example within {max_attempts} attempts."
    )


def continuation_text_for_label(label: int) -> str:
    if label not in (0, 1):
        raise ValueError(f"Unsupported label: {label}")
    return f" {label}"


@torch.no_grad()
def score_continuation(
    tokenizer,
    model,
    prompt: str,
    continuation: str,
) -> dict[str, Any]:
    """
    Compute conditional continuation score:
    mean log p(continuation tokens | prompt + previous continuation tokens)

    Returns detailed token-level debug info.
    """
    device = next(model.parameters()).device

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(
        prompt + continuation,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    if full_ids.shape[1] <= prompt_ids.shape[1]:
        raise RuntimeError("Continuation tokenization produced no continuation tokens.")

    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :]
    target_ids = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)

    prompt_len = prompt_ids.shape[1]
    continuation_start = prompt_len - 1

    continuation_log_probs = log_probs[:, continuation_start:, :]
    continuation_target_ids = target_ids[:, continuation_start:]

    gathered = continuation_log_probs.gather(
        dim=-1,
        index=continuation_target_ids.unsqueeze(-1),
    ).squeeze(-1)

    token_ids = continuation_target_ids[0].tolist()
    token_logprobs = gathered[0].tolist()
    mean_logprob = float(sum(token_logprobs) / len(token_logprobs))

    token_texts = [
        tokenizer.decode([tok_id], skip_special_tokens=False)
        for tok_id in token_ids
    ]

    return {
        "continuation": continuation,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "token_logprobs": token_logprobs,
        "sum_logprob": float(sum(token_logprobs)),
        "mean_logprob": mean_logprob,
        "num_tokens": len(token_ids),
    }


@torch.no_grad()
def predict_binary_label(
    tokenizer,
    model,
    prompt: str,
) -> tuple[int, dict[str, Any]]:
    candidate_payloads: dict[int, dict[str, Any]] = {}

    for label in (0, 1):
        continuation = continuation_text_for_label(label)
        candidate_payloads[label] = score_continuation(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            continuation=continuation,
        )

    pred = max(
        candidate_payloads.items(),
        key=lambda kv: kv[1]["mean_logprob"],
    )[0]

    return pred, {
        "predicted_label": pred,
        "candidate_scores": {
            str(label): payload["mean_logprob"]
            for label, payload in candidate_payloads.items()
        },
        "candidate_debug": {
            str(label): payload
            for label, payload in candidate_payloads.items()
        },
    }


def compute_binomial_stderr(acc: float, n: int) -> float | None:
    if n <= 0:
        return None
    return math.sqrt(acc * (1.0 - acc) / n)


def run_eval(
    tokenizer,
    model,
    num_fewshot: int,
    limit: int,
    seed: int,
    synth_cfg: SynthTaskConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    if num_fewshot < 0:
        raise ValueError("--num_fewshot must be >= 0")
    if limit <= 0:
        raise ValueError("--limit must be > 0")

    set_seed(seed)

    start_time = time.time()
    total = 0
    correct = 0
    per_example_records: list[dict[str, Any]] = []

    logger.info("Starting synthetic linear ICL evaluation")
    logger.info("limit=%d num_fewshot=%d dim=%d", limit, num_fewshot, synth_cfg.dim)

    for idx in range(limit):
        w = sample_nonzero_separator(
            dim=synth_cfg.dim,
            low=synth_cfg.low,
            high=synth_cfg.high,
        )

        support = sample_balanced_support(
            w=w,
            n=num_fewshot,
            dim=synth_cfg.dim,
            low=synth_cfg.low,
            high=synth_cfg.high,
            max_attempts=synth_cfg.max_sampling_attempts,
        )

        query_x, gold_y = sample_query(
            w=w,
            support=support,
            dim=synth_cfg.dim,
            low=synth_cfg.low,
            high=synth_cfg.high,
            max_attempts=synth_cfg.max_sampling_attempts,
            force_balanced_query_label=synth_cfg.query_label_balance,
            example_index=idx,
        )

        prompt = build_prompt(support=support, query_x=query_x)
        pred_y, debug_payload = predict_binary_label(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
        )

        is_correct = int(pred_y == gold_y)
        total += 1
        correct += is_correct

        if idx < synth_cfg.store_max_examples:
            per_example_records.append(
                {
                    "index": idx,
                    "separator_w": w,
                    "support": [
                        {"x": x, "y": y}
                        for x, y in support
                    ],
                    "query": {"x": query_x, "y": gold_y},
                    "prompt": prompt,
                    "gold": gold_y,
                    "pred": pred_y,
                    "correct": bool(is_correct),
                    "debug": debug_payload,
                }
            )

        logger.info(
            "[%d/%d] pred=%d gold=%d correct=%s",
            idx + 1,
            limit,
            pred_y,
            gold_y,
            bool(is_correct),
        )

    elapsed = time.time() - start_time
    acc = correct / total
    stderr = compute_binomial_stderr(acc, total)

    return {
        "task": "synthetic_linear_icl",
        "metric_name": "acc",
        "metric_value": acc,
        "metric_stderr": stderr,
        "sample_len": total,
        "num_correct": correct,
        "num_incorrect": total - correct,
        "eval_time_sec": elapsed,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "seed": seed,
        "synth_config": asdict(synth_cfg),
        "examples": per_example_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic linear classification ICL evaluator for causal LMs."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model YAML config.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional local checkpoint directory override for pruned models.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        required=True,
        help="Number of support examples in the prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of synthetic tasks to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save raw JSON results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = build_logger(args.verbose)

    cfg, runtime_cfg, synth_cfg = read_model_runtime_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
    )

    logger.info("Loading model from: %s", runtime_cfg.model_source)
    tokenizer, model = load_tokenizer_and_model(runtime_cfg)

    result = run_eval(
        tokenizer=tokenizer,
        model=model,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        seed=args.seed,
        synth_cfg=synth_cfg,
        logger=logger,
    )

    result["model_name"] = runtime_cfg.model_source
    result["model_dtype"] = str(next(model.parameters()).dtype)
    result["config_path"] = args.config
    result["checkpoint_path"] = args.checkpoint_path
    result["resolved_config"] = cfg

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    summary = {
        "task": result["task"],
        "metric_name": result["metric_name"],
        "metric_value": result["metric_value"],
        "metric_stderr": result["metric_stderr"],
        "sample_len": result["sample_len"],
        "eval_time_sec": result["eval_time_sec"],
        "model_name": result["model_name"],
        "model_dtype": result["model_dtype"],
        "output_json": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise