"""
Synthetic linear in-context learning (ICL) evaluator for LLM pruning experiments.

Each evaluation instance draws a random linear separator w, builds a balanced
few-shot prompt of labeled integer vectors, then asks the model to classify a
held-out query. Labels are {-1, +1}. Scoring is done via candidate log-prob
comparison rather than free generation, which is more reliable for binary tasks.

Usage:
    python eval_synthetic_linear_icl.py \
        --config configs/my_model.yaml \
        --num_fewshot 16 \
        --limit 100 \
        --output_json results/run.json \
        [--checkpoint_path /path/to/pruned_checkpoint] \
        [--dim 4] \
        [--low -10] \
        [--high 10] \
        [--seed 13] \
        [--verbose]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.utils.model_config import load_model_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

LABEL_CANDIDATES: dict[int, str] = {
    -1: "-1",
    1: "1",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tokenizer_and_model(
    config_path: str,
    checkpoint_path: str | None = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, dict[str, Any], str]:
    """
    Load tokenizer and model from a YAML config.

    Args:
        config_path:      Path to the model config YAML.
        checkpoint_path:  Optional local directory override (e.g. a pruned
                          checkpoint). When supplied it replaces the HF hub
                          model name from the config.

    Returns:
        (tokenizer, model, cfg_dict, model_source_string)
    """
    cfg = load_model_config(config_path)

    model_source: str = checkpoint_path if checkpoint_path else cfg["hf_model_name"]
    trust_remote_code: bool = bool(cfg.get("trust_remote_code", False))
    torch_dtype: torch.dtype = DTYPE_MAP[cfg.get("torch_dtype", "float16")]
    device_map: str = cfg.get("device_map", "auto")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    return tokenizer, model, cfg, model_source


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def sample_int_vector(dim: int, low: int, high: int) -> list[int]:
    """Sample a uniform random integer vector."""
    return [random.randint(low, high) for _ in range(dim)]


def make_linear_separator(dim: int, low: int, high: int) -> list[int]:
    """
    Sample a non-zero weight vector that defines the linear decision boundary.
    Resamples until at least one weight is non-zero.
    """
    while True:
        w = sample_int_vector(dim, low, high)
        if any(v != 0 for v in w):
            return w


def classify(x: list[int], w: list[int]) -> int:
    """
    Binary label under the linear rule: sign(w · x).

    Returns:
        +1  if dot product >= 0
        -1  otherwise
    """
    score = sum(a * b for a, b in zip(x, w))
    return 1 if score >= 0 else -1


def sample_balanced_support(
    w: list[int],
    n: int,
    dim: int,
    low: int,
    high: int,
) -> list[tuple[list[int], int]]:
    """
    Build a support set with as close to n//2 negative and n//2 positive
    examples as possible (balanced within ±1 for odd n).

    The support set is shuffled before returning so label order is random.
    """
    target_neg = n // 2
    target_pos = n - target_neg

    neg_examples: list[tuple[list[int], int]] = []
    pos_examples: list[tuple[list[int], int]] = []

    while len(neg_examples) < target_neg or len(pos_examples) < target_pos:
        x = sample_int_vector(dim=dim, low=low, high=high)
        y = classify(x, w)
        if y == -1 and len(neg_examples) < target_neg:
            neg_examples.append((x, y))
        elif y == 1 and len(pos_examples) < target_pos:
            pos_examples.append((x, y))

    support = neg_examples + pos_examples
    random.shuffle(support)
    return support


def sample_query(
    w: list[int],
    dim: int,
    low: int,
    high: int,
) -> tuple[list[int], int]:
    """Sample a single query vector and its gold label."""
    x = sample_int_vector(dim=dim, low=low, high=high)
    return x, classify(x, w)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_example(x: list[int], y: int) -> str:
    """Format a labeled support example as: [x1, x2, ...] = y"""
    return f"{x} = {y}"


def format_query(x: list[int]) -> str:
    """Format an unlabeled query as: [x1, x2, ...] = """
    return f"{x} = "


def build_prompt(
    support: list[tuple[list[int], int]],
    query_x: list[int],
) -> str:
    """
    Assemble the full few-shot prompt.

    Format:
        [x1, x2, x3, x4] = y
        [x1, x2, x3, x4] = y
        ...
        [x1, x2, x3, x4] =
    """
    lines = [format_example(x, y) for x, y in support]
    lines.append(format_query(query_x))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Candidate scoring  (the core inference routine)
# ---------------------------------------------------------------------------

def score_candidate(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    prompt: str,
    candidate: str,
) -> float:
    """
    Compute the total log-probability of `candidate` given `prompt`.

    Tokenizes prompt and candidate SEPARATELY, then concatenates at the token
    ID level.  This is the only safe approach: joint re-encoding of
    (prompt + candidate) is context-sensitive for BPE/SentencePiece tokenizers,
    meaning the number of tokens attributed to the candidate in the joint
    sequence can differ from standalone tokenization.  Explicit concatenation
    gives exact, verifiable control over the candidate token positions.

    Uses SUM (not mean) of token log-probs so that candidates of different
    lengths ("-1" may be 1-2 tokens; "1" is always 1 token) are compared on
    equal total-probability grounds.

    Returns:
        Sum of token-level log-probabilities.  Returns -inf for empty candidates.
    """
    device = next(model.parameters()).device

    prompt_ids: torch.Tensor = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(device)                          # [1, prompt_len]

    cand_ids: torch.Tensor = tokenizer(
        candidate,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)                          # [1, cand_len]

    cand_len: int = cand_ids.shape[1]
    if cand_len == 0:
        return float("-inf")

    # Concatenate at token level — avoids any BPE re-encoding at the join.
    full_ids = torch.cat([prompt_ids, cand_ids], dim=1)   # [1, prompt_len + cand_len]
    prompt_len: int = prompt_ids.shape[1]

    with torch.no_grad():
        logits: torch.Tensor = model(full_ids).logits      # [1, L, vocab]

    # Causal shift: logits[:, i, :] predicts token at position i+1.
    # Candidate tokens sit at positions [prompt_len, prompt_len + cand_len).
    # Their predictors are logits at [prompt_len-1, prompt_len + cand_len - 1).
    pred_logits = logits[0, prompt_len - 1 : prompt_len + cand_len - 1, :]  # [cand_len, vocab]

    log_probs = torch.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=1,
        index=cand_ids[0].unsqueeze(1),
    ).squeeze(1)                                           # [cand_len]

    return token_log_probs.sum().item()


def predict_label(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    prompt: str,
) -> tuple[int, dict[str, Any]]:
    """
    Predict binary label {-1, +1} by scoring both candidate strings.

    Includes a one-time debug assertion (only fires on the first call) that
    verifies each candidate tokenizes to at least one token, which catches
    misconfigured tokenizers early.
    """
    scores: dict[int, float] = {}
    for label, cand_str in LABEL_CANDIDATES.items():
        # Defensive check: ensure the candidate string is not empty after
        # tokenization.  An empty candidate (e.g. due to a stripped special
        # token) returns -inf and would silently bias every prediction.
        check_ids = tokenizer(cand_str, add_special_tokens=False).input_ids
        if len(check_ids) == 0:
            raise ValueError(
                f"Candidate string {cand_str!r} for label {label} tokenizes to "
                f"zero tokens with this tokenizer.  Check LABEL_CANDIDATES."
            )
        scores[label] = score_candidate(tokenizer, model, prompt, cand_str)

    predicted: int = max(scores, key=lambda lbl: scores[lbl])
    return predicted, {"scores": scores}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    num_fewshot: int,
    limit: int,
    seed: int,
    dim: int = 4,
    low: int = -10,
    high: int = 10,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run the synthetic linear ICL evaluation.

    For each of `limit` instances:
        1. Sample a random non-zero linear separator w.
        2. Sample a balanced few-shot support set of size `num_fewshot`.
        3. Sample a query (x, gold_label).
        4. Build the prompt and score both candidates.
        5. Record correctness.

    Args:
        tokenizer:    HuggingFace tokenizer.
        model:        Causal LM in eval mode.
        num_fewshot:  Number of labeled examples in the prompt context.
        limit:        Number of evaluation instances to generate.
        seed:         Random seed for reproducibility.
        dim:          Dimensionality of the integer vectors.
        low:          Lower bound (inclusive) for integer sampling.
        high:         Upper bound (inclusive) for integer sampling.
        verbose:      If True, print per-example results to stdout.

    Returns:
        Result dict containing aggregate metrics and per-example debug data.
    """
    set_seed(seed)

    if verbose:
        print(f"Starting eval: limit={limit}, num_fewshot={num_fewshot}, "
              f"dim={dim}, low={low}, high={high}, seed={seed}")

    start_time = time.perf_counter()
    total = 0
    correct = 0
    examples: list[dict[str, Any]] = []

    for idx in range(limit):
        # --- generate a fresh classification problem ---
        w = make_linear_separator(dim=dim, low=low, high=high)
        support = sample_balanced_support(w=w, n=num_fewshot, dim=dim, low=low, high=high)
        query_x, gold_y = sample_query(w=w, dim=dim, low=low, high=high)

        prompt = build_prompt(support, query_x)
        pred_y, debug_info = predict_label(tokenizer, model, prompt)

        is_correct = pred_y == gold_y
        total += 1
        correct += int(is_correct)

        examples.append({
            "index": idx,
            "gold": gold_y,
            "pred": pred_y,
            "correct": is_correct,
            "scores": {str(k): v for k, v in debug_info["scores"].items()},
            "prompt_preview": prompt[:500],
        })

        if verbose:
            print(
                f"[{idx + 1:>4}/{limit}] "
                f"pred={pred_y:+d}  gold={gold_y:+d}  "
                f"correct={is_correct}  "
                f"running_acc={correct / total:.3f}"
            )

    elapsed = time.perf_counter() - start_time
    accuracy = correct / total if total > 0 else 0.0

    return {
        "task": "synthetic_linear_icl",
        "metric_name": "accuracy",
        "metric_value": accuracy,
        "metric_stderr": None,
        "sample_len": total,
        "correct": correct,
        "eval_time_sec": round(elapsed, 4),
        "num_fewshot": num_fewshot,
        "limit": limit,
        "seed": seed,
        "dim": dim,
        "low": low,
        "high": high,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic linear ICL evaluator for pruned/base LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the model config YAML.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Optional local checkpoint directory to override the config's HF model name.",
    )
    parser.add_argument(
        "--num_fewshot", type=int, required=True,
        help="Number of labeled in-context examples per prompt.",
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Number of evaluation instances (sampled classification problems).",
    )
    parser.add_argument(
        "--seed", type=int, default=13,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--dim", type=int, default=4,
        help="Dimensionality of the integer vectors.",
    )
    parser.add_argument(
        "--low", type=int, default=-10,
        help="Lower bound (inclusive) for uniform integer sampling.",
    )
    parser.add_argument(
        "--high", type=int, default=10,
        help="Upper bound (inclusive) for uniform integer sampling.",
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Destination path for the raw JSON result file.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-example predictions and running accuracy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer, model, cfg, model_source = load_tokenizer_and_model(
        args.config,
        checkpoint_path=args.checkpoint_path,
    )

    result = run_eval(
        tokenizer=tokenizer,
        model=model,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        seed=args.seed,
        dim=args.dim,
        low=args.low,
        high=args.high,
        verbose=args.verbose,
    )

    # Attach model metadata to the result before saving.
    result["model_name"] = model_source
    result["model_dtype"] = str(next(model.parameters()).dtype)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    # Print a compact summary to stdout regardless of --verbose.
    summary_keys = (
        "task", "metric_name", "metric_value", "correct", "sample_len",
        "eval_time_sec", "num_fewshot", "dim", "seed",
        "model_name", "model_dtype",
    )
    print("\n=== Evaluation complete ===")
    print(json.dumps({k: result[k] for k in summary_keys}, indent=2))
    print(f"Full results written to: {output_path}")


if __name__ == "__main__":
    main()