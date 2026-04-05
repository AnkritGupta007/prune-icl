"""
Paper-aligned synthetic in-context linear classification evaluator.

What this file does:
- loads a causal LM using the shared project loader
- creates a random 2-way linear classification function in D=4
- builds paper-style prompts of the form "[x1, x2, x3, x4] = y"
- evaluates exact-match label prediction accuracy
- saves one raw JSON result file

Why we are doing this:
- this matches the linear classification ICL task from
  "The Cost of Down-Scaling Language Models"
- this is the mechanism-oriented ICL task from the project plan
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from src.utils.load_model import load_tokenizer_and_model


def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_int_vector(dim: int = 4, low: int = -10, high: int = 10) -> list[int]:
    """
    Sample one integer feature vector x in [-10, 10]^dim.
    """
    return [random.randint(low, high) for _ in range(dim)]


def make_linear_separator(dim: int = 4, low: int = -10, high: int = 10) -> list[int]:
    """
    Sample a random linear separator vector w.

    We reject the all-zero vector so the classifier is non-trivial.
    """
    while True:
        w = [random.randint(low, high) for _ in range(dim)]
        if any(v != 0 for v in w):
            return w


def classify(x: list[int], w: list[int]) -> int:
    """
    2-way linear classification using sign(w·x).

    Returns:
    -1 if score < 0
     1 if score >= 0
    """
    score = sum(a * b for a, b in zip(x, w))
    return 1 if score >= 0 else 0


def format_example(x: list[int], y: int) -> str:
    """
    Format one support example in the paper style.
    Example:
    [5, -5, -1, 6] = -1
    """
    return f"{x} = {y}"


def format_query(x: list[int]) -> str:
    """
    Format the query line in the paper style.
    Example:
    [3, 1, -2, 7] =
    """
    return f"{x} = "


def build_prompt(support: list[tuple[list[int], int]], query_x: list[int]) -> str:
    """
    Build a paper-style prompt with support examples followed by the query.
    """
    lines = [format_example(x, y) for x, y in support]
    lines.append(format_query(query_x))
    return "\n".join(lines)


def extract_label(text: str) -> int | None:
    """
    Extract the predicted label from generated text.

    Accepted outputs:
    -1
     1

    We check the first few non-space characters conservatively.
    """
    s = text.strip()

    if s.startswith("-1"):
        return -1
    if s.startswith("1"):
        return 1

    # fallback, still conservative
    if "-1" in s[:8]:
        return -1
    if "1" in s[:4]:
        return 1

    return None


@torch.no_grad()
def generate_label(tokenizer, model, prompt: str, max_new_tokens: int = 1) -> tuple[int | None, str]:
    """
    Generate a label from the model for one prompt.

    Returns:
    - predicted label in {-1, 1} or None
    - raw decoded generated suffix for debugging
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    pred = extract_label(decoded)
    return pred, decoded

@torch.no_grad()
def score_candidate(tokenizer, model, prompt: str, candidate: str) -> float:
    """
    Score one candidate continuation using conditional log-probability.

    We compute:
    log p(candidate | prompt)

    Returns:
        Sum of token log-probabilities for the candidate continuation.
    """
    device = next(model.parameters()).device

    full_text = prompt + candidate

    # Tokenize prompt and full text separately so we know which tokens belong
    # to the continuation we want to score.
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    # Forward pass over the full sequence.
    outputs = model(full_ids)
    logits = outputs.logits[:, :-1, :]
    target_ids = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)

    # Continuation tokens start after the prompt tokens.
    prompt_len = prompt_ids.shape[1]
    continuation_start = prompt_len - 1

    continuation_log_probs = log_probs[:, continuation_start:, :]
    continuation_target_ids = target_ids[:, continuation_start:]

    token_log_probs = continuation_log_probs.gather(
        dim=-1,
        index=continuation_target_ids.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs.mean().item()


@torch.no_grad()
def predict_label(tokenizer, model, prompt: str) -> tuple[int, dict]:
    """
    Predict between the two allowed labels {-1, 1} by scoring both candidates.

    Returns:
        predicted_label
        debug_info
    """
    candidates = {
        0: "0",
        1: "1",
    }

    scores = {
        label: score_candidate(tokenizer, model, prompt, cand)
        for label, cand in candidates.items()
    }

    pred = max(scores.items(), key=lambda x: x[1])[0]
    return pred, {"scores": scores}

def sample_balanced_support(
    w: list[int],
    n: int,
    dim: int = 4,
    low: int = -10,
    high: int = 10,
) -> list[tuple[list[int], int]]:
    """
    Sample a roughly balanced support set for labels -1 and 1.

    We keep drawing points until we have n total examples,
    aiming for half from each class.
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
    dim: int = 4,
    low: int = -10,
    high: int = 10,
) -> tuple[list[int], int]:
    """
    Sample one query example.
    """
    x = sample_int_vector(dim=dim, low=low, high=high)
    y = classify(x, w)
    return x, y


def run_eval(
    tokenizer,
    model,
    num_fewshot: int,
    limit: int,
    seed: int,
    dim: int = 4,
) -> dict:
    """
    Run the paper-aligned linear classification ICL evaluation.

    Notes:
    - each evaluation example uses a newly sampled linear function
    - each function gets its own support examples and query
    - metric is exact-match accuracy on {-1, 1}
    """
    set_seed(seed)

    start = time.time()
    total = 0
    correct = 0
    examples = []

    print("Loaded model and tokenizer")
    print("Building synthetic dataset...")
    print(f"Number of eval instances: {limit}")
    print("Starting evaluation loop...")

    for idx in range(limit):
        print(f"Running example {idx+1}/{limit}")
        w = make_linear_separator(dim=dim)
        support = sample_balanced_support(w=w, n=num_fewshot, dim=dim)
        query_x, gold_y = sample_query(w=w, dim=dim)

        prompt = build_prompt(support, query_x)
        print("Prompt:")
        print(prompt)
        print("Scoring candidates...")
        pred_y, debug_info = predict_label(tokenizer, model, prompt)
        decoded = str(debug_info)
        

        is_correct = pred_y == gold_y
        total += 1
        correct += int(is_correct)

        examples.append(
            {
                "index": idx,
                "gold": gold_y,
                "pred": pred_y,
                "correct": is_correct,
                "generated_text": decoded,
                "prompt_preview": prompt[:500],
            }
        )
        print(f"Decoded: {decoded!r}, Pred: {pred_y}, Gold: {gold_y}")
        print(f"Finished example {idx+1}/{limit}")

    acc = correct / total if total > 0 else 0.0
    elapsed = time.time() - start

    return {
        "task": "synthetic_linear_icl",
        "metric_name": "acc",
        "metric_value": acc,
        "metric_stderr": None,
        "sample_len": total,
        "eval_time_sec": elapsed,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "examples": examples,
    }


def main() -> None:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config YAML.")
    parser.add_argument("--num_fewshot", type=int, required=True,
                        help="Number of in-context examples.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of sampled function instances to evaluate.")
    parser.add_argument("--seed", type=int, default=13,
                        help="Random seed.")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to save raw JSON result.")
    args = parser.parse_args()

    tokenizer, model, cfg = load_tokenizer_and_model(args.config)

    result = run_eval(
        tokenizer=tokenizer,
        model=model,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        seed=args.seed,
    )

    result["model_name"] = cfg["hf_model_name"]
    result["model_dtype"] = str(next(model.parameters()).dtype)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Synthetic linear ICL eval completed.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()