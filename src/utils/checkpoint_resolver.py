from pathlib import Path

PRUNED_MODEL_ROOT = Path("/data/ankritgupta/iclprune/wanda")

def resolve_pruned_checkpoint(model: str, method: str, sparsity: int) -> str:
    parent = PRUNED_MODEL_ROOT / f"{method}_{sparsity}"

    if not parent.exists():
        raise FileNotFoundError(f"Missing checkpoint parent directory: {parent}")

    candidates = [
        p for p in parent.iterdir()
        if p.is_dir() and p.name.startswith(f"{model}_{sparsity}_")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for model={model}, method={method}, sparsity={sparsity} under {parent}"
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    chosen = candidates[-1]

    required_files = [
        chosen / "config.json",
        chosen / "model.safetensors.index.json",
    ]
    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"Resolved checkpoint is incomplete. Missing: {f}")

    return str(chosen)