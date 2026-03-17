METHOD_BACKEND = {
    "dense": "none",
    "magnitude": "ria_core",
    "wanda": "ria_core",
    "ria": "ria_core",
    "sparsegpt": "ria_core",
    "wandapp": "local",
}


def get_backend(method: str) -> str:
    if method not in METHOD_BACKEND:
        raise ValueError(f"Unknown pruning method: {method}")
    return METHOD_BACKEND[method]