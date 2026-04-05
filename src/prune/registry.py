METHOD_BACKEND = {
    "dense": "none",
    "magnitude": "ria",
    "wanda": "ria",
    "ria": "ria",
    "sparsegpt": "ria",
    "wandapp": "local",
}


def get_backend(method: str) -> str:
    if method not in METHOD_BACKEND:
        raise ValueError(f"Unknown pruning method: {method}")
    return METHOD_BACKEND[method]