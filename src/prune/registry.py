METHOD_BACKEND = {
    "dense": "none",
    "magnitude": "wandaplus",
    "wanda": "wanda",
    "ria": "ria",
    "sparsegpt": "wandaplus",
    "wandaplus": "wandaplus",
    "wanda_owl": "wanda_owl",
}


def get_backend(method: str) -> str:
    if method not in METHOD_BACKEND:
        raise ValueError(f"Unknown pruning method: {method}")
    return METHOD_BACKEND[method]