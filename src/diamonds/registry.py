
from pathlib import Path

from joblib import dump, load
from sklearn.base import BaseEstimator

from diamonds.params import MODEL_REGISTRY


def save_model(model: BaseEstimator, path: str | Path) -> Path:
    """Save a trained model to disk and return its path."""
    if MODEL_REGISTRY != "local":
        raise NotImplementedError(f"Unsupported model registry: {MODEL_REGISTRY}")

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, target_path)
    return target_path


def load_model(path: str | Path) -> BaseEstimator:
    """Load a previously saved model from disk."""
    if MODEL_REGISTRY != "local":
        raise NotImplementedError(f"Unsupported model registry: {MODEL_REGISTRY}")

    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Model file not found: {target_path}")

    return load(target_path)