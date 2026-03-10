import pickle
import os
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_PATH


def save_model(model: BaseEstimator, name: str):
    """Save a trained model to disk and return its path."""
    model_path = os.path.join(MODEL_PATH, f"{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def load_model(name: str) -> BaseEstimator:
    """Load a previously saved model from disk."""
    model_path = os.path.join(MODEL_PATH, f"{name}.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)
