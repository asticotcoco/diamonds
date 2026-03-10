import os
from diamonds.data import load_data, clean_data, preprocess_data, create_X_y
from diamonds.model import create_model, train_model, evaluate_model
from sklearn.model_selection import train_test_split
from diamonds.registry import save_model
from diamonds.params import MODEL_PATH


def train(
    model_name: str = "baseline",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Simple end‑to‑end pipeline:

    - load and clean the raw data
    - preprocess it and build X, y
    - split into train / test
    - build the model and preprocessing
    - train, evaluate, and save the trained model
    """
    # 1) Data
    df = load_data()
    df_clean = clean_data(df)
    df_preprocessed = preprocess_data(df_clean)
    X, y = create_X_y(df_preprocessed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 2) Model + preprocessing
    resolved_model_name = "linear" if model_name == "baseline" else model_name
    model = create_model(resolved_model_name)
    trained_model = train_model(model, X_train, y_train)

    # 3) Evaluation
    evaluate_model(trained_model, X_test, y_test)

    # 4) Persistence
    model_dir = os.path(MODEL_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{resolved_model_name}.joblib"
    saved_path = save_model(trained_model, model_path)
    print(f"Model saved to: {saved_path}")


if __name__ == "__main__":
    train()
