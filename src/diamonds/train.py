import random

import mlflow
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
<<<<<<< HEAD
from diamonds.data import load_data
import random

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None
=======
from sklearn.model_selection import train_test_split

from diamonds.data import clean_data, create_X_y, load_data, preprocess_data
from diamonds.model import create_model, create_training_pipeline, evaluate_model, train_model
from diamonds.params import MLFLOW_TRACKING_URI
>>>>>>> 848d30b (clean a bit the code and add CI to check linter when pushing)


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
    # 2) Model + preprocessing
    X, y = create_X_y(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_preproc = preprocess_data(X_train, train=True)
    X_test_preproc = preprocess_data(X_test, train=False)

    model = create_model(model_name)
    train_model(model, X_train_preproc, y_train)
    # 3) Evaluation
    evaluate_model(model, X_test_preproc, y_test)


def autolog_mlflow(
    model_name: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
):
    if mlflow is None:
        raise ModuleNotFoundError(
            "mlflow is not installed in the active Python interpreter. "
            "Install mlflow or run train() without experiment tracking."
        )

    # 1. Set the tracking URI to the MLflow server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # 2. Set the Experiment name
    mlflow.set_experiment("Diamonds_experiment")
    mlflow.autolog(log_input_examples=True)
    n_estimators = random.randint(50, 200)  # Randomly choose n_estimators for demonstration
    depth = random.randint(5, 20)
    # 1) Data
    df = load_data()
    df_clean = clean_data(df)
    # 2) Model + preprocessing
    X, y = create_X_y(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline = create_training_pipeline(
        model_name=model_name,
        estimators=n_estimators,
        max_depth=depth,
        random_state=random_state,
    )
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        print(f"Model: {model_name}")
        print(f"n_estimators: {n_estimators}")
        print(f"max_depth: {depth}")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2:   {r2:.4f}")
        return pipeline


if __name__ == "__main__":
<<<<<<< HEAD
    if mlflow is None:
        print("mlflow not found in this interpreter. Running training without tracking.")
        train("random_forest")
    else:
        autolog_mlflow()
=======
    autolog_mlflow()
    # train("random_forest")
>>>>>>> 848d30b (clean a bit the code and add CI to check linter when pushing)
