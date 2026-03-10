# Import other necessary libraries here
import os
import pandas as pd
import loguru
from diamonds.params import DATA_PATH
import seaborn as sns
from diamonds.model import create_preproc

logger = loguru.logger

df_diamonds = sns.load_dataset("diamonds")


def load_data() -> pd.DataFrame:
    df_diamonds = sns.load_dataset("diamonds")
    """
    Load the diamonds dataset.

    Parameters
    ----------
    cache : bool, optional
        Whether to cache the dataset, by default True

    Returns
    -------
    pd.DataFrame
        The diamonds dataset
    """
    logger.info("Loading diamonds dataset...")
    csv_path = os.path.join(DATA_PATH, "raw", "diamonds.csv")
    if not os.path.exists(csv_path):
        logger.info("Caching the diamonds dataset...")
        df_diamonds = sns.load_dataset("diamonds")
        df_diamonds.to_csv(csv_path, index=False)
    else:
        logger.info("Loading diamonds dataset from cache...")
        df_diamonds = pd.read_csv(csv_path)
    return df_diamonds


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    def keep_not_allow(row):
        if 0 in row:
            return False
        return True

    df_diamonds = df[df.apply(keep_not_allow, axis=1)]
    df_diamonds[df_diamonds["x"] == 0]
    """
    Clean the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The diamonds dataset

    Returns
    -------
    pd.DataFrame
        The cleaned diamonds dataset
    """

    return df_diamonds


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned diamonds dataset

    Returns
    -------
    pd.DataFrame
        The preprocessed diamonds dataset
    """
    preprocessor = create_preproc()
    df_preprocessed = preprocessor.transform(df)

    return df_preprocessed


def create_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Split target first so preprocessing columns only reference feature columns.
    X = df.drop(columns="price")
    y = df["price"]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    preprocessor = create_preproc(num_cols, cat_cols)
    """
    Create the feature matrix X and target vector y from the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed diamonds dataset

    Returns
    -------
    (pd.DataFrame, pd.Series)
        The feature matrix X and target vector y
    """
    X_transform = preprocessor.fit_transform(X)
    X_transform = pd.DataFrame(
        X_transform,
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )
    return X_transform, y


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    df_preprocessed = preprocess_data(df_clean)
    X, y = create_X_y(df_preprocessed)
