# Import other necessary libraries here
import pandas as pd
import seaborn as sns
from diamonds.model import create_preproc

df_diamonds = sns.load_dataset("diamonds")

def load_data(cache = True) -> pd.DataFrame:
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

    return df_diamonds

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    def keep_not_allow(row):
        if 0 in row: return  False
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

    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_diamonds = df.select_dtypes(include=["category"])
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

    return df

def create_X_y(df: pd.DataFrame) ->tuple[pd.DataFrame, pd.Series]:
    # cat_pipe = Pipeline(
    #     [("cat_imp",SimpleImputer(strategy="most_frequent")), 
    #      ("onehot", OneHotEncoder(dropt="first"), sparsde_output=True)]
    # )
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="category").columns.tolist()
    preprocessor = create_preproc()
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
    return preprocessor
    



if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    df_preprocessed = preprocess_data(df_clean)
    # X, y = create_X_y(df_preprocessed)