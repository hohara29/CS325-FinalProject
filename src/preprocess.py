import pandas as pd
from sklearn.model_selection import train_test_split

def clean_features(X: pd.DataFrame, missing_threshold: float = 0.50):
    #Cleans feature dataframe by:
        # dropping ID if present
        # dropping columns with too many mssing values
        # dropping constant columsn

    # Args:
        # X : raw feature dataframe
        # missing_threshold : max allowed fraction of missing values

    # Returns:
        # X_cleaned : cleaned feature dataframe
        # dropped_info : info about dropped columns

    X = X.copy()
    dropped_info = {
        "dropped_id": [],
        "dropped_high_missing": [],
        "dropped_constant": []
    }

    # Drop ID column if present
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])
        dropped_info["dropped_id"].append("ID")

    # Drop columns with too much missing data
    missing_fraction = X.isna().mean()
    high_missing_cols = missing_fraction[missing_fraction > missing_threshold].index.tolist()
    if high_missing_cols:
        X = X.drop(columns=high_missing_cols)
        dropped_info["dropped_high_missing"] = high_missing_cols

    # Drop constant columns
    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        dropped_info["dropped_constant"] = constant_cols

    return X, dropped_info

def select_target(y_all: pd.DataFrame, target_col: str):
    # Selects a single target column from all target columns

    # Args:
        # y_all : all output columns
        # target_col : target column name

    # Return:
        # y : slected target

    if target_col not in y_all.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(y_all.columns)}")

    return y_all[target_col].copy()

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    # Splits data into train and test sets using stratification

    # Returns:
        # X_train, X_test, y_train, y_test

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )