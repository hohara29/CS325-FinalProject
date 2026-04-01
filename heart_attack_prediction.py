from src.load_data import load_uci_mi_data
from src.preprocess import clean_features, select_target, split_data
from src.train_model import build_logistic_regression_model, build_random_forest_model
from src.evaluate import evaluate_model

def main():
    # 1. Load data
    X, y_all = load_uci_mi_data()

    print("Original feature shape:", X.shape)
    print("Available target columns:")
    print(list(y_all.columns))

    # 2. Choose target
    target_col = "REC_IM"
    y = select_target(y_all, target_col)

    print(f"\nSelected target: {target_col}")
    print("Target distribution:")
    print(y.value_counts(dropna=False))
    print(y.value_counts(normalize=True, dropna=False))

    # 3. Clean features
    X_clean, dropped_info = clean_features(X, missing_threshold=0.50)

    print("\nCleaned feature shape:", X_clean.shape)
    print("Dropped columns summary:")
    for key, value in dropped_info.items():
        print(f"{key}: {len(value)}")

    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(X_clean, y)