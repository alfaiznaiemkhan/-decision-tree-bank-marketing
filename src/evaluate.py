from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "bank-full.csv"
MODEL_PATH = ROOT_DIR / "model" / "decision_tree_final.joblib"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run src/train.py first."
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    y = df["y"].map({"no": 0, "yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected target values found.")

    X = df.drop(columns=["y", "duration"]).copy()

    X["previously_contacted"] = (
        X["pdays"] != -1
    ).astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))

    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Subscription", "Subscription"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()