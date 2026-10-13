from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "bank-full.csv"
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "decision_tree_final.joblib"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH, sep=";")


def prepare_features(df):
    df = df.copy()

    y = df["y"].map({"no": 0, "yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected target values found.")

    X = df.drop(columns=["y", "duration"])

    X["previously_contacted"] = (
        X["pdays"] != -1
    ).astype(int)

    return X, y


def build_pipeline():
    categorical_features = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]

    numerical_features = [
        "age",
        "balance",
        "day",
        "campaign",
        "pdays",
        "previous",
        "previously_contacted",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Training completed.")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()