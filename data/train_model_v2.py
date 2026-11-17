
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_FILE = Path("data/cleaned_data.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("evaluation_results")

MODEL_FILE = MODEL_DIR / "churn_model_v2.joblib"
METRICS_FILE = RESULT_DIR / "model_v2_metrics.json"


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def find_best_threshold(y_true, probabilities):
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(
            y_true, probabilities
        )),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found.")

    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected or missing target labels.")

    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No usable features found.")

    # 60% train, 20% validation, 20% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("Dataset split:")
    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))

    preprocessor = build_preprocessor(
        numeric_features, categorical_features
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for name, classifier in candidates.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(
                numeric_features, categorical_features
            )),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        val_probabilities = pipeline.predict_proba(X_val)[:, 1]

        val_pr_auc = average_precision_score(
            y_val, val_probabilities
        )

        threshold, val_f1 = find_best_threshold(
            y_val, val_probabilities
        )

        results[name] = {
            "validation_pr_auc": float(val_pr_auc),
            "validation_f1": float(val_f1),
            "threshold": threshold,
        }

        fitted_models[name] = pipeline

        print("Validation PR-AUC:", round(val_pr_auc, 4))
        print("Validation F1:", round(val_f1, 4))
        print("Selected threshold:", round(threshold, 2))

    # Select using validation PR-AUC only.
    best_name = max(
        results,
        key=lambda name: results[name]["validation_pr_auc"],
    )

    best_model = fitted_models[best_name]
    best_threshold = results[best_name]["threshold"]

    # Evaluate the selected model on the untouched test set.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate(
        y_test, test_probabilities, best_threshold
    )

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\nBEST MODEL:", best_name)
    print("\nFINAL TEST METRICS")
    print(json.dumps(test_metrics, indent=2))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        target_names=["Retained", "Churned"],
        zero_division=0,
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "threshold": best_threshold,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_metrics": test_metrics,
        "validation_results": results,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(bundle, MODEL_FILE)

    report = {
        "best_model": best_name,
        "validation_results": results,
        "test_metrics": test_metrics,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nModel saved:", MODEL_FILE)
    print("Metrics saved:", METRICS_FILE)


if __name__ == "__main__":
    main()
