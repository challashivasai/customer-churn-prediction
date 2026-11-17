"""Explain the saved customer-churn model with SHAP.

This script only reads the saved model bundle and processed dataset. It does
not retrain the model or overwrite models/best_churn_model.joblib.
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


MODEL_PATH = "models/best_churn_model.joblib"
DATA_PATH = "data/processed_data.csv"
OUTPUT_DIR = "explanation_results"
RANDOM_STATE = 42
MAX_EXPLAIN_ROWS = 500
MAX_BACKGROUND_ROWS = 100


def select_positive_class(values, base_values, n_features):
    """Normalize SHAP output to the positive (Churn=1) class."""
    if isinstance(values, list):
        # Older SHAP versions return one array per class.
        values = values[1] if len(values) > 1 else values[0]

    values = np.asarray(values)
    base_values = np.asarray(base_values)

    # Newer SHAP versions can return (rows, features, classes).
    if values.ndim == 3:
        values = values[:, :, 1]

    if values.ndim != 2 or values.shape[1] != n_features:
        raise ValueError(
            f"Unexpected SHAP value shape {values.shape}; "
            f"expected (rows, {n_features}) for Churn=1."
        )

    # Binary classifiers may return one base value per class.
    if base_values.ndim == 2:
        if base_values.shape[1] > 1:
            base_values = base_values[:, 1]
        else:
            base_values = base_values[:, 0]
    elif base_values.ndim == 1 and len(base_values) == 2 and len(values) != 2:
        # Some SHAP versions return one expected value per class.
        base_values = np.repeat(float(base_values[1]), len(values))

    if base_values.ndim == 0:
        base_values = np.repeat(float(base_values), len(values))
    elif base_values.ndim == 1 and len(base_values) == 1:
        base_values = np.repeat(float(base_values[0]), len(values))

    return values, base_values


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    print("Loading saved model bundle...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    model_name = bundle.get("model_name", type(model).__name__)
    threshold = float(bundle.get("threshold", 0.5))
    feature_columns = bundle.get("feature_columns")

    if not feature_columns:
        raise ValueError("The saved model bundle has no feature_columns list.")

    print(f"Model: {model_name}")
    print(f"Saved decision threshold: {threshold:.4f}")
    print(f"Expected input features: {len(feature_columns)}")

    print("\nLoading processed dataset...")
    df = pd.read_csv(DATA_PATH)
    if "Churn" not in df.columns:
        raise ValueError("The processed dataset must contain the 'Churn' column.")

    X = df.drop(columns=["Churn"])
    missing = [column for column in feature_columns if column not in X.columns]
    extra = [column for column in X.columns if column not in feature_columns]

    if missing:
        raise ValueError(f"Dataset is missing model features: {missing}")
    if extra:
        print(f"Ignoring {len(extra)} dataset column(s) not used by the saved model.")

    # Preserve the exact feature order used during training.
    X = X.loc[:, feature_columns]
    if X.isna().any().any():
        raise ValueError("Missing feature values found in processed_data.csv.")

    rng = np.random.RandomState(RANDOM_STATE)
    explain_count = min(MAX_EXPLAIN_ROWS, len(X))
    background_count = min(MAX_BACKGROUND_ROWS, len(X))
    explain_indices = rng.choice(X.index.to_numpy(), size=explain_count, replace=False)
    background_indices = rng.choice(X.index.to_numpy(), size=background_count, replace=False)
    X_explain = X.loc[explain_indices].copy()
    X_background = X.loc[background_indices].copy()

    print(f"Rows selected for explanation: {len(X_explain)}")
    print(f"Background rows: {len(X_background)}")

    # Use the explainer suited to the model saved by train_model.py.
    if isinstance(model, Pipeline) and "model" in model.named_steps:
        final_estimator = model.named_steps["model"]

        if isinstance(final_estimator, LogisticRegression):
            print("\nUsing LinearExplainer for the Logistic Regression pipeline...")
            if "scaler" in model.named_steps:
                X_background_model = model.named_steps["scaler"].transform(X_background)
                X_explain_model = model.named_steps["scaler"].transform(X_explain)
            else:
                X_background_model = X_background
                X_explain_model = X_explain

            explainer = shap.LinearExplainer(final_estimator, X_background_model)
            raw_shap_values = explainer.shap_values(X_explain_model)
            raw_base_values = explainer.expected_value

        else:
            print("Using model-agnostic SHAP explainer for this pipeline...")
            explainer = shap.Explainer(model.predict_proba, X_background)
            shap_output = explainer(X_explain)
            raw_shap_values = shap_output.values
            raw_base_values = shap_output.base_values

    elif isinstance(model, RandomForestClassifier):
        print("\nUsing TreeExplainer for the Random Forest model...")
        explainer = shap.TreeExplainer(model)
        shap_output = explainer(X_explain)
        raw_shap_values = shap_output.values
        raw_base_values = shap_output.base_values

    else:
        print("Using model-agnostic SHAP explainer...")
        explainer = shap.Explainer(model.predict_proba, X_background)
        shap_output = explainer(X_explain)
        raw_shap_values = shap_output.values
        raw_base_values = shap_output.base_values

    shap_values, base_values = select_positive_class(
        raw_shap_values, raw_base_values, len(feature_columns)
    )

    # For Logistic Regression, LinearExplainer values explain the model's
    # linear output (log-odds), not probability percentage points.
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_explain.to_numpy(),
        feature_names=feature_columns,
    )

    print("\nGenerating global SHAP plots...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_feature_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Global importance table: mean absolute SHAP value per feature.
    importance = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"), index=False)

    # Explain the highest predicted churn-risk customer in the sampled rows.
    probabilities = model.predict_proba(X_explain)[:, 1]
    highest_risk_position = int(np.argmax(probabilities))
    highest_risk_probability = float(probabilities[highest_risk_position])
    local_row = X_explain.iloc[highest_risk_position]
    local_values = shap_values[highest_risk_position]
    local_base = float(base_values[highest_risk_position])

    local_explanation = shap.Explanation(
        values=local_values,
        base_values=local_base,
        data=local_row.to_numpy(),
        feature_names=feature_columns,
    )

    plt.figure()
    shap.plots.waterfall(local_explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_highest_risk_customer.png"), dpi=200, bbox_inches="tight")
    plt.close()

    local_details = pd.DataFrame({
        "feature": feature_columns,
        "feature_value": local_row.to_numpy(),
        "shap_value": local_values,
    })
    local_details["impact"] = np.where(
        local_details["shap_value"] > 0,
        "Increases churn score",
        np.where(local_details["shap_value"] < 0, "Reduces churn score", "No impact"),
    )
    local_details.reindex(
        local_details["shap_value"].abs().sort_values(ascending=False).index
    ).to_csv(os.path.join(OUTPUT_DIR, "shap_highest_risk_customer.csv"), index=False)

    # Save a compact summary of the explained customer's prediction.
    prediction_summary = pd.DataFrame([{
        "model_name": model_name,
        "saved_threshold": threshold,
        "churn_probability": highest_risk_probability,
        "predicted_class_at_saved_threshold": int(highest_risk_probability >= threshold),
        "explained_dataset_row_index": X_explain.index[highest_risk_position],
        "note": "This is the highest-risk row in the SHAP sample, not necessarily a live customer.",
    }])
    prediction_summary.to_csv(os.path.join(OUTPUT_DIR, "shap_prediction_summary.csv"), index=False)

    print("\n========== SHAP EXPLANATION COMPLETED ==========")
    print("Highest sampled churn probability:", f"{highest_risk_probability:.2%}")
    print("Saved threshold:", f"{threshold:.4f}")
    print("Prediction at saved threshold:", "Churn" if highest_risk_probability >= threshold else "Stay")
    print("\nTop 10 global churn factors:")
    print(importance.head(10).to_string(index=False))
    print(f"\nAll explanation files saved in: {OUTPUT_DIR}/")
    print("The original trained model file was only read, not modified.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
