
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_model_v2.joblib"
DATA_PATH = BASE_DIR / "data" / "cleaned_data.csv"

OUTPUT_DIR = BASE_DIR / "explanation_results"

RANDOM_STATE = 42
BACKGROUND_SIZE = 20
EXPLANATION_SIZE = 30


# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def to_dense_array(data):
    """Convert transformed sparse or dense data into NumPy."""

    if hasattr(data, "toarray"):
        data = data.toarray()

    return np.asarray(data, dtype=float)


def get_feature_names(preprocessor, feature_count):
    """Get names of transformed model features."""

    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return [
            f"transformed_feature_{i}"
            for i in range(feature_count)
        ]


# =====================================================
# 3. MAIN SHAP ANALYSIS
# =====================================================

def main():

    # Import optional packages
    try:
        import shap
        import matplotlib.pyplot as plt

    except ImportError as error:
        raise RuntimeError(
            "Required packages are missing. Run:\n"
            r".\venv\Scripts\python.exe -m pip install shap matplotlib"
        ) from error

    # Check files
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("=" * 55)
    print("CUSTOMER CHURN - GLOBAL SHAP ANALYSIS")
    print("=" * 55)

    # -------------------------------------------------
    # LOAD MODEL AND DATASET
    # -------------------------------------------------

    print("\nLoading trained model...")

    bundle = joblib.load(MODEL_PATH)

    print("Loading cleaned dataset...")

    data = pd.read_csv(DATA_PATH)

    model = bundle["model"]

    numeric_features = list(
        bundle["numeric_features"]
    )

    categorical_features = list(
        bundle["categorical_features"]
    )

    input_features = (
        numeric_features + categorical_features
    )

    missing_columns = [
        col for col in input_features
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    X = data[input_features].copy()

    # Validate numeric columns
    for col in numeric_features:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    if X[numeric_features].isna().any().any():
        raise ValueError(
            "Invalid numeric values found in the dataset."
        )

    # -------------------------------------------------
    # EXTRACT TRAINED PREPROCESSING PIPELINE
    # -------------------------------------------------

    if not hasattr(model, "steps"):
        raise TypeError(
            "Expected the saved model to be a fitted "
            "scikit-learn Pipeline."
        )

    if len(model.steps) < 2:
        raise ValueError(
            "The saved pipeline must contain a "
            "preprocessor and a final estimator."
        )

    # All steps except the final estimator
    preprocessor = model[:-1]

    # Final trained classifier
    classifier = model.steps[-1][1]

    if not hasattr(classifier, "predict_proba"):
        raise TypeError(
            "The final classifier does not support predict_proba."
        )

    print("\nApplying the trained preprocessing pipeline...")

    transformed = preprocessor.transform(X)

    transformed = to_dense_array(transformed)

    feature_names = get_feature_names(
        preprocessor,
        transformed.shape[1]
    )

    if transformed.shape[1] != len(feature_names):
        raise ValueError(
            "Transformed feature names do not match data dimensions."
        )

    print(
        f"Original input features: {len(input_features)}"
    )

    print(
        f"Transformed model features: {transformed.shape[1]}"
    )

    # -------------------------------------------------
    # SELECT BACKGROUND AND EXPLANATION SAMPLES
    # -------------------------------------------------

    rng = np.random.RandomState(RANDOM_STATE)

    background_size = min(
        BACKGROUND_SIZE,
        len(transformed)
    )

    explanation_size = min(
        EXPLANATION_SIZE,
        len(transformed)
    )

    background_indices = rng.choice(
        len(transformed),
        size=background_size,
        replace=False
    )

    explanation_indices = rng.choice(
        len(transformed),
        size=explanation_size,
        replace=False
    )

    background = transformed[background_indices]

    explanation_data = transformed[explanation_indices]

    # -------------------------------------------------
    # SHAP EXPLAINER
    # -------------------------------------------------

    print("\nCreating SHAP explainer...")

    explainer = shap.Explainer(
        classifier.predict_proba,
        background,
        algorithm="permutation",
        feature_names=feature_names,
        output_names=["Stay", "Churn"]
    )

    print("\nCalculating SHAP values...")
    print("Please wait. This may take several minutes.")

    explanation = explainer(
        explanation_data,
        max_evals=2 * transformed.shape[1] + 1
    )

    shap_values = np.asarray(
        explanation.values
    )

    # Select churn probability (class 1)
    if shap_values.ndim == 3:

        if shap_values.shape[2] < 2:
            raise ValueError(
                "SHAP output does not contain a churn class."
            )

        churn_values = shap_values[:, :, 1]

    elif shap_values.ndim == 2:

        churn_values = shap_values

    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    if churn_values.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP feature count does not match transformed features."
        )

    # -------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # -------------------------------------------------

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Mean_Absolute_SHAP": np.mean(
            np.abs(churn_values),
            axis=0
        ),

        "Mean_SHAP": np.mean(
            churn_values,
            axis=0
        )

    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        OUTPUT_DIR / "global_shap_importance.csv"
    )

    png_path = (
        OUTPUT_DIR / "global_shap_importance.png"
    )

    importance.to_csv(
        csv_path,
        index=False
    )

    # -------------------------------------------------
    # CREATE GLOBAL SHAP CHART
    # -------------------------------------------------

    top_features = importance.head(15)

    top_features = top_features.sort_values(
        "Mean_Absolute_SHAP"
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.barh(
        top_features["Feature"],
        top_features["Mean_Absolute_SHAP"]
    )

    ax.set_title(
        "Global SHAP Feature Importance",
        fontsize=15
    )

    ax.set_xlabel(
        "Mean Absolute SHAP Impact"
    )

    ax.set_ylabel(
        "Transformed Customer Features"
    )

    fig.tight_layout()

    fig.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------

    print("\n" + "=" * 55)
    print("GLOBAL SHAP ANALYSIS COMPLETED")
    print("=" * 55)

    print("\nTOP 10 INFLUENTIAL FEATURES")

    print(
        importance.head(10).to_string(index=False)
    )

    print("\nOUTPUT FILES")

    print(f"CSV: {csv_path}")

    print(f"Chart: {png_path}")

    print("\nIMPORTANT:")
    print(
        "Mean absolute SHAP impact measures the average "
        "magnitude of a feature's contribution to the "
        "model's churn-probability output."
    )

    print(
        "These explanations describe model behaviour; "
        "they do not establish real-world causation."
    )


# =====================================================
# 4. PROGRAM ENTRY POINT
# =====================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nSHAP ANALYSIS FAILED")

        print(f"Reason: {error}")

        sys.exit(1)
