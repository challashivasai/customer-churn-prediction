import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)

# --------------------------------------------------
# Step 1: Load the processed dataset
# --------------------------------------------------
df = pd.read_csv("data/processed_data.csv")

print("DATASET LOADED")
print("Dataset Shape:", df.shape)

if "Churn" not in df.columns:
    raise ValueError("The dataset must contain a target column named 'Churn'.")

X = df.drop("Churn", axis=1)
y = df["Churn"]

if set(pd.Series(y).dropna().unique()) != {0, 1}:
    raise ValueError("This script expects Churn labels to be encoded as 0 and 1.")

if y.isna().any() or X.isna().any().any():
    raise ValueError(
        "Missing values were found. Handle them during preprocessing before training."
    )

# --------------------------------------------------
# Step 2: Create train, validation and test sets
# 60% train, 20% validation, 20% test
# Validation is used to choose thresholds/model.
# Test is kept separate for final evaluation.
# --------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("\nTRAINING RECORDS:", len(X_train))
print("VALIDATION RECORDS:", len(X_validation))
print("TESTING RECORDS:", len(X_test))

# --------------------------------------------------
# Step 3: Define class-balanced models
# --------------------------------------------------
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
}

# --------------------------------------------------
# Step 4: Find the best probability threshold using
# validation data only. This avoids tuning on test data.
# The threshold is selected to maximize validation F1.
# --------------------------------------------------
def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5, 0.0

    denominator = precision[:-1] + recall[:-1]
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (denominator + 1e-12)
    )

    best_index = int(f1_values.argmax())
    return float(thresholds[best_index]), float(f1_values[best_index])


validation_results = {}
test_results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    threshold, validation_f1 = find_best_threshold(
        y_validation, validation_probabilities
    )

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    validation_auc = roc_auc_score(
        y_validation, validation_probabilities
    )

    validation_results[model_name] = {
        "threshold": threshold,
        "f1": validation_f1,
        "roc_auc": validation_auc,
    }

    # Evaluate on the untouched test set using the threshold
    # selected from validation data.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    test_results[model_name] = {
        "accuracy": accuracy_score(y_test, test_predictions),
        "roc_auc": roc_auc_score(y_test, test_probabilities),
        "f1": f1_score(y_test, test_predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, test_predictions),
        "classification_report": classification_report(
            y_test, test_predictions, zero_division=0
        ),
        "threshold": threshold,
    }

# --------------------------------------------------
# Step 5: Compare models using validation F1.
# Do not choose the winner using test-set performance.
# --------------------------------------------------
best_model_name = max(
    validation_results,
    key=lambda name: (
        validation_results[name]["f1"],
        validation_results[name]["roc_auc"],
    ),
)

best_model = models[best_model_name]
best_threshold = validation_results[best_model_name]["threshold"]

print("\n========== VALIDATION MODEL COMPARISON ==========")
for name, result in validation_results.items():
    print(f"\n{name}")
    print("Best threshold:", round(result["threshold"], 4))
    print("Validation F1:", round(result["f1"], 4))
    print("Validation ROC AUC:", round(result["roc_auc"], 4))

print("\n========== FINAL TEST RESULTS ==========")
for name, result in test_results.items():
    print(f"\n========== {name.upper()} ==========")
    print("Threshold:", round(result["threshold"], 4))
    print("Accuracy:", round(result["accuracy"], 4))
    print("ROC AUC:", round(result["roc_auc"], 4))
    print("F1-score:", round(result["f1"], 4))
    print("\nConfusion Matrix:")
    print(result["confusion_matrix"])
    print("\nClassification Report:")
    print(result["classification_report"])

# --------------------------------------------------
# Step 6: Save the selected model and its threshold
# --------------------------------------------------
os.makedirs("models", exist_ok=True)

model_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "threshold": best_threshold,
    "feature_columns": list(X.columns),
}

joblib.dump(model_bundle, "models/best_churn_model.joblib")

print("\n========== BEST MODEL SELECTED ==========")
print("Model:", best_model_name)
print("Threshold:", round(best_threshold, 4))
print("Saved to: models/best_churn_model.joblib")
print("\nMACHINE LEARNING TRAINING COMPLETED!")
