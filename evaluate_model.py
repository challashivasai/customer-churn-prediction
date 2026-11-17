
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# ==================================================
# STEP 1: LOAD EXISTING DATA AND TRAINED MODEL
# ==================================================

print("\nLOADING EXISTING MODEL...")

df = pd.read_csv("data/processed_data.csv")

model_bundle = joblib.load(
    "models/best_churn_model.joblib"
)

model = model_bundle["model"]
model_name = model_bundle["model_name"]
threshold = model_bundle["threshold"]
feature_columns = model_bundle["feature_columns"]

print("Model:", model_name)
print("Threshold:", threshold)

# ==================================================
# STEP 2: PREPARE THE SAME TEST DATA
# ==================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

missing_features = set(feature_columns) - set(X.columns)

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )

X = X[feature_columns]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTEST RECORDS:", len(X_test))

# ==================================================
# STEP 3: PREDICT USING SAVED MODEL
# ==================================================

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

# ==================================================
# STEP 4: CALCULATE PERFORMANCE METRICS
# ==================================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

roc_auc = roc_auc_score(
    y_test, probabilities
)

cm = confusion_matrix(
    y_test, predictions
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) else 0

print("\n========== MODEL EVALUATION ==========")

print("Model:", model_name)
print("Threshold:", round(threshold, 4))
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC AUC:", round(roc_auc, 4))
print("Specificity:", round(specificity, 4))

print("\nCONFUSION MATRIX")
print(cm)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# ==================================================
# STEP 5: CREATE OUTPUT DIRECTORY
# ==================================================

os.makedirs("evaluation_results", exist_ok=True)

# Save performance metrics

metrics = pd.DataFrame([{
    "Model": model_name,
    "Threshold": threshold,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC AUC": roc_auc,
    "Specificity": specificity
}])

metrics.to_csv(
    "evaluation_results/metrics.csv",
    index=False
)

# Save classification report

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    "evaluation_results/classification_report.csv"
)

# ==================================================
# STEP 6: CONFUSION MATRIX VISUALIZATION
# ==================================================

plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest")
plt.title("Customer Churn - Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

plt.tight_layout()

plt.savefig(
    "evaluation_results/confusion_matrix.png",
    dpi=300
)

plt.close()

# ==================================================
# STEP 7: ROC CURVE
# ==================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test, probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation_results/roc_curve.png",
    dpi=300
)

plt.close()

# ==================================================
# STEP 8: PRECISION-RECALL CURVE
# ==================================================

pr_precision, pr_recall, pr_thresholds = (
    precision_recall_curve(y_test, probabilities)
)

plt.figure(figsize=(7, 5))

plt.plot(
    pr_recall,
    pr_precision,
    label="Precision-Recall Curve"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation_results/precision_recall_curve.png",
    dpi=300
)

plt.close()

# ==================================================
# STEP 9: FEATURE IMPORTANCE
# ==================================================

estimator = (
    model.named_steps["model"]
    if hasattr(model, "named_steps")
    else model
)

if hasattr(estimator, "coef_"):
    importance = np.abs(estimator.coef_[0])
    feature_type = "Absolute Logistic Regression Coefficients"

elif hasattr(estimator, "feature_importances_"):
    importance = estimator.feature_importances_
    feature_type = "Random Forest Feature Importance"

else:
    importance = None

if importance is not None:

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        "evaluation_results/feature_importance.csv",
        index=False
    )

    top_features = importance_df.head(15).sort_values(
        by="Importance"
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Importance")
    plt.title(f"Top 15 Features - {feature_type}")

    plt.tight_layout()

    plt.savefig(
        "evaluation_results/feature_importance.png",
        dpi=300
    )

    plt.close()

# ==================================================
# STEP 10: COMPLETION
# ==================================================

print("\n===================================")
print("PROFESSIONAL MODEL EVALUATION DONE")
print("===================================")

print("\nSaved evaluation files:")

for filename in os.listdir("evaluation_results"):
    print(" -", filename)

print("\nOriginal trained model remains unchanged.")
