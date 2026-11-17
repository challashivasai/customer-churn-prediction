
import pandas as pd
from retention_engine import generate_retention_plan

# Simulated SHAP results for one customer
shap_details = pd.DataFrame({
    "Feature": [
        "tenure",
        "MonthlyCharges",
        "Contract_Two year",
        "PaymentMethod_Electronic check",
        "TotalCharges"
    ],
    "Customer value": [12, 85.5, 0, 1, 1026],
    "SHAP impact": [0.82, 0.35, -0.41, 0.29, -0.21]
})

# Test prediction
churn_probability = 0.70
threshold = 0.6176

# Generate retention plan
result = generate_retention_plan(
    churn_probability,
    threshold,
    shap_details
)

# Display results
print("\n===== RETENTION ENGINE TEST =====")

print("Risk Level:", result["risk_level"])
print("Churn Probability:", result["churn_probability"])
print("Decision Threshold:", result["decision_threshold"])

print("\nSummary:")
print(result["summary"])

print("\nFactors Increasing Churn:")
print(result["increasing_factors"])

print("\nFactors Reducing Churn:")
print(result["reducing_factors"])

print("\nRecommended Actions:")
for action in result["recommendations"]:
    print("-", action["Recommended action"])

print("\nRETENTION ENGINE TEST COMPLETED!")
