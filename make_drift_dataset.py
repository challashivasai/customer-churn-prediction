
import pandas as pd

# Change this to your original CSV filename
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Create a separate comparison dataset
comparison = df.copy()

# 1. Reduce customer tenure
comparison["tenure"] = (
    comparison["tenure"] * 0.7
).round().clip(0, 72)

# 2. Increase monthly charges
comparison["MonthlyCharges"] = (
    comparison["MonthlyCharges"] * 1.15
).clip(0, 120)

# 3. Increase month-to-month contracts
if "Contract" in comparison.columns:
    comparison["Contract"] = "Month-to-month"

# Save the modified dataset
comparison.to_csv(
    "telco_comparison_drift.csv",
    index=False
)

print("Comparison dataset created successfully!")
print("Rows:", len(comparison))
print("Saved: telco_comparison_drift.csv")
