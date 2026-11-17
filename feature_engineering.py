
import pandas as pd

# Step 1: Load cleaned dataset
df = pd.read_csv("data/cleaned_data.csv")

print("Original Dataset Shape:")
print(df.shape)

# Step 2: Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 Months", "13-24 Months",
            "25-48 Months", "49-72 Months"],
    include_lowest=True
)

# Step 3: Convert Churn into numerical values
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Step 4: Remove customer ID
df.drop("customerID", axis=1, inplace=True)

# Step 5: Encode categorical columns
df = pd.get_dummies(
    df,
    drop_first=True,
    dtype=int
)

# Step 6: Check final dataset
print("\nFEATURE ENGINEERING COMPLETE")
print("New Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

# Step 7: Save processed dataset
df.to_csv("data/processed_data.csv", index=False)

print("\nProcessed dataset saved successfully!")
