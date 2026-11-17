
import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

print("BEFORE CLEANING")
print("Dataset shape:", df.shape)
print("Missing values:")
print(df.isnull().sum())

# Step 2: Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Step 3: Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Step 4: Check missing values again
print("\nAFTER CLEANING")
print(df.isnull().sum())

# Step 5: Separate numerical columns
numerical = df.select_dtypes(include=["int64", "float64"])

print("\nNUMERICAL COLUMNS")
print(numerical.columns.tolist())

# Step 6: Separate categorical columns
categorical = df.select_dtypes(include=["object"])

print("\nCATEGORICAL COLUMNS")
print(categorical.columns.tolist())

# Step 7: Save cleaned dataset
df.to_csv("data/cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
