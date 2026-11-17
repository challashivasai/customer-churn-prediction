
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Step 2: Display first 5 rows
print("\n--- FIRST 5 CUSTOMERS ---")
print(df.head())

# Step 3: Display dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Step 4: Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Step 5: Dataset information
print("\n--- DATASET INFORMATION ---")
df.info()

# Step 6: Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Step 7: Check customer churn distribution
print("\n--- CUSTOMER CHURN ---")
print(df["Churn"].value_counts())
# Step 8: Check data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

# Step 9: Check blank TotalCharges
print("\n--- BLANK TOTAL CHARGES ---")
print((df["TotalCharges"].str.strip() == "").sum())

# Step 10: Check duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())