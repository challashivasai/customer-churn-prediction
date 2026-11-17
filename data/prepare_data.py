
from pathlib import Path

import pandas as pd

RAW_FILE = Path("data/raw/Telco-Customer-Churn.csv")
OUTPUT_FILE = Path("data/cleaned_data.csv")


def prepare_data():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    required_columns = {
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    original_rows = len(df)

    # Convert numeric columns safely.
    df["tenure"] = pd.to_numeric(df["tenure"], errors="raise")
    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"], errors="raise"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True),
        errors="coerce",
    )

    # Blank charges are valid only for zero-tenure customers.
    blank_charges = df["TotalCharges"].isna()
    invalid_blank_rows = blank_charges & (df["tenure"] != 0)

    if invalid_blank_rows.any():
        raise ValueError(
            "Found blank TotalCharges for customers with non-zero tenure."
        )

    df.loc[blank_charges, "TotalCharges"] = 0.0

    # Validate target labels.
    if not set(df["Churn"].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("Unexpected values found in Churn column.")

    if df["Churn"].isna().any():
        raise ValueError("Churn contains missing values.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    if df.isna().any().any():
        raise ValueError("Missing values remain after cleaning.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("DATA PREPARATION SUCCESSFUL")
    print("Original rows:", original_rows)
    print("Cleaned rows:", len(df))
    print("Blank charges corrected:", int(blank_charges.sum()))
    print("Remaining missing values:", int(df.isna().sum().sum()))
    print("Output:", OUTPUT_FILE.resolve())
    print("\nChurn distribution:")
    print(df["Churn"].value_counts())


if __name__ == "__main__":
    prepare_data()
