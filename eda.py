
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_data.csv")

# Step 1: Dataset overview
print("DATASET OVERVIEW")
print(df.shape)

# Step 2: Customer churn count
print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())

# Step 3: Churn percentage
print("\nCHURN PERCENTAGE")
print(df["Churn"].value_counts(normalize=True) * 100)

# Step 4: Churn visualization
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

# Step 5: Contract vs Churn
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Customer Churn")
plt.xticks(rotation=15)
plt.show()

# Step 6: Monthly Charges vs Churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.show()

# Step 7: Contract-wise churn percentage

print("\nCONTRACT-WISE CHURN PERCENTAGE")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn)

# Step 8: Visualize contract churn

contract_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.legend(title="Churn")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 9: Customer tenure analysis

print("\nTENURE VS CHURN")

tenure_churn = df.groupby("Churn")["tenure"].mean()

print(tenure_churn)

# Step 10: Visualize tenure

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Tenure (Months)")
plt.show()


# Step 11: Monthly charges analysis

print("\nMONTHLY CHARGES VS CHURN")

print(df.groupby("Churn")["MonthlyCharges"].mean())


# Step 12: Visualize monthly charges

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")
plt.show()

