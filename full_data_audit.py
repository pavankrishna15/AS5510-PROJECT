
import pandas as pd

FILE = "swarm_criticality_dataset_100_scenarios.csv"

print("Reading complete dataset...")

df = pd.read_csv(FILE, low_memory=False)

print("\n1. Dataset dimensions")
print(df.shape)

print("\n2. Column names")
print(df.columns.tolist())

print("\n3. Data types")
print(df.dtypes)

print("\n4. Missing values")
print(df.isnull().sum().sum())
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n5. Unique scenarios")
print(df["case_id"].nunique())

print("\n6. Unique UAVs")
print(df["agent_id"].nunique())

print("\n7. Critical label distribution")
print(df["is_critical"].value_counts(dropna=False))

print("\n8. Promotion label distribution")
print(df["was_promoted"].value_counts(dropna=False))

print("\n9. Scenario sizes")
print(df.groupby("case_id")["agent_id"].nunique().describe())

print("\nFull audit completed.")
