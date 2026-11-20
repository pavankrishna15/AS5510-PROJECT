
import pandas as pd
import os

FILE = "swarm_criticality_dataset_100_scenarios.csv"

print("File exists:", os.path.exists(FILE))
print("File size (GB):", round(os.path.getsize(FILE) / (1024**3), 3))

# Read only the target column first
target = pd.read_csv(
    FILE,
    usecols=["is_critical"],
    low_memory=False
)

print("\nTotal rows:", len(target))
print("\nClass distribution:")
print(target["is_critical"].value_counts(dropna=False))

print("\nClass percentages:")
print(
    target["is_critical"]
    .value_counts(normalize=True, dropna=False) * 100
)

print("\nData types:")
print(target["is_critical"].dtype)
