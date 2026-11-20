
import pandas as pd

FILE = "swarm_criticality_dataset_100_scenarios.csv"

print("Loading required columns...")

df = pd.read_csv(
    FILE,
    usecols=["case_id", "step", "agent_id", "is_critical"],
    low_memory=False
)

print("\nTotal rows:", len(df))

# Check duplicate agent records within each timestep
duplicate_mask = df.duplicated(
    subset=["case_id", "step", "agent_id"],
    keep=False
)

print("Rows involved in duplicate keys:", duplicate_mask.sum())

print(
    "Number of duplicated keys:",
    df.loc[duplicate_mask, ["case_id", "step", "agent_id"]]
      .drop_duplicates()
      .shape[0]
)

# Check whether duplicate keys have conflicting labels
duplicate_rows = df.loc[duplicate_mask]

conflicts = (
    duplicate_rows.groupby(["case_id", "step", "agent_id"])
    ["is_critical"]
    .nunique()
)

print("Duplicate keys with conflicting labels:", (conflicts > 1).sum())

print("\nAudit complete.")
