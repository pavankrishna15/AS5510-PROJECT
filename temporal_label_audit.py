
import pandas as pd

FILE = "swarm_criticality_dataset_100_scenarios.csv"

df = pd.read_csv(
    FILE,
    usecols=[
        "case_id",
        "step",
        "agent_id",
        "is_critical",
        "criticality_score"
    ]
)

# Find UAVs whose labels change over time
summary = (
    df.groupby(["case_id", "agent_id"])
    .agg(
        total_observations=("is_critical", "size"),
        critical_observations=("is_critical", "sum"),
        distinct_labels=("is_critical", "nunique"),
        first_step=("step", "min"),
        last_step=("step", "max")
    )
)

changing = summary[summary["distinct_labels"] > 1]

print("Changing UAVs:", len(changing))
print("\nChanging UAV details:")
print(changing.to_string())

print("\nSample of changing UAV timelines:")

for case_id, agent_id in changing.index[:5]:
    sample = df[
        (df["case_id"] == case_id) &
        (df["agent_id"] == agent_id)
    ].sort_values("step")

    print(f"\nScenario: {case_id}, UAV: {agent_id}")
    print(sample.to_string(index=False))

print("\nTemporal audit completed.")

