
import pandas as pd

FILE = "swarm_criticality_dataset_100_scenarios.csv"

df = pd.read_csv(
    FILE,
    usecols=[
        "case_id",
        "step",
        "agent_id",
        "is_critical",
        "was_promoted",
        "attrition_rate",
        "num_agents",
        "n_obstacles"
    ]
)

print("\n1. LABEL CROSS-TABULATION")
print(pd.crosstab(df["is_critical"], df["was_promoted"]))

print("\n2. UNIQUE UAV LABEL CONSISTENCY")

agent_labels = (
    df.groupby(["case_id", "agent_id"])["is_critical"]
    .nunique()
)

print("Total unique scenario-UAV combinations:", len(agent_labels))
print("UAVs with changing criticality labels:", (agent_labels > 1).sum())

print("\n3. SCENARIO-WISE CRITICALITY")

scenario_stats = (
    df.groupby("case_id")["is_critical"]
    .agg(["count", "sum", "mean"])
)

scenario_stats["critical_percentage"] = scenario_stats["mean"] * 100

print(scenario_stats.describe())
print("\nLowest criticality scenarios:")
print(scenario_stats.nsmallest(5, "critical_percentage"))

print("\nHighest criticality scenarios:")
print(scenario_stats.nlargest(5, "critical_percentage"))

print("\n4. ATTRITION-WISE DISTRIBUTION")
print(
    df.groupby("attrition_rate")["is_critical"]
    .agg(["count", "sum", "mean"])
)

print("\n5. OBSTACLE-WISE DISTRIBUTION")
print(
    df.groupby("n_obstacles")["is_critical"]
    .agg(["count", "sum", "mean"])
)

print("\nLabel analysis complete.")
