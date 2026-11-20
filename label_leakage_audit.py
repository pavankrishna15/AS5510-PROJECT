
import pandas as pd
import numpy as np

FILE = "swarm_criticality_dataset_100_scenarios.csv"

print("Loading dataset...")
df = pd.read_csv(FILE, low_memory=False)

y = df["is_critical"].astype(bool).astype(int)
score = pd.to_numeric(df["criticality_score"], errors="coerce")

print("\n1. LABEL COUNTS")
print(y.value_counts().sort_index().rename(
    index={0: "Non-critical", 1: "Critical"}
))

print("\n2. CRITICALITY SCORE BY LABEL")
for label, name in [(0, "Non-critical"), (1, "Critical")]:
    values = score[y == label].dropna()
    print(f"\n{name}:")
    print(values.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))

print("\n3. TESTING SCORE THRESHOLDS")
print("Prediction rule: critical if criticality_score >= threshold")

for threshold in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    pred = (score >= threshold).astype(int)
    valid = score.notna()
    accuracy = (pred[valid] == y[valid]).mean()
    print(f"Threshold {threshold:.2f}: accuracy = {accuracy:.6f}")

print("\n4. BEST SCORE THRESHOLD")
valid = score.notna()
s = score[valid].to_numpy()
labels = y[valid].to_numpy()

thresholds = np.linspace(float(s.min()), float(s.max()), 1001)

best_accuracy = (-1, None)
best_balanced = (-1, None)

for threshold in thresholds:
    pred = (s >= threshold).astype(int)

    tp = np.sum((pred == 1) & (labels == 1))
    tn = np.sum((pred == 0) & (labels == 0))
    fp = np.sum((pred == 1) & (labels == 0))
    fn = np.sum((pred == 0) & (labels == 1))

    accuracy = (tp + tn) / len(labels)
    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    balanced = (sensitivity + specificity) / 2

    if accuracy > best_accuracy[0]:
        best_accuracy = (accuracy, threshold)

    if balanced > best_balanced[0]:
        best_balanced = (balanced, threshold)

print("Best accuracy:", best_accuracy)
print("Best balanced accuracy:", best_balanced)

print("\n5. SCORE-LABEL CORRELATION")
correlation = np.corrcoef(s, labels)[0, 1]
print("Pearson correlation:", correlation)

print("\n6. PROMOTION LABEL CROSS-TABULATION")
print(pd.crosstab(df["is_critical"], df["was_promoted"]))

print("\n7. LABEL CONSISTENCY WITHIN EACH UAV")
grouped = df.groupby(["case_id", "agent_id"])["is_critical"].nunique()
changing = grouped[grouped > 1]

print("Total scenario-UAV combinations:", len(grouped))
print("Combinations with changing labels:", len(changing))

print("\n8. FEATURES THAT NEED SOURCE-LOGIC REVIEW")
candidate_columns = [
    "criticality_score",
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "num_neighbors",
    "weighted_density",
    "angular_coverage",
    "was_promoted"
]

for col in candidate_columns:
    if col in df.columns:
        print(f"{col}: present")

print("\n9. FINAL DATA CHECK")
print("Missing criticality scores:", score.isna().sum())
print("Score minimum:", score.min())
print("Score maximum:", score.max())
print("Score outside [0, 1]:",
      ((score < 0) | (score > 1)).sum())

print("\nLABEL-LEAKAGE AUDIT COMPLETED.")
