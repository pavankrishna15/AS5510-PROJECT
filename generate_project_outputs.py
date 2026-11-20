import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("outputs", exist_ok=True)

# Test metrics
models = ["Random Forest", "GraphSAGE"]
metrics = {
    "Accuracy": [0.9136, 0.9348],
    "Precision": [0.9288, 0.9476],
    "Recall": [0.8616, 0.8951],
    "F1-score": [0.8939, 0.9206],
    "PR-AUC": [0.9672, 0.9793],
}

# 1. Model comparison
names = list(metrics.keys())
x = np.arange(len(names))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, [metrics[m][0] for m in names],
       width, label="Random Forest")
ax.bar(x + width/2, [metrics[m][1] for m in names],
       width, label="GraphSAGE")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Random Forest vs GraphSAGE — Test Performance")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("outputs/model_comparison.png", dpi=300)
plt.close(fig)

# 2. Random Forest confusion matrix
rf_cm = np.array([[136228, 6923], [14501, 90286]])

# 3. GraphSAGE confusion matrix
gs_cm = np.array([[46892, 1760], [3729, 31803]])

for name, cm in [
    ("random_forest_confusion_matrix", rf_cm),
    ("graphsage_confusion_matrix", gs_cm)
]:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_title(name.replace("_", " ").title())
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-critical", "Critical"])
    ax.set_yticklabels(["Non-critical", "Critical"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center")

    fig.tight_layout()
    fig.savefig(f"outputs/{name}.png", dpi=300)
    plt.close(fig)

# 4. Random Forest feature importance
features = [
    "min_neighbor_dist", "weighted_density", "accel",
    "rank_dist_to_centroid", "speed", "mean_neighbor_dist",
    "heading_rate", "angular_coverage", "rank_dist_to_target",
    "rel_z"
]
importance = [
    0.182011, 0.098800, 0.086825, 0.076186, 0.067054,
    0.065887, 0.057748, 0.038742, 0.037258, 0.036370
]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(features[::-1], importance[::-1])
ax.set_xlabel("Feature importance")
ax.set_title("Top 10 Random Forest Features")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig("outputs/feature_importance.png", dpi=300)
plt.close(fig)

# 5. Consolidated results summary
summary = """# Critical Node Identification — Experimental Results

## Dataset
- Dataset: swarm_criticality_dataset_100_scenarios.csv
- Usable observations: 1,639,166
- Scenarios: 100
- Features used: 26
- Duplicate agent-timestep observations removed: 0
- Missing values in the audited dataset: 0

## Experimental approach
A scenario-aware split was used in each experiment:
70 training scenarios, 15 validation scenarios and 15 test scenarios.

The Random Forest baseline used a stratified sample of 250,000
training observations. Its classification threshold was selected
using validation F1-score; the selected threshold was 0.40.

GraphSAGE used 3,000 training graphs, 1,000 validation graphs
and 2,000 test graphs. Graphs were constructed from agent
observations using a 55-unit sensing range. The model used
26 input features, two GraphSAGE layers (64 and 32 hidden
dimensions), dropout 0.3 and 25 training epochs.

## Test performance

| Metric | Random Forest | GraphSAGE |
|---|---:|---:|
| Accuracy | 91.36% | 93.48% |
| Precision | 92.88% | 94.76% |
| Recall | 86.16% | 89.51% |
| F1-score | 89.39% | 92.06% |
| PR-AUC | 96.72% | 97.93% |

## Confusion matrices

Class order: non-critical, critical.

Random Forest:
[[136228, 6923],
 [14501, 90286]]

GraphSAGE:
[[46892, 1760],
 [3729, 31803]]

## Feature importance
The three most important Random Forest features were:
1. min_neighbor_dist — 18.20%
2. weighted_density — 9.88%
3. accel — 8.68%

## Interpretation
GraphSAGE achieved higher reported test metrics than the
Random Forest baseline. This suggests that learning from
neighborhood graph structure may provide useful information
for identifying critical agents in the dataset.

## Limitations
- The models used independently generated scenario splits;
  their comparison is therefore indicative rather than a
  strictly controlled head-to-head benchmark.
- Random Forest used a 250,000-observation training sample.
- GraphSAGE used sampled graphs rather than every available graph.
- The target is the dataset-provided is_critical label. These
  classification results do not independently establish that
  removing a predicted agent causes loss of swarm connectivity.
- criticality_score and was_promoted were excluded from model
  inputs to reduce potential target leakage.
- The source code that generated the dataset labels was not
  available for verification.

"""
with open("outputs/project_results_summary.md", "w") as f:
    f.write(summary)

print("PROJECT OUTPUTS GENERATED")
for filename in sorted(os.listdir("outputs")):
    print("outputs/" + filename)
