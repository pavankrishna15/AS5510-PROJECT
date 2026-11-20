# Critical Node Identification — Experimental Results

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

