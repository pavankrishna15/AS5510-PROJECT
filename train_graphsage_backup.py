import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

FILE = "swarm_criticality_dataset_100_scenarios.csv"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Dataset defines neighbors within 55 units
SENSING_RANGE = 55.0

# Keep first experiment manageable
MAX_TRAIN_GRAPHS = 3000
MAX_VAL_GRAPHS = 1000
MAX_TEST_GRAPHS = 2000

EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Same 26 features used in RF baseline
FEATURES = [
    "num_agents",
    "n_obstacles",
    "obstacle_density",
    "mean_obstacle_size",
    "clutter_index",
    "obstacle_clustering",

    "rel_x",
    "rel_y",
    "rel_z",
    "dist_to_centroid",
    "rank_dist_to_centroid",
    "dist_to_target",
    "rank_dist_to_target",

    "vx",
    "vy",
    "vz",
    "speed",
    "heading",
    "flight_path_angle",
    "accel",
    "heading_rate",

    "num_neighbors",
    "mean_neighbor_dist",
    "min_neighbor_dist",
    "weighted_density",
    "angular_coverage",
]

USECOLS = [
    "case_id",
    "step",
    "agent_id",
    "is_critical",
    "attrition_rate",
] + FEATURES

# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("GRAPH SAGE — CRITICAL NODE IDENTIFICATION")
print("=" * 70)
print("Device:", DEVICE)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

chunks = []

for chunk in pd.read_csv(
    FILE,
    usecols=USECOLS,
    chunksize=100_000,
    on_bad_lines="skip",
    low_memory=False,
):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

print("Loaded rows:", len(df))
print("Scenarios:", df["case_id"].nunique())

# Convert numeric columns
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["is_critical"] = df["is_critical"].astype(int)

df = df.dropna(subset=FEATURES + ["is_critical"])

# Remove duplicate agent observations
before = len(df)

df = df.drop_duplicates(
    subset=["case_id", "step", "agent_id"]
)

print("Duplicates removed:", before - len(df))
print("Usable rows:", len(df))

# ============================================================
# SCENARIO SPLIT
# ============================================================

scenarios = df["case_id"].unique()

train_cases, temp_cases = train_test_split(
    scenarios,
    test_size=0.30,
    random_state=RANDOM_STATE,
)

val_cases, test_cases = train_test_split(
    temp_cases,
    test_size=0.50,
    random_state=RANDOM_STATE,
)

train_cases = set(train_cases)
val_cases = set(val_cases)
test_cases = set(test_cases)

print("\nScenario split:")
print("Train:", len(train_cases))
print("Validation:", len(val_cases))
print("Test:", len(test_cases))

train_df = df[df["case_id"].isin(train_cases)].copy()
val_df = df[df["case_id"].isin(val_cases)].copy()
test_df = df[df["case_id"].isin(test_cases)].copy()

print("\nRows:")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

# ============================================================
# FEATURE SCALING
# ============================================================

print("\nFitting feature scaler...")

scaler = StandardScaler()

# Fit only on training data
scaler.fit(train_df[FEATURES])

train_df[FEATURES] = scaler.transform(train_df[FEATURES])
val_df[FEATURES] = scaler.transform(val_df[FEATURES])
test_df[FEATURES] = scaler.transform(test_df[FEATURES])

# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graphs(dataframe, max_graphs=None, seed=42):

    print("\nBuilding graphs...")

    groups = list(
        dataframe.groupby(["case_id", "step"], sort=False)
    )

    # Randomly sample graph instances for quick experiment
    if max_graphs is not None and len(groups) > max_graphs:

        rng = random.Random(seed)
        groups = rng.sample(groups, max_graphs)

    graphs = []

    for graph_number, ((case_id, step), group) in enumerate(groups):

        # ----------------------------------------------------
        # Node features
        # ----------------------------------------------------

        x = torch.tensor(
            group[FEATURES].values,
            dtype=torch.float32
        )

        y = torch.tensor(
            group["is_critical"].values,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Reconstruct connectivity graph
        #
        # Relative positions:
        # rel_x, rel_y, rel_z
        #
        # Edge exists when distance <= 55 units
        # ----------------------------------------------------

        positions = group[
            ["rel_x", "rel_y", "rel_z"]
        ].values.astype(np.float32)

        n = len(group)

        edge_pairs = []

        for i in range(n):

            # Vectorized distance calculation
            diff = positions - positions[i]

            distances = np.sqrt(
                np.sum(diff * diff, axis=1)
            )

            neighbors = np.where(
                (distances <= SENSING_RANGE)
                & (distances > 0)
            )[0]

            for j in neighbors:
                edge_pairs.append([i, j])

        if len(edge_pairs) == 0:
            # Isolated graph — skip
            continue

        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long
        ).t().contiguous()

        graph = Data(
            x=x,
            edge_index=edge_index,
            y=y,
        )

        graph.case_id = case_id
        graph.step = int(step)
        graph.num_agents_meta = float(group["num_agents"].iloc[0])
        graph.obstacle_density_meta = float(group["obstacle_density"].iloc[0])
        graph.attrition_rate_meta = float(group["attrition_rate"].iloc[0])

        graphs.append(graph)

        if (graph_number + 1) % 500 == 0:
            print(
                f"  Built {graph_number + 1}/{len(groups)} graphs"
            )

    print("Graphs created:", len(graphs))

    return graphs


train_graphs = build_graphs(
    train_df,
    MAX_TRAIN_GRAPHS,
    RANDOM_STATE
)

val_graphs = build_graphs(
    val_df,
    MAX_VAL_GRAPHS,
    RANDOM_STATE + 1
)

test_graphs = build_graphs(
    test_df,
    MAX_TEST_GRAPHS,
    RANDOM_STATE + 2
)

# ============================================================
# GRAPH STATISTICS
# ============================================================

def graph_statistics(graphs, name):

    nodes = sum(g.num_nodes for g in graphs)
    edges = sum(g.num_edges for g in graphs)

    critical = sum(
        int(g.y.sum())
        for g in graphs
    )

    total = sum(
        int(g.y.numel())
        for g in graphs
    )

    print(f"\n{name} graph statistics:")
    print("Graphs:", len(graphs))
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Critical nodes:", critical)
    print("Critical rate:", round(critical / total, 4))


graph_statistics(train_graphs, "TRAIN")
graph_statistics(val_graphs, "VALIDATION")
graph_statistics(test_graphs, "TEST")

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ============================================================
# GRAPHSAGE MODEL
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            64
        )

        self.conv2 = SAGEConv(
            64,
            32
        )

        self.classifier = nn.Linear(
            32,
            2
        )

        self.dropout = nn.Dropout(0.30)

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.classifier(x)

        return x


model = GraphSAGE(
    input_dim=len(FEATURES)
).to(DEVICE)

print("\nModel:")
print(model)

# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = torch.cat(
    [g.y for g in train_graphs]
)

class_counts = torch.bincount(
    train_labels,
    minlength=2
).float()

class_weights = (
    class_counts.sum() / (2 * class_counts)
).to(DEVICE)

print("\nClass counts:")
print("Non-critical:", int(class_counts[0]))
print("Critical:", int(class_counts[1]))

print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(DEVICE)

            logits = model(
                batch.x,
                batch.edge_index
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            predictions = (
                probabilities >= 0.5
            ).long()

            all_labels.extend(
                batch.y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_prob
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred
        ),
    }


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING GRAPH SAGE")
print("=" * 70)

best_val_f1 = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    total_nodes = 0

    for batch in train_loader:

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            batch.x,
            batch.edge_index
        )

        loss = criterion(
            logits,
            batch.y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * batch.num_nodes
        )

        total_nodes += batch.num_nodes

    avg_loss = total_loss / total_nodes

    val_results = evaluate(val_loader)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val F1: {val_results['f1']:.4f} | "
        f"Val PR-AUC: {val_results['pr_auc']:.4f}"
    )

    if val_results["f1"] > best_val_f1:

        best_val_f1 = val_results["f1"]

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

# ============================================================
# LOAD BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )

    model.to(DEVICE)

# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL GRAPH SAGE TEST RESULTS")
print("=" * 70)

results = evaluate(test_loader)

# ============================================================
# TOP-K RANKING AND ROBUSTNESS EVALUATION
# ============================================================

print("\\nRunning Top-K and robustness evaluation...")

node_records = []
graph_records = []

model.eval()

with torch.no_grad():
    for graph_idx, graph in enumerate(test_graphs):
        x = graph.x.to(DEVICE)
        edge_index = graph.edge_index.to(DEVICE)

        logits = model(x, edge_index)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        probabilities = probabilities.cpu().numpy()
        labels = graph.y.cpu().numpy()

        graph_records.append({
            "graph_idx": graph_idx,
            "case_id": graph.case_id,
            "step": graph.step,
            "num_agents": graph.num_agents_meta,
            "obstacle_density": graph.obstacle_density_meta,
            "attrition_rate": graph.attrition_rate_meta,
            "nodes": len(labels)
        })

        for i in range(len(labels)):
            node_records.append({
                "graph_idx": graph_idx,
                "y_true": int(labels[i]),
                "probability": float(probabilities[i])
            })

graph_df = pd.DataFrame(graph_records)
node_df = pd.DataFrame(node_records)

# Top-K: rank UAVs separately within each swarm snapshot
topk_rows = []

for graph_idx, group in node_df.groupby("graph_idx"):
    group = group.sort_values("probability", ascending=False)
    total_critical = int(group["y_true"].sum())

    for k in [5, 10, 20]:
        actual_k = min(k, len(group))
        selected = group.head(actual_k)
        hits = int(selected["y_true"].sum())

        topk_rows.append({
            "graph_idx": graph_idx,
            "K": k,
            "actual_k": actual_k,
            "critical_found": hits,
            "precision_at_k": hits / actual_k,
            "recall_at_k": hits / total_critical
                if total_critical > 0 else np.nan
        })

topk_df = pd.DataFrame(topk_rows)

topk_summary = (
    topk_df.groupby("K")
    .agg(
        graphs=("graph_idx", "count"),
        mean_precision_at_k=("precision_at_k", "mean"),
        mean_recall_at_k=("recall_at_k", "mean")
    )
    .reset_index()
)

topk_df.to_csv("topk_ranking_per_graph.csv", index=False)
topk_summary.to_csv("topk_ranking_results.csv", index=False)

print("\\nTOP-K RANKING RESULTS")
print(topk_summary.to_string(index=False))

# Robustness: evaluate pooled node predictions within
# low, medium and high scenario-attribute groups.
robustness_outputs = []

for column, label in [
    ("num_agents", "Swarm size"),
    ("obstacle_density", "Obstacle density"),
    ("attrition_rate", "Attrition rate")
]:
    unique_values = graph_df[column].nunique()

    if unique_values < 2:
        print(f"Skipping {label}: insufficient variation")
        continue

    bins = min(3, unique_values)

    graph_df["_group"] = pd.qcut(
        graph_df[column].rank(method="first"),
        q=bins,
        labels=False
    )

    node_df = node_df.merge(
        graph_df[["graph_idx", "_group"]],
        on="graph_idx",
        how="left"
    )

    for group_number, subset in node_df.groupby("_group"):
        y = subset["y_true"].to_numpy()
        pred = (subset["probability"].to_numpy() >= 0.5).astype(int)

        robustness_outputs.append({
            "factor": label,
            "group": int(group_number) + 1,
            "graphs": subset["graph_idx"].nunique(),
            "nodes": len(subset),
            "f1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "critical_rate": float(y.mean())
        })

    node_df.drop(columns=["_group"], inplace=True)
    graph_df.drop(columns=["_group"], inplace=True)

robustness_df = pd.DataFrame(robustness_outputs)
robustness_df.to_csv("robustness_results.csv", index=False)

print("\\nROBUSTNESS RESULTS")
print(robustness_df.to_string(index=False))

with open("additional_analysis_summary.txt", "w") as f:
    f.write("TOP-K RANKING EVALUATION\\n")
    f.write(topk_summary.to_string(index=False))
    f.write("\\n\\nROBUSTNESS EVALUATION\\n")
    f.write(robustness_df.to_string(index=False))

print("\\nAdditional evaluation files saved.")

print(
    f"Accuracy : {results['accuracy']:.4f}"
)

print(
    f"Precision: {results['precision']:.4f}"
)

print(
    f"Recall   : {results['recall']:.4f}"
)

print(
    f"F1       : {results['f1']:.4f}"
)

print(
    f"PR-AUC   : {results['pr_auc']:.4f}"
)

print("\nConfusion Matrix:")
print(results["confusion_matrix"])

# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "graphsage_results.txt",
    "w"
) as f:

    f.write(
        "GRAPH SAGE CRITICAL NODE IDENTIFICATION\n"
    )

    f.write("=" * 60 + "\n")

    f.write(
        f"Device: {DEVICE}\n"
    )

    f.write(
        f"Train graphs: {len(train_graphs)}\n"
    )

    f.write(
        f"Validation graphs: {len(val_graphs)}\n"
    )

    f.write(
        f"Test graphs: {len(test_graphs)}\n"
    )

    f.write(
        f"Sensing range: {SENSING_RANGE}\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        f"Best validation F1: {best_val_f1:.4f}\n\n"
    )

    f.write(
        f"Test Accuracy: {results['accuracy']:.4f}\n"
    )

    f.write(
        f"Test Precision: {results['precision']:.4f}\n"
    )

    f.write(
        f"Test Recall: {results['recall']:.4f}\n"
    )

    f.write(
        f"Test F1: {results['f1']:.4f}\n"
    )

    f.write(
        f"Test PR-AUC: {results['pr_auc']:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(results["confusion_matrix"])
    )

print(
    "\nResults saved to graphsage_results.txt"
)

print("\nGraphSAGE experiment complete.")