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
# CONFIG
# ============================================================

FILE = "swarm_criticality_dataset_100_scenarios.csv"

RANDOM_STATE = 42
SENSING_RANGE = 55.0

EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001

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
] + FEATURES

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

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
print("FULL-SCALE GRAPHSAGE")
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

print("Rows loaded:", len(df))
print("Scenarios:", df["case_id"].nunique())

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["is_critical"] = df["is_critical"].astype(int)

df = df.dropna(
    subset=FEATURES + ["is_critical"]
)

df = df.drop_duplicates(
    subset=["case_id", "step", "agent_id"]
)

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

train_df = df[
    df["case_id"].isin(train_cases)
].copy()

val_df = df[
    df["case_id"].isin(val_cases)
].copy()

test_df = df[
    df["case_id"].isin(test_cases)
].copy()

print("\nScenario split:")
print("Train scenarios:", len(train_cases))
print("Validation scenarios:", len(val_cases))
print("Test scenarios:", len(test_cases))

print("\nRows:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

scaler.fit(train_df[FEATURES])

train_df[FEATURES] = scaler.transform(
    train_df[FEATURES]
).astype(np.float32)

val_df[FEATURES] = scaler.transform(
    val_df[FEATURES]
).astype(np.float32)

test_df[FEATURES] = scaler.transform(
    test_df[FEATURES]
).astype(np.float32)

# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graphs(dataframe, name):

    print("\n" + "-" * 60)
    print("Building", name, "graphs")
    print("-" * 60)

    groups = dataframe.groupby(
        ["case_id", "step"],
        sort=False
    )

    graphs = []

    total_groups = len(groups)

    for graph_no, ((case_id, step), group) in enumerate(
        groups,
        start=1
    ):

        positions = group[
            ["rel_x", "rel_y", "rel_z"]
        ].values.astype(np.float32)

        n = len(group)

        # Node features
        x = torch.tensor(
            group[FEATURES].values,
            dtype=torch.float32
        )

        # Node labels
        y = torch.tensor(
            group["is_critical"].values,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Build edges using 55-unit sensing range
        # ----------------------------------------------------

        edge_pairs = []

        for i in range(n):

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

        if not edge_pairs:
            continue

        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long
        ).t().contiguous()

        graph = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        graph.case_id = case_id
        graph.step = int(step)

        graphs.append(graph)

        if graph_no % 2000 == 0:
            print(
                f"{name}: {graph_no}/{total_groups} graphs"
            )

    print(
        f"{name} graphs created: {len(graphs)}"
    )

    return graphs


train_graphs = build_graphs(
    train_df,
    "TRAIN"
)

val_graphs = build_graphs(
    val_df,
    "VALIDATION"
)

test_graphs = build_graphs(
    test_df,
    "TEST"
)

# ============================================================
# GRAPH STATISTICS
# ============================================================

def statistics(graphs, name):

    nodes = sum(
        g.num_nodes
        for g in graphs
    )

    edges = sum(
        g.num_edges
        for g in graphs
    )

    critical = sum(
        int(g.y.sum())
        for g in graphs
    )

    total = sum(
        g.y.numel()
        for g in graphs
    )

    print("\n", name)

    print("Graphs:", len(graphs))
    print("Nodes:", nodes)
    print("Edges:", edges)
    print(
        "Critical rate:",
        round(critical / total, 4)
    )


statistics(train_graphs, "TRAIN STATISTICS")
statistics(val_graphs, "VALIDATION STATISTICS")
statistics(test_graphs, "TEST STATISTICS")

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# GRAPHSAGE
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

        self.dropout = nn.Dropout(
            0.30
        )

        self.classifier = nn.Linear(
            32,
            2
        )

    def forward(
        self,
        x,
        edge_index
    ):

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

        return self.classifier(x)


model = GraphSAGE(
    len(FEATURES)
).to(DEVICE)

print("\nModel:")
print(model)

# ============================================================
# CLASS WEIGHTS
# ============================================================

labels = torch.cat(
    [g.y for g in train_graphs]
)

counts = torch.bincount(
    labels,
    minlength=2
).float()

weights = (
    counts.sum()
    /
    (2 * counts)
)

weights = weights.to(DEVICE)

print("\nClass counts:")
print("Non-critical:", int(counts[0]))
print("Critical:", int(counts[1]))

print("Class weights:", weights)

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

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

            y_true.extend(
                batch.y.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

            y_prob.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
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
        "cm": confusion_matrix(
            y_true,
            y_pred
        )
    }


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

best_f1 = 0
best_state = None

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0
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
            loss.item()
            *
            batch.num_nodes
        )

        total_nodes += batch.num_nodes

    avg_loss = (
        total_loss /
        total_nodes
    )

    val = evaluate(
        val_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss {avg_loss:.4f} | "
        f"Val F1 {val['f1']:.4f} | "
        f"Val PR-AUC {val['pr_auc']:.4f}"
    )

    if val["f1"] > best_f1:

        best_f1 = val["f1"]

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

# ============================================================
# BEST MODEL
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
print("FULL GRAPHSAGE TEST RESULTS")
print("=" * 70)

results = evaluate(
    test_loader
)

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
print(results["cm"])

# ============================================================
# SAVE
# ============================================================

with open(
    "graphsage_full_results.txt",
    "w"
) as f:

    f.write(
        "FULL-SCALE GRAPHSAGE RESULTS\n"
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
        f"Best validation F1: {best_f1:.4f}\n\n"
    )

    f.write(
        f"Accuracy: {results['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: {results['precision']:.4f}\n"
    )

    f.write(
        f"Recall: {results['recall']:.4f}\n"
    )

    f.write(
        f"F1: {results['f1']:.4f}\n"
    )

    f.write(
        f"PR-AUC: {results['pr_auc']:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(results["cm"])
    )

print(
    "\nSaved: graphsage_full_results.txt"
)

print("\nFULL GRAPHSAGE COMPLETE")
