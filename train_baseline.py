import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# SETTINGS
# =========================================================

FILE = "swarm_criticality_dataset_100_scenarios.csv"

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 250000
CHUNK_SIZE = 100000


# =========================================================
# FEATURES
# =========================================================

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
    "angular_coverage"
]


USECOLS = [
    "case_id",
    "step",
    "agent_id",
    "attrition_rate",
    "is_critical"
] + FEATURES


# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

parts = []
total_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        FILE,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE,
        on_bad_lines="skip",
        low_memory=False
    ),
    start=1
):

    # Remove rows containing missing values
    chunk = chunk.dropna()

    # Convert target
    if chunk["is_critical"].dtype == object:
        chunk["is_critical"] = (
            chunk["is_critical"]
            .astype(str)
            .str.lower()
            .map({
                "true": 1,
                "false": 0
            })
        )

    chunk["is_critical"] = pd.to_numeric(
        chunk["is_critical"],
        errors="coerce"
    )

    # Convert feature columns to numeric
    for col in FEATURES:
        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce"
        )

    # Remove rows that became invalid
    chunk = chunk.dropna(
        subset=FEATURES + ["is_critical"]
    )

    # Ensure target is integer
    chunk["is_critical"] = chunk["is_critical"].astype(int)

    parts.append(chunk)

    total_rows += len(chunk)

    print(
        f"Chunk {chunk_number:02d} | "
        f"Rows retained so far: {total_rows:,}"
    )


# =========================================================
# COMBINE DATA
# =========================================================

print("\nCombining chunks...")

if len(parts) == 0:
    raise RuntimeError(
        "No valid rows were loaded. "
        "Check FILE path and CSV format."
    )

df = pd.concat(
    parts,
    ignore_index=True
)

print(f"Total usable rows: {len(df):,}")
print(f"Unique scenarios: {df['case_id'].nunique()}")


# =========================================================
# REMOVE DUPLICATE AGENT-TIMESTEP OBSERVATIONS
# =========================================================

print("\nRemoving duplicate agent-timestep observations...")

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=[
        "case_id",
        "step",
        "agent_id"
    ],
    keep="first"
)

after_duplicates = len(df)

print(
    f"Removed duplicates: "
    f"{before_duplicates - after_duplicates:,}"
)

print(
    f"Rows remaining: "
    f"{after_duplicates:,}"
)


# =========================================================
# SCENARIO-AWARE TRAIN / VALIDATION / TEST SPLIT
# =========================================================

print("\n" + "=" * 60)
print("SCENARIO-AWARE SPLIT")
print("=" * 60)

# IMPORTANT:
# We split using complete scenarios rather than individual rows.
# This prevents the same scenario appearing in both train and test.

all_cases = sorted(
    df["case_id"].unique()
)

print(f"Total scenarios: {len(all_cases)}")


# 70% training, 30% temporary
train_cases, temp_cases = train_test_split(
    all_cases,
    test_size=0.30,
    random_state=RANDOM_STATE
)


# Split remaining 30% equally:
# 15% validation
# 15% test

val_cases, test_cases = train_test_split(
    temp_cases,
    test_size=0.50,
    random_state=RANDOM_STATE
)


train_cases = set(train_cases)
val_cases = set(val_cases)
test_cases = set(test_cases)


print(f"Training scenarios:   {len(train_cases)}")
print(f"Validation scenarios: {len(val_cases)}")
print(f"Test scenarios:       {len(test_cases)}")


# =========================================================
# CREATE DATAFRAMES
# =========================================================

train_df = df[
    df["case_id"].isin(train_cases)
].copy()

val_df = df[
    df["case_id"].isin(val_cases)
].copy()

test_df = df[
    df["case_id"].isin(test_cases)
].copy()


print("\nRows in each split:")

print(
    f"Train:      {len(train_df):,}"
)

print(
    f"Validation: {len(val_df):,}"
)

print(
    f"Test:       {len(test_df):,}"
)


# =========================================================
# CHECK TARGET DISTRIBUTION
# =========================================================

print("\nTarget distribution:")

print(
    "\nTRAIN:"
)

print(
    train_df["is_critical"]
    .value_counts(normalize=True)
    .sort_index()
)


print(
    "\nVALIDATION:"
)

print(
    val_df["is_critical"]
    .value_counts(normalize=True)
    .sort_index()
)


print(
    "\nTEST:"
)

print(
    test_df["is_critical"]
    .value_counts(normalize=True)
    .sort_index()
)


# =========================================================
# SAMPLE TRAINING DATA
# =========================================================

print("\n" + "=" * 60)
print("PREPARING MODEL DATA")
print("=" * 60)

if len(train_df) > TRAIN_SAMPLE_SIZE:

    print(
        f"Training set is {len(train_df):,} rows."
    )

    print(
        f"Sampling {TRAIN_SAMPLE_SIZE:,} rows "
        f"for the first experiment..."
    )

    # Stratified sampling using sklearn
    train_df, _ = train_test_split(
        train_df,
        train_size=TRAIN_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["is_critical"]
    )

else:

    print(
        f"Using all {len(train_df):,} training rows."
    )


# =========================================================
# X / Y
# =========================================================

X_train = train_df[FEATURES]
y_train = train_df["is_critical"]

X_val = val_df[FEATURES]
y_val = val_df["is_critical"]

X_test = test_df[FEATURES]
y_test = test_df["is_critical"]


print("\nFeature count:", len(FEATURES))

print(
    "Training matrix:",
    X_train.shape
)

print(
    "Validation matrix:",
    X_val.shape
)

print(
    "Test matrix:",
    X_test.shape
)


# =========================================================
# RANDOM FOREST BASELINE
# =========================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST BASELINE")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)


print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

print("\nGenerating validation probabilities...")

val_prob = model.predict_proba(
    X_val
)[:, 1]


# =========================================================
# THRESHOLD TUNING
# =========================================================

print("\n" + "=" * 60)
print("TUNING CLASSIFICATION THRESHOLD")
print("=" * 60)

best_threshold = 0.50
best_f1 = 0.0

threshold_results = []

for threshold in np.arange(
    0.20,
    0.81,
    0.05
):

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    threshold_results.append(
        (
            threshold,
            precision,
            recall,
            f1
        )
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print(
    f"\nBest threshold: {best_threshold:.2f}"
)

print(
    f"Best validation F1: {best_f1:.4f}"
)


# =========================================================
# TEST EVALUATION
# =========================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_prob = model.predict_proba(
    X_test
)[:, 1]


test_pred = (
    test_prob >= best_threshold
).astype(int)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

report = classification_report(
    y_test,
    test_pred,
    target_names=[
        "Non-Critical",
        "Critical"
    ],
    digits=4,
    zero_division=0
)

print(report)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    test_pred
)

print("Confusion Matrix:")

print(cm)


# =========================================================
# PR-AUC
# =========================================================

pr_auc = average_precision_score(
    y_test,
    test_prob
)

print(
    f"\nPR-AUC: {pr_auc:.4f}"
)


# =========================================================
# OTHER METRICS
# =========================================================

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)


print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)


# =========================================================
# SAVE RESULTS
# =========================================================

output_file = "rf_baseline_results.txt"

with open(
    output_file,
    "w"
) as f:

    f.write(
        "CRITICAL NODE IDENTIFICATION\n"
    )

    f.write(
        "RANDOM FOREST BASELINE\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Total usable rows: {len(df):,}\n"
    )

    f.write(
        f"Unique scenarios: {df['case_id'].nunique()}\n\n"
    )

    f.write(
        f"Training scenarios: {len(train_cases)}\n"
    )

    f.write(
        f"Validation scenarios: {len(val_cases)}\n"
    )

    f.write(
        f"Test scenarios: {len(test_cases)}\n\n"
    )

    f.write(
        f"Training rows used: {len(train_df):,}\n"
    )

    f.write(
        f"Validation rows: {len(val_df):,}\n"
    )

    f.write(
        f"Test rows: {len(test_df):,}\n\n"
    )

    f.write(
        f"Best threshold: {best_threshold:.2f}\n"
    )

    f.write(
        f"Validation F1: {best_f1:.4f}\n\n"
    )

    f.write(
        "TEST RESULTS\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1 Score: {f1:.4f}\n"
    )

    f.write(
        f"PR-AUC: {pr_auc:.4f}\n\n"
    )

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        str(cm) + "\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        report + "\n"
    )

    f.write(
        "\nFEATURE IMPORTANCE\n"
    )

    f.write(
        importance.to_string(index=False)
    )


print(
    f"\nResults saved to: {output_file}"
)

print("\n" + "=" * 60)
print("BASELINE COMPLETE")
print("=" * 60)