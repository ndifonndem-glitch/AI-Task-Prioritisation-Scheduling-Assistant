import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score
)

# ============================================================
# Directories & Dataset Path
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "knowledge_worker_task_priority_5000.csv"
MODEL_DIR = BASE_DIR / "ml_service" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "decision_tree_priority.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ============================================================
# Load Dataset
# ============================================================
print("=" * 60)
print("LOADING DATASET (Dataset 2 Schema)")
print("=" * 60)
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} rows from {DATA_FILE}")

# ============================================================
# Feature Engineering (Encoding Stated Claims)
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df["urgency_density"] = 1.0 / (df["hours_to_deadline"] + 0.1)
df["deferral_starvation_index"] = df["times_deferred"] * (df["age_days"] + 1.0)
df["work_pressure_ratio"] = df["est_duration_h"] / (df["hours_to_deadline"] + 0.1)

FEATURE_COLS = [
    "hours_to_deadline",
    "est_duration_h",
    "stated_importance",
    "slack_ratio",
    "fits_largest_block",
    "dep_blocked",
    "n_dependents",
    "times_deferred",
    "age_days",
    "urgency_density",
    "deferral_starvation_index",
    "work_pressure_ratio"
]

TARGET_COL = "label"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ============================================================
# Train / Validation / Test Split (70 / 15 / 15)
# ============================================================
print("\n" + "=" * 60)
print("DATA SPLIT (70% Train, 15% Val, 15% Test)")
print("=" * 60)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train set: {len(X_train)} samples")
print(f"Val set:   {len(X_val)} samples")
print(f"Test set:  {len(X_test)} samples")

# ============================================================
# Baseline 1: Majority Class Dummy Classifier
# ============================================================
print("\n" + "=" * 60)
print("BASELINE 1: MAJORITY CLASS DUMMY CLASSIFIER")
print("=" * 60)

dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)
dummy_pred = dummy_model.predict(X_test)

dummy_acc = accuracy_score(y_test, dummy_pred)
dummy_f1 = f1_score(y_test, dummy_pred, average="weighted")
print(f"Dummy Baseline Accuracy: {dummy_acc:.4f}")
print(f"Dummy Baseline Weighted F1: {dummy_f1:.4f}")

# ============================================================
# Baseline 2: Logistic Regression
# ============================================================
print("\n" + "=" * 60)
print("BASELINE 2: LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average="weighted")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"Logistic Regression Weighted F1: {lr_f1:.4f}")

# ============================================================
# Target Model: Symbolist Decision Tree Classifier
# ============================================================
print("\n" + "=" * 60)
print("TRAINING DECISION TREE CLASSIFIER (SYMBOLIST TRIBE)")
print("=" * 60)

best_depth = None
best_val_f1 = 0
best_tree = None

for depth in range(3, 12):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42, criterion="gini")
    dt.fit(X_train, y_train)
    val_pred = dt.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="weighted")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_depth = depth
        best_tree = dt

print(f"Selected Optimal max_depth: {best_depth} (Val F1: {best_val_f1:.4f})")

# Final Evaluation on Test Set
test_pred = best_tree.predict(X_test)

dt_acc = accuracy_score(y_test, test_pred)
dt_prec = precision_score(y_test, test_pred, average="weighted")
dt_rec = recall_score(y_test, test_pred, average="weighted")
dt_f1 = f1_score(y_test, test_pred, average="weighted")

print("\n--- TEST SET RESULTS ---")
print(f"Accuracy:  {dt_acc:.4f}")
print(f"Precision: {dt_prec:.4f}")
print(f"Recall:    {dt_rec:.4f}")
print(f"F1 Score:  {dt_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, test_pred))

# Feature Importance Analysis
print("\nFeature Importances:")
importances = pd.Series(best_tree.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for col, imp in importances.items():
    print(f"  {col:<26}: {imp:.4f}")

# Extract Symbolic Rules
print("\nExtracted Symbolic Rules (First 3 levels):")
tree_rules = export_text(best_tree, feature_names=FEATURE_COLS, max_depth=3)
print(tree_rules)

# ============================================================
# Save Model Artifacts
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL ARTIFACTS")
print("=" * 60)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_tree, f)
print(f"Saved Decision Tree model to {MODEL_PATH}")

metadata = {
    "model_type": "DecisionTreeClassifier",
    "max_depth": best_depth,
    "feature_names": FEATURE_COLS,
    "target_classes": list(best_tree.classes_),
    "metrics": {
        "accuracy": round(dt_acc, 4),
        "precision": round(dt_prec, 4),
        "recall": round(dt_rec, 4),
        "f1_score": round(dt_f1, 4),
        "baseline_dummy_f1": round(dummy_f1, 4),
        "baseline_logistic_f1": round(lr_f1, 4)
    },
    "feature_importances": {col: round(imp, 4) for col, imp in importances.items()}
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved model metadata to {METADATA_PATH}")

print("\nModel training and artifact serialization complete.")
