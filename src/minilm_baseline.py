"""
MiniLM + Logistic Regression Baseline
--------------------------------------

Pretrained language-embedding baseline for the canonical Reddit
troll-detection experiment.

Model:
    sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:
    384

Classifier:
    Logistic Regression

Canonical split:
    Train      = 5546
    Validation = 1189
    Test       = 1189

Random seed:
    42
"""

import os
import random
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\sanow\Desktop\sanower\dataset.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RANDOM_STATE = 42
BATCH_SIZE = 32

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("MINILM + LOGISTIC REGRESSION BASELINE")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:", df.shape)
print("Columns:", list(df.columns))


# ============================================================
# REMOVE DUPLICATE TEXTS
# ============================================================

df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("\nAfter deduplication:")
print("Shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# CANONICAL SPLIT
# ============================================================
#
# IMPORTANT:
# The manuscript uses the already established canonical
# Reddit split. This script therefore reconstructs the same
# 70/15/15 stratified split with random_state = 42.
#
# ============================================================

from sklearn.model_selection import train_test_split


train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=RANDOM_STATE
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE
)


# ============================================================
# VERIFY CANONICAL SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CANONICAL SPLIT")
print("=" * 70)

print("Training   :", len(train_df))
print("Validation :", len(validation_df))
print("Test       :", len(test_df))

assert len(train_df) == 5546
assert len(validation_df) == 1189
assert len(test_df) == 1189

print("\nTest distribution:")
print(test_df["label"].value_counts().sort_index())


# ============================================================
# EXTRACT TEXT AND LABELS
# ============================================================

X_train_text = train_df["text"].astype(str).tolist()
X_val_text = validation_df["text"].astype(str).tolist()
X_test_text = test_df["text"].astype(str).tolist()

y_train = train_df["label"].to_numpy()
y_val = validation_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()


# ============================================================
# LOAD PRETRAINED MINILM
# ============================================================

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n" + "=" * 70)
print("LOADING PRETRAINED LANGUAGE EMBEDDING MODEL")
print("=" * 70)

print("Model :", MODEL_NAME)
print("Device:", device)

encoder = SentenceTransformer(
    MODEL_NAME,
    device=device
)

print("Embedding dimension:", encoder.get_sentence_embedding_dimension())


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(texts, name):

    print("\nGenerating", name, "embeddings...")
    print("Number of texts:", len(texts))

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(np.float32)

    print("Shape:", embeddings.shape)
    print("dtype:", embeddings.dtype)

    return embeddings


train_embeddings = generate_embeddings(
    X_train_text,
    "training"
)

validation_embeddings = generate_embeddings(
    X_val_text,
    "validation"
)

test_embeddings = generate_embeddings(
    X_test_text,
    "test"
)


# ============================================================
# EMBEDDING INTEGRITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("EMBEDDING INTEGRITY CHECK")
print("=" * 70)

for name, emb in [
    ("TRAIN", train_embeddings),
    ("VALIDATION", validation_embeddings),
    ("TEST", test_embeddings)
]:

    norms = np.linalg.norm(emb, axis=1)

    print("\n" + name)
    print("Shape     :", emb.shape)
    print("dtype     :", emb.dtype)
    print("Finite    :", np.isfinite(emb).all())
    print("Mean norm :", norms.mean())
    print("Min norm  :", norms.min())
    print("Max norm  :", norms.max())

    assert np.isfinite(emb).all()


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

classifier = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=2000
)

classifier.fit(
    train_embeddings,
    y_train
)

print("Training completed.")


# ============================================================
# VALIDATION
# ============================================================

val_predictions = classifier.predict(validation_embeddings)
val_probabilities = classifier.predict_proba(validation_embeddings)[:, 1]

print("\n" + "=" * 70)
print("MINILM BASELINE — VALIDATION")
print("=" * 70)

print(
    classification_report(
        y_val,
        val_predictions,
        target_names=["Non-Troll", "Troll"],
        digits=4
    )
)

print(
    "ROC-AUC:",
    f"{roc_auc_score(y_val, val_probabilities):.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_predictions = classifier.predict(test_embeddings)

test_probabilities = classifier.predict_proba(
    test_embeddings
)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

cm = confusion_matrix(
    y_test,
    test_predictions
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MINILM BASELINE — FINAL CANONICAL TEST")
print("=" * 70)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Non-Troll", "Troll"],
        digits=4
    )
)

print("Accuracy :", f"{accuracy:.4f}")
print("Precision:", f"{precision:.4f}")
print("Recall   :", f"{recall:.4f}")
print("F1       :", f"{f1:.4f}")
print("ROC-AUC  :", f"{roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# VERIFY EXPECTED FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT VERIFICATION")
print("=" * 70)

print("Expected:")
print("Accuracy : 0.9243")
print("Precision: 0.9297")
print("Recall   : 0.9814")
print("F1       : 0.9549")
print("ROC-AUC  : 0.9609")

print("\nObtained:")
print("Accuracy :", f"{accuracy:.4f}")
print("Precision:", f"{precision:.4f}")
print("Recall   :", f"{recall:.4f}")
print("F1       :", f"{f1:.4f}")
print("ROC-AUC  :", f"{roc_auc:.4f}")


# ============================================================
# SAVE METRICS
# ============================================================

metrics_file = os.path.join(
    OUTPUT_DIR,
    "minilm_metrics.txt"
)

with open(metrics_file, "w", encoding="utf-8") as f:

    f.write("MiniLM + Logistic Regression\n")
    f.write("=" * 50 + "\n")

    f.write("Model: sentence-transformers/all-MiniLM-L6-v2\n")
    f.write("Embedding dimension: 384\n")
    f.write("Batch size: 32\n")
    f.write("Normalized embeddings: True\n")
    f.write("Random state: 42\n\n")

    f.write("Canonical Reddit Test Set\n")
    f.write("N = 1189\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1       : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")

    f.write("Confusion Matrix\n")
    f.write(str(cm))


# ============================================================
# SAVE ROC DATA
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    test_probabilities
)

roc_file = os.path.join(
    OUTPUT_DIR,
    "minilm_roc_data.npz"
)

np.savez(
    roc_file,
    fpr=fpr,
    tpr=tpr,
    thresholds=thresholds
)


print("\nResults saved to:")
print(metrics_file)
print(roc_file)

print("\n" + "=" * 70)
print("MINILM BASELINE COMPLETE")
print("=" * 70)