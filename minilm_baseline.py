"""
MiniLM + Logistic Regression Baseline
--------------------------------------

Frozen pretrained MiniLM sentence embeddings followed by
Logistic Regression classification.

This baseline uses the same canonical Reddit partition
as the main experiments:

    Train      : 5546
    Validation : 1189
    Test       : 1189

Model:
    sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:
    384

Classifier:
    Logistic Regression

Random seed:
    42
"""

import os
import random
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RANDOM_STATE = 42
BATCH_SIZE = 32

TRAIN_SIZE = 5546
VAL_SIZE = 1189
TEST_SIZE = 1189


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# CANONICAL SPLIT
# ============================================================

def canonical_split(df):
    """
    Reproduce the canonical Reddit partition used in the
    main experiments.

    Split:
        70% train
        15% validation
        15% test

    The split is stratified and uses random_state=42.
    """

    texts = df["text"].fillna("").astype(str).to_numpy()
    labels = df["label"].astype(int).to_numpy()

    # First: isolate 15% test
    X_rest, X_test, y_rest, y_test = train_test_split(
        texts,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    # Second: isolate 15% validation from the original dataset.
    # Relative fraction within the remaining 85%.
    val_fraction = 0.15 / 0.85

    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_fraction,
        stratify=y_rest,
        random_state=RANDOM_STATE
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="MiniLM + Logistic Regression baseline"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_data.csv",
        help="CSV containing text and label columns"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Directory for MiniLM results"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)


    # ========================================================
    # LOAD DATA
    # ========================================================

    print("=" * 70)
    print("MINILM + LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    print("\nLoading dataset:")
    print(args.input)

    df = pd.read_csv(args.input)

    if "text" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'text' column."
        )

    if "label" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'label' column."
        )

    print("\nDataset shape:", df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())


    # ========================================================
    # CANONICAL SPLIT
    # ========================================================

    (
        X_train_text,
        X_val_text,
        X_test_text,
        y_train,
        y_val,
        y_test
    ) = canonical_split(df)

    print("\n" + "=" * 70)
    print("CANONICAL REDDIT SPLIT")
    print("=" * 70)

    print("Training   :", len(X_train_text))
    print("Validation :", len(X_val_text))
    print("Test       :", len(X_test_text))

    # Strict verification
    assert len(X_train_text) == TRAIN_SIZE
    assert len(X_val_text) == VAL_SIZE
    assert len(X_test_text) == TEST_SIZE

    print("\nCanonical split verified.")


    # ========================================================
    # LOAD MINILM
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING PRETRAINED MINILM")
    print("=" * 70)

    print("Model:", MODEL_NAME)

    model = SentenceTransformer(
        MODEL_NAME
    )

    embedding_dim = model.get_sentence_embedding_dimension()

    print("Embedding dimension:", embedding_dim)

    assert embedding_dim == 384


    # ========================================================
    # GENERATE EMBEDDINGS
    # ========================================================

    def encode(texts, name):

        print("\nGenerating", name, "embeddings...")
        print("Samples:", len(texts))

        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        print("Shape:", embeddings.shape)

        return embeddings


    X_train = encode(
        X_train_text,
        "training"
    )

    X_val = encode(
        X_val_text,
        "validation"
    )

    X_test = encode(
        X_test_text,
        "test"
    )


    # ========================================================
    # VERIFY EMBEDDINGS
    # ========================================================

    print("\n" + "=" * 70)
    print("EMBEDDING VERIFICATION")
    print("=" * 70)

    for name, X in [
        ("Train", X_train),
        ("Validation", X_val),
        ("Test", X_test)
    ]:

        print(
            f"{name:12s}: shape={X.shape}, "
            f"finite={np.isfinite(X).all()}"
        )

        assert X.shape[1] == 384
        assert np.isfinite(X).all()


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    classifier = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=2000
    )

    classifier.fit(
        X_train,
        y_train
    )

    print("Training completed.")


    # ========================================================
    # VALIDATION
    # ========================================================

    val_predictions = classifier.predict(X_val)

    val_probabilities = classifier.predict_proba(
        X_val
    )[:, 1]

    val_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    val_f1 = f1_score(
        y_val,
        val_predictions
    )

    print("\n" + "=" * 70)
    print("VALIDATION PERFORMANCE")
    print("=" * 70)

    print(
        "Validation F1    :",
        f"{val_f1:.4f}"
    )

    print(
        "Validation ROC-AUC:",
        f"{val_auc:.4f}"
    )


    # ========================================================
    # FINAL TEST PREDICTIONS
    # ========================================================

    test_predictions = classifier.predict(
        X_test
    )

    test_probabilities = classifier.predict_proba(
        X_test
    )[:, 1]


    # ========================================================
    # TEST METRICS
    # ========================================================

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


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("MINILM — FINAL CANONICAL REDDIT TEST RESULTS")
    print("=" * 70)

    print(
        "\nAccuracy :",
        f"{accuracy:.4f}"
    )

    print(
        "Precision:",
        f"{precision:.4f}"
    )

    print(
        "Recall   :",
        f"{recall:.4f}"
    )

    print(
        "F1       :",
        f"{f1:.4f}"
    )

    print(
        "ROC-AUC  :",
        f"{roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Non-Troll",
                "Troll"
            ],
            digits=4
        )
    )


    # ========================================================
    # VERIFY AGAINST MANUSCRIPT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("MANUSCRIPT RESULT CHECK")
    print("=" * 70)

    expected = {
        "accuracy": 0.9243,
        "precision": 0.9297,
        "recall": 0.9814,
        "f1": 0.9549,
        "roc_auc": 0.9609
    }

    obtained = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    for metric in expected:

        print(
            f"{metric:10s} | "
            f"expected={expected[metric]:.4f} | "
            f"obtained={obtained[metric]:.4f}"
        )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = os.path.join(
        args.output,
        "minilm_metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "MiniLM + Logistic Regression Baseline\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            "Model: sentence-transformers/all-MiniLM-L6-v2\n"
        )

        f.write(
            "Embedding dimension: 384\n"
        )

        f.write(
            "Random seed: 42\n"
        )

        f.write(
            "Canonical split: 5546 / 1189 / 1189\n\n"
        )

        f.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        f.write(
            f"Precision: {precision:.4f}\n"
        )

        f.write(
            f"Recall   : {recall:.4f}\n"
        )

        f.write(
            f"F1       : {f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC  : {roc_auc:.4f}\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(cm)
        )


    # ========================================================
    # SAVE ROC DATA
    # ========================================================

    fpr, tpr, thresholds = roc_curve(
        y_test,
        test_probabilities
    )

    roc_path = os.path.join(
        args.output,
        "minilm_roc_data.npz"
    )

    np.savez(
        roc_path,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds
    )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame({
        "label": y_test,
        "prediction": test_predictions,
        "probability": test_probabilities
    })

    predictions_path = os.path.join(
        args.output,
        "minilm_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )


    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(metrics_path)
    print(roc_path)
    print(predictions_path)

    print("\nMiniLM baseline completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()