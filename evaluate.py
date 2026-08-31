#!/usr/bin/env python3

"""
evaluate.py

Evaluate the trained Hybrid Contrastive–Classification model
on the held-out test partition.

Expected input CSV:

    f0 ... f767
    aux_len
    aux_punct_count
    aux_uppercase_ratio
    aux_sentiment
    label

Total input dimensions = 772.

Usage:

    python evaluate.py \
        --model results/best_model.pt \
        --input data/sample_data_embeddings.csv \
        --outdir results
"""

import os
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

from src.model_architecture import ProjectionWithClassifier


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_EMBEDDING_DIM = 768

AUXILIARY_COLUMNS = [
    "aux_len",
    "aux_punct_count",
    "aux_uppercase_ratio",
    "aux_sentiment"
]

EXPECTED_INPUT_DIM = 772

TEST_SIZE = 1189
VALIDATION_SIZE = 1189

BATCH_SIZE = 32
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# ============================================================
# LOAD DATA
# ============================================================

def load_features(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    df = pd.read_csv(path)

    if "label" not in df.columns:

        raise ValueError(
            "Input CSV must contain a 'label' column."
        )

    # --------------------------------------------------------
    # MPNet features
    # --------------------------------------------------------

    embedding_columns = [
        f"f{i}"
        for i in range(EXPECTED_EMBEDDING_DIM)
    ]

    missing_embeddings = [
        col
        for col in embedding_columns
        if col not in df.columns
    ]

    if missing_embeddings:

        raise ValueError(
            "Missing embedding columns: "
            f"{missing_embeddings[:10]}"
        )

    # --------------------------------------------------------
    # Auxiliary features
    # --------------------------------------------------------

    missing_aux = [
        col
        for col in AUXILIARY_COLUMNS
        if col not in df.columns
    ]

    if missing_aux:

        raise ValueError(
            "Missing auxiliary columns: "
            f"{missing_aux}"
        )

    feature_columns = (
        embedding_columns +
        AUXILIARY_COLUMNS
    )

    X = df[
        feature_columns
    ].values.astype(np.float32)

    y = df[
        "label"
    ].values.astype(np.int64)

    if X.shape[1] != EXPECTED_INPUT_DIM:

        raise ValueError(
            f"Expected {EXPECTED_INPUT_DIM} features, "
            f"but found {X.shape[1]}."
        )

    return X, y


# ============================================================
# ECE
# ============================================================

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10
):

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:

            mask = (
                (probabilities >= bins[i]) &
                (probabilities <= bins[i + 1])
            )

        else:

            mask = (
                (probabilities >= bins[i]) &
                (probabilities < bins[i + 1])
            )

        if np.sum(mask) == 0:

            continue

        confidence = np.mean(
            probabilities[mask]
        )

        accuracy = np.mean(
            y_true[mask]
        )

        ece += (
            np.sum(mask) /
            len(y_true)
        ) * abs(
            accuracy - confidence
        )

    return ece


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    device,
    batch_size=32
):

    dataset = TestDataset(
        X_test,
        y_test
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()

    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            embeddings, logits = model(
                X_batch
            )

            probabilities = F.softmax(
                logits,
                dim=1
            )[:, 1]

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                y_batch.numpy()
            )

    y_true = np.asarray(
        all_labels
    )

    y_pred = np.asarray(
        all_predictions
    )

    y_prob = np.asarray(
        all_probabilities
    )

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

    except ValueError:

        roc_auc = float("nan")

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    ece = expected_calibration_error(
        y_true,
        y_prob
    )

    brier = brier_score_loss(
        y_true,
        y_prob
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "ece": ece,
        "brier_score": brier,
        "confusion_matrix": cm,
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    outdir
):

    os.makedirs(
        outdir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    report_file = os.path.join(
        outdir,
        "test_metrics_report.txt"
    )

    with open(
        report_file,
        "w"
    ) as f:

        f.write(
            "FINAL TEST EVALUATION\n"
        )

        f.write(
            "=" * 60 +
            "\n\n"
        )

        f.write(
            f"Accuracy          : "
            f"{results['accuracy']:.6f}\n"
        )

        f.write(
            f"Macro-Precision   : "
            f"{results['macro_precision']:.6f}\n"
        )

        f.write(
            f"Macro-Recall      : "
            f"{results['macro_recall']:.6f}\n"
        )

        f.write(
            f"Macro-F1          : "
            f"{results['macro_f1']:.6f}\n"
        )

        f.write(
            f"ROC-AUC           : "
            f"{results['roc_auc']:.6f}\n"
        )

        f.write(
            f"ECE               : "
            f"{results['ece']:.6f}\n"
        )

        f.write(
            f"Brier Score       : "
            f"{results['brier_score']:.6f}\n"
        )

        f.write(
            "\n\nClassification Report\n"
        )

        f.write(
            "-" * 60 +
            "\n"
        )

        f.write(
            results["report"]
        )

        f.write(
            "\nConfusion Matrix\n"
        )

        f.write(
            str(
                results["confusion_matrix"]
            )
        )

    # --------------------------------------------------------
    # CSV summary
    # --------------------------------------------------------

    summary = pd.DataFrame([{
        "accuracy": results["accuracy"],
        "macro_precision": results["macro_precision"],
        "macro_recall": results["macro_recall"],
        "macro_f1": results["macro_f1"],
        "roc_auc": results["roc_auc"],
        "ece": results["ece"],
        "brier_score": results["brier_score"]
    }])

    summary.to_csv(
        os.path.join(
            outdir,
            "test_metrics.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = results[
        "confusion_matrix"
    ]

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            outdir,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    if not np.isnan(
        results["roc_auc"]
    ):

        fpr, tpr, _ = roc_curve(
            results["y_true"],
            results["y_prob"]
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        ax.plot(
            fpr,
            tpr,
            label=(
                f"ROC-AUC = "
                f"{results['roc_auc']:.4f}"
            )
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "ROC Curve"
        )

        ax.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                outdir,
                "roc_curve.png"
            ),
            dpi=300
        )

        plt.close()

    # --------------------------------------------------------
    # Reliability diagram
    # --------------------------------------------------------

    prob_true, prob_pred = calibration_curve(
        results["y_true"],
        results["y_prob"],
        n_bins=10
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.plot(
        prob_pred,
        prob_true,
        marker="o",
        label="Model"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    ax.set_xlabel(
        "Mean Predicted Probability"
    )

    ax.set_ylabel(
        "Fraction of Positives"
    )

    ax.set_title(
        "Reliability Diagram"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            outdir,
            "reliability_diagram.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="results/best_model.pt"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_data_embeddings.csv"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 70
    )

    print(
        "FINAL MODEL EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Load full feature matrix
    # --------------------------------------------------------

    X, y = load_features(
        args.input
    )

    print(
        "\nTotal samples:",
        len(X)
    )

    print(
        "Feature dimensions:",
        X.shape[1]
    )

    # --------------------------------------------------------
    # Reproduce the same deterministic partitioning
    # --------------------------------------------------------

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=args.seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        stratify=y_train_val,
        random_state=args.seed
    )

    print(
        "\nPartitions"
    )

    print(
        "Training   :",
        X_train.shape
    )

    print(
        "Validation :",
        X_val.shape
    )

    print(
        "Test       :",
        X_test.shape
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not os.path.exists(
        args.model
    ):

        raise FileNotFoundError(
            f"Model checkpoint not found: "
            f"{args.model}"
        )

    model = ProjectionWithClassifier(
        input_dim=EXPECTED_INPUT_DIM
    )

    checkpoint = torch.load(
        args.model,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)

    print(
        "\nLoaded checkpoint:",
        args.model
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        device=device,
        batch_size=args.batch_size
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy        : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Macro-Precision : "
        f"{results['macro_precision']:.4f}"
    )

    print(
        f"Macro-Recall    : "
        f"{results['macro_recall']:.4f}"
    )

    print(
        f"Macro-F1        : "
        f"{results['macro_f1']:.4f}"
    )

    print(
        f"ROC-AUC         : "
        f"{results['roc_auc']:.4f}"
    )

    print(
        f"ECE             : "
        f"{results['ece']:.4f}"
    )

    print(
        f"Brier Score     : "
        f"{results['brier_score']:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        results["report"]
    )

    print(
        "Confusion Matrix:"
    )

    print(
        results["confusion_matrix"]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        args.outdir
    )

    print(
        "\nEvaluation files saved in:",
        args.outdir
    )


if __name__ == "__main__":

    main()