#!/usr/bin/env python3

"""
train_hybrid_model.py

Training pipeline for the Hybrid Contrastive–Classification framework.

Expected input:
    CSV containing:

        f0, f1, ..., f767,
        aux_len,
        aux_punct_count,
        aux_uppercase_ratio,
        aux_sentiment,
        label

The resulting feature vector contains:

        768 MPNet dimensions
        + 4 auxiliary linguistic features
        --------------------------------
        = 772 dimensions

Training configuration:
    Optimizer       : AdamW
    Learning rate   : 2e-5
    Weight decay    : 0.01
    Epochs          : 5
    Batch size      : 16
    Random seed     : 42
    Alpha           : 0 during warm-up epoch,
                      0.5 thereafter
    Early stopping  : patience = 2
    Model selection : highest validation macro-F1

Usage:

    python train_hybrid_model.py \
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

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


# ============================================================
# PROJECT MODEL
# ============================================================

from src.model_architecture import ProjectionWithClassifier


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_EMBEDDING_DIM = 768
EXPECTED_AUX_DIM = 4
EXPECTED_INPUT_DIM = 772

AUXILIARY_COLUMNS = [
    "aux_len",
    "aux_punct_count",
    "aux_uppercase_ratio",
    "aux_sentiment"
]

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_ALPHA = 0.5
DEFAULT_PATIENCE = 2
DEFAULT_SEED = 42


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

    # Deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DIRECTORY
# ============================================================

def ensure_dir(path):

    os.makedirs(path, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

class NumpyDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            np.asarray(X),
            dtype=torch.float32
        )

        self.y = torch.tensor(
            np.asarray(y),
            dtype=torch.long
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# ============================================================
# DATA LOADING
# ============================================================

def load_features_from_csv(path):

    """
    Loads the final 772-dimensional hybrid representation.

    Required columns:

        f0 ... f767
        aux_len
        aux_punct_count
        aux_uppercase_ratio
        aux_sentiment
        label
    """

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
    # MPNet columns
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
            "Missing MPNet embedding columns. "
            f"First missing columns: {missing_embeddings[:10]}"
        )

    # --------------------------------------------------------
    # Auxiliary columns
    # --------------------------------------------------------

    missing_aux = [
        col
        for col in AUXILIARY_COLUMNS
        if col not in df.columns
    ]

    if missing_aux:

        raise ValueError(
            "Missing auxiliary feature columns: "
            f"{missing_aux}"
        )

    # --------------------------------------------------------
    # Construct hybrid representation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Verify dimensionality
    # --------------------------------------------------------

    if X.shape[1] != EXPECTED_INPUT_DIM:

        raise ValueError(
            f"Expected {EXPECTED_INPUT_DIM} input features, "
            f"but obtained {X.shape[1]}."
        )

    if len(X) != len(y):

        raise ValueError(
            "Feature and label lengths do not match."
        )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    unique_labels = np.unique(y)

    if not np.array_equal(
        unique_labels,
        np.array([0, 1])
    ):

        raise ValueError(
            "Expected binary labels {0, 1}, "
            f"but found {unique_labels}."
        )

    print("\nInput verification")
    print("-" * 60)
    print("Samples              :", len(df))
    print("MPNet dimensions     :", EXPECTED_EMBEDDING_DIM)
    print("Auxiliary dimensions :", EXPECTED_AUX_DIM)
    print("Total dimensions     :", X.shape[1])
    print("Labels               :", unique_labels)
    print("-" * 60)

    return X, y


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.5,
    eps=1e-8
):

    """
    Supervised contrastive loss.

    For each anchor, samples with the same class label are
    treated as positives and samples from the other class
    are treated as negatives.

    embeddings are expected to be normalized.
    """

    device = embeddings.device

    batch_size = embeddings.shape[0]

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarity = torch.matmul(
        embeddings,
        embeddings.T
    )

    similarity = similarity / temperature

    # --------------------------------------------------------
    # Remove self-comparisons
    # --------------------------------------------------------

    self_mask = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=device
    )

    labels = labels.view(-1, 1)

    positive_mask = torch.eq(
        labels,
        labels.T
    )

    positive_mask = (
        positive_mask &
        (~self_mask)
    )

    # --------------------------------------------------------
    # Numerical stability
    # --------------------------------------------------------

    similarity = similarity.masked_fill(
        self_mask,
        -1e9
    )

    log_prob = similarity - torch.logsumexp(
        similarity,
        dim=1,
        keepdim=True
    )

    # --------------------------------------------------------
    # Mean positive log probability
    # --------------------------------------------------------

    positive_count = positive_mask.sum(
        dim=1
    )

    valid = positive_count > 0

    if not torch.any(valid):

        return torch.tensor(
            0.0,
            device=device,
            requires_grad=True
        )

    mean_log_prob_pos = (
        (
            positive_mask.float() *
            log_prob
        ).sum(dim=1)
        /
        positive_count.clamp(min=1)
    )

    loss = -mean_log_prob_pos[valid].mean()

    return loss


# ============================================================
# HYBRID LOSS
# ============================================================

def hybrid_loss(
    embeddings,
    logits,
    labels,
    alpha=0.5,
    temperature=0.5
):

    """
    Combined objective:

        L = (1-alpha) * CE
            + alpha * SupCon
    """

    ce_loss = F.cross_entropy(
        logits,
        labels
    )

    supcon_loss = supervised_contrastive_loss(
        embeddings,
        labels,
        temperature=temperature
    )

    total_loss = (
        (1.0 - alpha) * ce_loss
        +
        alpha * supcon_loss
    )

    return (
        total_loss,
        ce_loss.item(),
        supcon_loss.item()
    )


# ============================================================
# EXPECTED CALIBRATION ERROR
# ============================================================

def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10
):

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        if i < n_bins - 1:

            mask = (
                (y_prob >= bins[i]) &
                (y_prob < bins[i + 1])
            )

        else:

            mask = (
                (y_prob >= bins[i]) &
                (y_prob <= bins[i + 1])
            )

        if np.sum(mask) == 0:

            continue

        accuracy = np.mean(
            y_true[mask]
        )

        confidence = np.mean(
            y_prob[mask]
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
    X,
    y,
    device,
    batch_size=128
):

    model.eval()

    dataset = NumpyDataset(
        X,
        y
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    predictions = []
    probabilities = []
    labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            embeddings, logits = model(
                X_batch
            )

            probs = F.softmax(
                logits,
                dim=1
            )[:, 1]

            preds = torch.argmax(
                logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels.extend(
                y_batch.numpy()
            )

    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)
    y_prob = np.asarray(probabilities)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        digits=4
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

    except Exception:

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
        y_prob,
        n_bins=10
    )

    brier = brier_score_loss(
        y_true,
        y_prob
    )

    return {
        "report": report,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "ece": ece,
        "brier": brier,
        "preds": y_pred,
        "probs": y_prob,
        "labels": y_true
    }


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

def save_evaluation_results(
    metrics,
    outdir,
    prefix
):

    ensure_dir(outdir)

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_path = os.path.join(
        outdir,
        f"{prefix}_classification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            metrics["report"]
        )

        f.write(
            "\n\nConfusion Matrix:\n"
        )

        f.write(
            np.array2string(
                metrics["confusion_matrix"]
            )
        )

        f.write(
            f"\n\nMacro-F1: "
            f"{metrics['macro_f1']:.6f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.6f}\n"
        )

        f.write(
            f"ECE: "
            f"{metrics['ece']:.6f}\n"
        )

        f.write(
            f"Brier Score: "
            f"{metrics['brier']:.6f}\n"
        )

    # --------------------------------------------------------
    # Calibration metrics
    # --------------------------------------------------------

    calibration_path = os.path.join(
        outdir,
        f"{prefix}_calibration_metrics.txt"
    )

    with open(
        calibration_path,
        "w"
    ) as f:

        f.write(
            f"ROC-AUC: {metrics['roc_auc']:.6f}\n"
        )

        f.write(
            f"ECE: {metrics['ece']:.6f}\n"
        )

        f.write(
            f"Brier Score: {metrics['brier']:.6f}\n"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = metrics[
        "confusion_matrix"
    ]

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    im = ax.imshow(
        cm,
        interpolation="nearest"
    )

    ax.set_title(
        f"Confusion Matrix - {prefix}"
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
        im,
        ax=ax
    )

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            outdir,
            f"{prefix}_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close(fig)

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    if not np.isnan(
        metrics["roc_auc"]
    ):

        fpr, tpr, _ = roc_curve(
            metrics["labels"],
            metrics["probs"]
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        ax.plot(
            fpr,
            tpr,
            label=(
                f"ROC-AUC = "
                f"{metrics['roc_auc']:.4f}"
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
            f"ROC Curve - {prefix}"
        )

        ax.legend(
            loc="lower right"
        )

        plt.tight_layout()

        fig.savefig(
            os.path.join(
                outdir,
                f"{prefix}_roc.png"
            ),
            dpi=300
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Reliability diagram
    # --------------------------------------------------------

    prob_true, prob_pred = calibration_curve(
        metrics["labels"],
        metrics["probs"],
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
        f"Reliability Diagram - {prefix}"
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            outdir,
            f"{prefix}_reliability.png"
        ),
        dpi=300
    )

    plt.close(fig)


# ============================================================
# TRAINING
# ============================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    alpha_final=DEFAULT_ALPHA,
    patience=DEFAULT_PATIENCE,
    outdir="results"
):

    ensure_dir(outdir)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ProjectionWithClassifier(
        input_dim=EXPECTED_INPUT_DIM
    ).to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_dataset = NumpyDataset(
        X_train,
        y_train
    )

    val_dataset = NumpyDataset(
        X_val,
        y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_f1 = -np.inf
    epochs_without_improvement = 0

    history = []

    best_model_path = os.path.join(
        outdir,
        "best_model.pt"
    )

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_supcon = 0.0

        num_batches = 0

        # ----------------------------------------------------
        # Manuscript schedule:
        #
        # Epoch 1 -> alpha = 0
        # Epoch 2+ -> alpha = 0.5
        # ----------------------------------------------------

        if epoch == 0:

            alpha = 0.0

        else:

            alpha = alpha_final

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            embeddings, logits = model(
                X_batch
            )

            loss, ce_value, supcon_value = hybrid_loss(
                embeddings,
                logits,
                y_batch,
                alpha=alpha
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_value
            total_supcon += supcon_value

            num_batches += 1

        # ----------------------------------------------------
        # Average losses
        # ----------------------------------------------------

        avg_loss = (
            total_loss / num_batches
            if num_batches > 0
            else 0.0
        )

        avg_ce = (
            total_ce / num_batches
            if num_batches > 0
            else 0.0
        )

        avg_supcon = (
            total_supcon / num_batches
            if num_batches > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate_model(
            model,
            X_val,
            y_val,
            device
        )

        val_f1 = val_metrics[
            "macro_f1"
        ]

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "ce_loss": avg_ce,
            "supcon_loss": avg_supcon,
            "alpha": alpha,
            "val_f1": val_f1,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_ece": val_metrics["ece"],
            "val_brier": val_metrics["brier"]
        })

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        print(
            f"  Loss       : {avg_loss:.6f}"
        )

        print(
            f"  CE Loss    : {avg_ce:.6f}"
        )

        print(
            f"  SupCon Loss: {avg_supcon:.6f}"
        )

        print(
            f"  Alpha      : {alpha:.2f}"
        )

        print(
            f"  Val Macro-F1: {val_f1:.6f}"
        )

        print(
            f"  Val ROC-AUC : "
            f"{val_metrics['roc_auc']:.6f}"
        )

        print(
            f"  Val ECE     : "
            f"{val_metrics['ece']:.6f}"
        )

        # ----------------------------------------------------
        # Checkpoint selection
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                "  ✓ Best validation checkpoint saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                "  No validation improvement "
                f"({epochs_without_improvement}/"
                f"{patience})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        os.path.join(
            outdir,
            "train_history.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    if not os.path.exists(
        best_model_path
    ):

        raise RuntimeError(
            "Best model checkpoint was not created."
        )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    print(
        f"\nBest validation Macro-F1: "
        f"{best_val_f1:.6f}"
    )

    print(
        f"Best checkpoint: "
        f"{best_model_path}"
    )

    return model, history_df


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train the 772-dimensional "
            "Hybrid Contrastive–Classification model."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_data_embeddings.csv",
        help="Preprocessed 772-dimensional feature CSV."
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Directory for model and evaluation outputs."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            "Supervised contrastive weight after "
            "the warm-up epoch."
        )
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "HYBRID CONTRASTIVE–CLASSIFICATION TRAINING"
    )
    print("=" * 70)

    print(
        "Device        :", device
    )

    print(
        "Random seed   :", args.seed
    )

    print(
        "Input         :", args.input
    )

    print(
        "Epochs        :", args.epochs
    )

    print(
        "Batch size    :", args.batch_size
    )

    print(
        "Learning rate :", args.lr
    )

    print(
        "Weight decay  :", args.weight_decay
    )

    print(
        "Alpha         :", args.alpha
    )

    print(
        "Patience      :", args.patience
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y = load_features_from_csv(
        args.input
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # This script expects train/validation/test partitions
    # to be supplied consistently with the canonical dataset
    # split used in the manuscript.
    #
    # For the repository's sample pipeline, the following
    # stratified split is used.
    # --------------------------------------------------------

    from sklearn.model_selection import train_test_split

    # First isolate the canonical test portion
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=1189,
        stratify=y,
        random_state=args.seed
    )

    # Then construct the validation partition
    #
    # The manuscript uses 1189 validation instances.
    #
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=1189,
        stratify=y_train_val,
        random_state=args.seed
    )

    print("\nDataset partitions")
    print("-" * 60)

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

    print("-" * 60)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model, history = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        alpha_final=args.alpha,
        patience=args.patience,
        outdir=args.outdir
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        device
    )

    save_evaluation_results(
        test_metrics,
        args.outdir,
        prefix="test_final"
    )

    # --------------------------------------------------------
    # Save final numerical summary
    # --------------------------------------------------------

    summary = pd.DataFrame([
        {
            "accuracy": np.mean(
                test_metrics["preds"]
                ==
                test_metrics["labels"]
            ),
            "macro_f1": test_metrics["macro_f1"],
            "roc_auc": test_metrics["roc_auc"],
            "ece": test_metrics["ece"],
            "brier_score": test_metrics["brier"]
        }
    ])

    summary.to_csv(
        os.path.join(
            args.outdir,
            "test_metrics.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print(
        "\nFinal test Macro-F1:",
        f"{test_metrics['macro_f1']:.6f}"
    )

    print(
        "Final test ROC-AUC :",
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        "Final test ECE     :",
        f"{test_metrics['ece']:.6f}"
    )

    print(
        "Final test Brier   :",
        f"{test_metrics['brier']:.6f}"
    )

    print(
        "\nClassification report:"
    )

    print(
        test_metrics["report"]
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        test_metrics["confusion_matrix"]
    )

    print(
        "\nResults saved to:",
        args.outdir
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()