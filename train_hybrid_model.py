#!/usr/bin/env python3
"""
train_hybrid_model.py

Usage examples:
  # If you already created numeric embeddings CSV (f0,f1,...,label):
  python train_hybrid_model.py --input data/sample_data_embeddings.csv --epochs 10 --batch_size 64

  # If you only have a raw CSV with 'text' and 'label' columns:
  python train_hybrid_model.py --input data/sample_data.csv --epochs 10 --batch_size 64

Notes:
- This script will compute sentence-transformer embeddings automatically if input only has 'text'.
- Requires: torch, numpy, pandas, scikit-learn, matplotlib, sentence-transformers (if text->embeddings needed)
"""
import os
import argparse
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, brier_score_loss
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------- Utility functions --------
def safe_train_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Returns X_train, X_val, X_test, y_train, y_val, y_test
    Ensures at least one sample per class in splits when stratifying.
    """
    n_classes = len(np.unique(y))
    n_samples = len(y)

    # ensure test_size and val_size are fractions
    if isinstance(test_size, int):
        if test_size < n_classes:
            raise ValueError(f"test_size as int must be >= number of classes ({n_classes}) for stratify.")
    if isinstance(val_size, int):
        if val_size < n_classes:
            raise ValueError(f"val_size as int must be >= number of classes ({n_classes}) for stratify.")
    # if fractional, ensure enough samples
    if isinstance(test_size, float):
        n_test = int(math.ceil(n_samples * test_size))
        if n_test < n_classes:
            test_size = n_classes / n_samples
    if isinstance(val_size, float):
        n_val = int(math.ceil(n_samples * val_size))
        if n_val < n_classes:
            val_size = n_classes / n_samples

    # First split out test
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split rest into train/val
    # val_size is relative to original; convert to fraction of X_rest
    val_frac_of_rest = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_frac_of_rest, stratify=y_rest, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# -------- Data handling --------
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_features_from_csv(path, model_for_text="sentence-transformers/all-MiniLM-L6-v2", batch_size=64, device="cpu"):
    """
    If the CSV has columns f0,f1,... use them as numeric features.
    Otherwise, if it has 'text' and 'label' compute sentence-transformer embeddings (automatically).
    Returns X (ndarray), y (ndarray)
    """
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    if "label" not in cols:
        raise ValueError("Input CSV must contain 'label' column.")
    # detect features columns starting with f0
    fcols = [c for c in cols if c.startswith("f")]
    if len(fcols) >= 5:
        X = df[fcols].values.astype(np.float32)
        y = df["label"].values.astype(int)
        return X, y
    # else try to compute embeddings from 'text'
    if "text" in cols:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers") from e
        texts = df["text"].fillna("").astype(str).tolist()
        model = SentenceTransformer(model_for_text)
        model.max_seq_length = 512
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        X = np.array(emb, dtype=np.float32)
        y = df["label"].values.astype(int)
        return X, y
    raise ValueError("CSV contains only 'text' and 'label'? or not recognizable numeric columns. Provide f* columns or text.")

# -------- Model --------
class ProjectionWithClassifier(nn.Module):
    def __init__(self, input_dim, proj_dim=128, hidden=256, num_classes=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(proj_dim, num_classes)
    def forward(self, x):
        z = self.proj(x)          # (B, proj_dim)
        z_norm = F.normalize(z, dim=1)
        logits = self.classifier(z)
        return z_norm, logits

# -------- Losses --------
def supervised_contrastive_loss(embeddings, labels, temperature=0.5, eps=1e-8):
    """
    Supervised contrastive NT-Xent style loss:
    For each anchor i, positives are other indices with same label.
    Loss_i = - 1/|P(i)| sum_{p in P(i)} log( exp(sim(i,p)/T) / sum_{k!=i} exp(sim(i,k)/T) )
    embeddings must be normalized.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]
    sim = torch.matmul(embeddings, embeddings.t())  # (B,B) cosine if normalized
    sim = sim / temperature
    labels = labels.view(-1, 1)
    mask_pos = torch.eq(labels, labels.t()).to(device)  # (B,B)
    # remove self positives
    mask_pos = mask_pos * (~torch.eye(batch_size, dtype=torch.bool, device=device))
    # For numerical stability, set diagonal to large negative
    logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    exp_sim = torch.exp(sim) * logits_mask
    denom = exp_sim.sum(dim=1, keepdim=True) + eps  # (B,1)
    # For anchors with no positives (rare), skip them in averaging
    loss = 0.0
    valid_count = 0
    for i in range(batch_size):
        pos_idx = mask_pos[i].nonzero(as_tuple=False).squeeze(1)
        if pos_idx.numel() == 0:
            continue
        numer = torch.exp(sim[i, pos_idx]).sum()
        denom_i = denom[i].squeeze()
        loss_i = - torch.log((numer + eps) / denom_i)
        loss += loss_i
        valid_count += 1
    if valid_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss = loss / valid_count
    return loss

def hybrid_loss(embeddings, logits, labels, alpha=0.5, temperature=0.5):
    """
    Combined loss: (1-alpha) * CE + alpha * SupCon
    embeddings should be normalized vectors used for contrastive loss.
    logits are classifier outputs (not softmaxed)
    """
    ce = F.cross_entropy(logits, labels)
    supcon = supervised_contrastive_loss(embeddings, labels, temperature=temperature)
    total = (1.0 - alpha) * ce + alpha * supcon
    return total, ce.item(), supcon.item()

# -------- Metrics & plots --------
def evaluate_and_save(model, X, y, device, outdir, prefix="test"):
    model.eval()
    ds = NumpyDataset(X, y)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            emb, logits = model(xb)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    rep = classification_report(all_labels, all_preds, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc_auc = float("nan")

    # Save classification report
    ensure_dir(outdir)
    with open(os.path.join(outdir, f"{prefix}_classification_report.txt"), "w") as f:
        f.write(rep)
        f.write("\n\nConfusion matrix:\n")
        f.write(np.array2string(cm))
        f.write(f"\n\nROC-AUC: {roc_auc:.6f}\n")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix ({prefix})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}_confusion_matrix.png"))
    plt.close(fig)

    # ROC curve
    if not math.isnan(roc_auc):
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, label=f"{prefix} (AUC={roc_auc:.4f})")
        ax.plot([0,1],[0,1], linestyle="--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curve ({prefix})")
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(outdir, f"{prefix}_roc.png"))
        plt.close(fig)

    # Calibration curve & ECE + Brier
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0,1],[0,1], linestyle='--', color='orange', label='Perfect')
    ax.set_xlabel("Predicted prob")
    ax.set_ylabel("True prob")
    ax.set_title(f"Calibration ({prefix})")
    ax.legend()
    fig.savefig(os.path.join(outdir, f"{prefix}_calibration.png"))
    plt.close(fig)

    # ECE (expected calibration error)
    def expected_calibration_error(y_true, y_prob, n_bins=10):
        bins = np.linspace(0.0,1.0,n_bins+1)
        ece = 0.0
        for i in range(n_bins):
            start, end = bins[i], bins[i+1]
            idx = (y_prob >= start) & (y_prob < end) if i < n_bins-1 else (y_prob >= start) & (y_prob <= end)
            if np.sum(idx) == 0:
                continue
            acc = np.mean(y_true[idx])
            conf = np.mean(y_prob[idx])
            ece += (np.sum(idx)/len(y_true)) * abs(acc - conf)
        return ece

    ece = expected_calibration_error(all_labels, all_probs, n_bins=10)
    brier = brier_score_loss(all_labels, all_probs)
    with open(os.path.join(outdir, f"{prefix}_calibration_metrics.txt"), "w") as f:
        f.write(f"ROC-AUC: {roc_auc:.6f}\n")
        f.write(f"ECE: {ece:.6f}\n")
        f.write(f"Brier: {brier:.6f}\n")

    # return metrics
    return {
        "report": rep,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "ece": ece,
        "brier": brier,
        "probs": all_probs,
        "preds": all_preds,
        "labels": all_labels
    }

def plot_tsne_embeddings(model, X, y, device, outdir, prefix="tsne", n_samples=2000):
    """
    Robust t-SNE plotting:
    - samples up to n_samples (random subset if larger)
    - adapts perplexity to dataset size (must be < n_samples)
    - falls back to PCA when samples are too few for TSNE
    """
    import math
    from sklearn.decomposition import PCA

    model.eval()
    # sample subset if large
    n_total = len(X)
    if n_total == 0:
        print("[WARN] Empty dataset passed to plot_tsne_embeddings; skipping.")
        return

    if n_total > n_samples:
        idx = np.random.choice(n_total, size=n_samples, replace=False)
        Xs = X[idx]
        ys = y[idx]
    else:
        Xs = X
        ys = y

    with torch.no_grad():
        xb = torch.tensor(Xs, dtype=torch.float32).to(device)
        emb, _ = model(xb)
        emb = emb.cpu().numpy()

    n_pts = emb.shape[0]

    # Choose sensible perplexity: must be < n_pts, and typically >= 5.
    # Use min(30, max(2, n_pts//3 - 1)) as heuristic; ensure at least 2.
    max_perp = max(2, (n_pts // 3) - 1)
    perp = min(30, max_perp)
    if perp >= n_pts:
        # if still invalid (very small n_pts), fallback to PCA visualization
        print(f"[INFO] Too few samples for t-SNE (n_pts={n_pts}), using PCA fallback.")
        pca = PCA(n_components=2, random_state=42)
        red = pca.fit_transform(emb)
        fig, ax = plt.subplots(figsize=(8,6))
        sc = ax.scatter(red[:,0], red[:,1], c=ys, cmap="coolwarm", alpha=0.7, s=10)
        ax.set_title("PCA projection of learned embeddings (fallback)")
        plt.colorbar(sc, ax=ax, label="Label")
        ensure_dir(outdir)
        fig.savefig(os.path.join(outdir, f"{prefix}_pca.png"))
        plt.close(fig)
        return

    # run t-SNE with the chosen perplexity
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=int(perp), random_state=42)
        red = tsne.fit_transform(emb)
    except Exception as e:
        print(f"[WARN] t-SNE failed with error: {e}. Falling back to PCA.")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        red = pca.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(red[:,0], red[:,1], c=ys, cmap="coolwarm", alpha=0.7, s=10)
    ax.set_title(f"t-SNE of learned embeddings (n={n_pts}, perp={int(perp)})")
    plt.colorbar(sc, ax=ax, label="Label")
    ensure_dir(outdir)
    fig.savefig(os.path.join(outdir, f"{prefix}.png"))
    plt.close(fig)
    print(f"[INFO] Saved TSNE/PCA plot to {outdir}/{prefix}.png (n_pts={n_pts}, perp={int(perp)})")

# -------- Training loop --------
def train_model(X_train, y_train, X_val, y_val, input_dim, device, epochs=10, batch_size=64, alpha_final=1.0, outdir="results", lr=1e-3):
    model = ProjectionWithClassifier(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_ds = NumpyDataset(X_train, y_train)
    val_ds = NumpyDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    ensure_dir(outdir)
    best_val_f1 = -1.0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        alpha = alpha_final * (epoch / (epochs - 1)) if epochs > 1 else alpha_final
        # optional warmup: start alpha at 0 for first epoch(s)
        # contrastive_on = epoch >= 1
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            emb, logits = model(xb)
            loss, ce_val, supcon_val = hybrid_loss(emb, logits, yb, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1
        epoch_loss = epoch_loss / n_batch if n_batch > 0 else epoch_loss

        # Eval on validation
        metrics = evaluate_and_save(model, X_val, y_val, device=device, outdir=outdir, prefix=f"val_epoch{epoch+1}")
        # parse f1 from classification report
        # We'll compute macro avg F1 from report via sklearn again for reliability
        from sklearn.metrics import f1_score
        y_val_preds = metrics["preds"]
        y_val_true = metrics["labels"]
        val_f1 = f1_score(y_val_true, y_val_preds, average="macro")

        history.append({
            "epoch": epoch+1,
            "loss": epoch_loss,
            "alpha": alpha,
            "val_f1": val_f1,
            "val_roc": metrics["roc_auc"],
            "ece": metrics["ece"],
            "brier": metrics["brier"]
        })
        print(f"Epoch {epoch+1}/{epochs}  Loss: {epoch_loss:.4f}  alpha: {alpha:.2f}  val_f1: {val_f1:.4f}  val_roc: {metrics['roc_auc']:.4f}  ece: {metrics['ece']:.4f}")

        # save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(outdir, "best_model.pt"))

    # finally, load best for return
    model.load_state_dict(torch.load(os.path.join(outdir, "best_model.pt"), map_location=device))
    return model, history

# -------- CLI & main --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="CSV path (either numeric features f0.. or text + label)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0, help="Final alpha weight for contrastive loss")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Loading features from:", args.input)
    X, y = load_features_from_csv(args.input, batch_size=args.batch_size, device=device)
    print("Data shapes:", X.shape, y.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = safe_train_test_split(X, y, test_size=args.test_size, val_size=args.val_size, random_state=args.seed)
    print("Split sizes: train:", X_train.shape[0], "val:", X_val.shape[0], "test:", X_test.shape[0])

    model, history = train_model(X_train, y_train, X_val, y_val,
                                 input_dim=X.shape[1],
                                 device=device,
                                 epochs=args.epochs,
                                 batch_size=args.batch_size,
                                 alpha_final=args.alpha,
                                 outdir=args.outdir,
                                 lr=args.lr)

    # Final evaluation on test
    print("Evaluating on test set...")
    test_metrics = evaluate_and_save(model, X_test, y_test, device=device, outdir=args.outdir, prefix="test_final")
    # t-SNE plot of embeddings
    plot_tsne_embeddings(model, X, y, device=device, outdir=args.outdir, prefix="tsne_full", n_samples=2000)

    # Summarize history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.outdir, "train_history.csv"), index=False)
    print("Done. Results and plots saved to:", args.outdir)
    print("Final test ROC-AUC:", test_metrics["roc_auc"])
    print("Final test ECE:", test_metrics["ece"])
    print("Final test Brier:", test_metrics["brier"])
    print("Classification report (test):\n", test_metrics["report"])

if __name__ == "__main__":
    main()
