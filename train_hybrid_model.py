import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import numpy as np
from src.model_architecture import ProjectionWithClassifier
from src.utils.losses import hybrid_loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_hybrid_model(X_train, y_train, X_val, y_val, epochs=10):
    model = ProjectionWithClassifier(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                            torch.tensor(y_train, dtype=torch.long)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                          torch.tensor(y_val, dtype=torch.long)), batch_size=64, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        alpha = min(1.0, epoch / epochs)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss, logits = hybrid_loss_fn(xb, yb, model, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, α={alpha:.2f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, logits = model(xb)# train_hybrid_model.py (update part that defines train_hybrid_model function)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
import os

from src.model_architecture import ProjectionWithClassifier
from src.utils.losses import hybrid_loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_hybrid_model(X_train, y_train, X_val, y_val,
                       epochs=10, batch_size=64, lr=1e-3, alpha_final=1.0,
                       warmup_epochs=2, input_dim=None, results_dir="results", save_name="hybrid_best.pt"):
    os.makedirs(results_dir, exist_ok=True)
    if input_dim is None:
        input_dim = X_train.shape[1]
    model = ProjectionWithClassifier(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_f1 = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        alpha = min(alpha_final, alpha_final * (epoch / max(1, epochs - 1)))
        contrastive_on = epoch >= warmup_epochs

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss, logits = hybrid_loss_fn(xb, yb, model, alpha=alpha, contrastive_on=contrastive_on)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(train_loader))
        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                emb, logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(yb.cpu().numpy().tolist())

        val_f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, α: {alpha:.2f}  Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(results_dir, save_name))
            print("Saved best model.")

    return model, {"best_epoch": best_epoch, "best_val_f1": best_val_f1}

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        print(classification_report(all_labels, all_preds))
    return model
