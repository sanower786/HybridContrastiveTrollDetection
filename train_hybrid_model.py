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
                _, logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        print(classification_report(all_labels, all_preds))
    return model
