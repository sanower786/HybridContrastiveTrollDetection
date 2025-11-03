import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from src.model_architecture import ProjectionWithClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_path, data_path):
    data = np.load(data_path)
    X_test, y_test = data['X'], data['y']
    model = ProjectionWithClassifier(input_dim=X_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
   # evaluate.py (or src/utils/metrics.py)

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

def evaluate_on_test(model, X_test, y_test, batch_size=64, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                       torch.tensor(y_test, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            if isinstance(out, tuple) or (isinstance(out, list) and len(out) == 2):
                emb, logits = out
            else:
                # if model returns embeddings only, compute logits via classifier
                emb = out
                logits = model.classifier(emb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # metrics
    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    try:
        roc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc = None

    # print human readable
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:\n", cm)
    print("ROC-AUC:", roc)

    return all_labels, all_preds, all_probs, {"report": report, "confusion_matrix": cm, "roc_auc": roc}

