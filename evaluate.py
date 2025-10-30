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
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        _, logits = model(X_test)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    print(classification_report(y_test.cpu().numpy(), preds))
    cm = confusion_matrix(y_test.cpu().numpy(), preds)
    print("Confusion Matrix:\n", cm)
    print("ROC-AUC:", roc_auc_score(y_test.cpu().numpy(), probs))

    fpr, tpr, _ = roc_curve(y_test.cpu().numpy(), probs)
    plt.plot(fpr, tpr, label='Hybrid Model (AUC={:.3f})'.format(roc_auc_score(y_test.cpu().numpy(), probs)))
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC Curve")
    plt.show()

if __name__ == "__main__":
    evaluate_model("models/hybrid_best.pt", "data/processed_data.npz")
