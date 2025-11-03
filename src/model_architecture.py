# src/model_architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionWithClassifier(nn.Module):
    """
    Lightweight projection head + classifier.
    Forward behavior:
      - forward(x) -> (emb, logits) where emb is normalized projection
      - forward(x, return_embedding=True) -> emb (normalized)
    """
    def __init__(self, input_dim=772, hidden_dim=256, projection_dim=128, num_classes=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, x, return_embedding=False):
        """
        x: Tensor [B, input_dim]
        return_embedding: if True, return projection embedding only
        Returns:
          (emb_normalized, logits)  OR emb_normalized (if return_embedding=True)
        """
        proj = self.proj(x)
        emb = F.normalize(proj, dim=1)   # normalized embedding for contrastive
        logits = self.classifier(proj)   # classifier from projection space (not normalized)
        if return_embedding:
            return emb
        return emb, logits
