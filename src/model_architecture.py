import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionWithClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, projection_dim=128, num_classes=2):
        super(ProjectionWithClassifier, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        z_proj = F.normalize(self.projection_head(x), dim=-1)
        logits_cls = self.classifier_head(x)
        return z_proj, logits_cls
