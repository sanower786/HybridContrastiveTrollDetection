# src/model_architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionWithClassifier(nn.Module):
    """
    Projection network with a classification head for the
    hybrid contrastive-classification framework.

    Input:
        772-dimensional hybrid feature vector
        (768-dimensional MPNet embedding + 4 auxiliary features)

    Outputs:
        - normalized projection embedding for supervised
          contrastive learning
        - class logits for cross-entropy classification
    """

    def __init__(
        self,
        input_dim=772,
        hidden_dim=256,
        projection_dim=128,
        num_classes=2
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(
            projection_dim,
            num_classes
        )

    def forward(
        self,
        x,
        return_embedding=False
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, 772].

        return_embedding : bool
            If True, return only the normalized projection
            embedding.

        Returns
        -------
        If return_embedding=True:
            normalized projection embedding

        Otherwise:
            (normalized embedding, class logits)
        """

        # Projection representation
        proj = self.proj(x)

        # Normalized representation used by
        # supervised contrastive learning
        emb = F.normalize(
            proj,
            dim=1
        )

        # Classification logits
        logits = self.classifier(proj)

        if return_embedding:
            return emb

        return emb, logits