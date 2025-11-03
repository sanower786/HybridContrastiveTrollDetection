# src/utils/losses.py

import torch
import torch.nn.functional as F

def supervised_contrastive_loss(embeddings, labels, temperature=0.5, eps=1e-8):
    """
    Simple supervised NT-Xent style loss.
    embeddings: normalized embeddings [B, D]
    labels: LongTensor [B]
    """
    device = embeddings.device
    sim = torch.matmul(embeddings, embeddings.T)  # [B, B]
    sim = sim / (temperature + eps)
    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)   # [B, B]
    mask = torch.eye(labels.size(0), dtype=torch.bool, device=device)
    labels_matrix = labels_matrix & ~mask

    # For each i, compute log(sum_{positives} exp(sim_i,j) / sum_{all != i} exp(sim_i,k))
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim))
    # avoid div by zero
    denom = denom + eps

    pos_sum = (exp_sim * labels_matrix.float()).sum(dim=1)
    # avoid log(0)
    pos_sum = pos_sum + eps
    loss = -torch.log(pos_sum / denom)
    return loss.mean()

def hybrid_loss_fn(xb, labels, model, alpha=0.5, temperature=0.5, contrastive_on=True):
    """
    xb: Tensor [B, D]
    labels: LongTensor [B]
    model: ProjectionWithClassifier instance
    alpha: weight for contrastive component (0..1)
    contrastive_on: whether to compute contrastive loss (for warmup scheduling)
    Returns: total_loss (Tensor), logits (Tensor)
    """
    device = xb.device
    # forward pass: model may return (emb, logits) or only emb if called with return_embedding
    out = model(xb)
    if isinstance(out, tuple) or (isinstance(out, list) and len(out) == 2):
        embeddings, logits = out
    else:
        # try calling model again to get logits
        # assume model(x) returned embeddings only if called with return_embedding=True
        # but that should not happen in training; handle gracefully:
        embeddings = out
        logits = model.classifier(embeddings) if hasattr(model, "classifier") else embeddings

    # classification loss
    cls_loss = F.cross_entropy(logits, labels)

    # contrastive loss
    if contrastive_on:
        # ensure embeddings are normalized
        embeddings = F.normalize(embeddings, dim=1)
        con_loss = supervised_contrastive_loss(embeddings, labels, temperature=temperature)
    else:
        con_loss = torch.tensor(0.0, device=device)

    total = (1.0 - alpha) * cls_loss + alpha * con_loss
    return total, logits
