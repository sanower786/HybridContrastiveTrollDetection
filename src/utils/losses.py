import torch
import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, temperature=0.5):
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    exp_logits = torch.exp(logits) * (1 - torch.eye(features.shape[0], device=features.device))
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = -mean_log_prob_pos.mean()
    return loss

def hybrid_loss_fn(xb, yb, model, alpha=0.5):
    z_proj, logits_cls = model(xb)
    loss_ce = F.cross_entropy(logits_cls, yb)
    loss_con = supervised_contrastive_loss(z_proj, yb)
    total_loss = (1 - alpha) * loss_ce + alpha * loss_con
    return total_loss, logits_cls
