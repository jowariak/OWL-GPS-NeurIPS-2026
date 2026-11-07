"""Two-head focal segmentation + Gaussian relevance KL from the research code."""
import torch
import torch.nn.functional as F


def focal_loss(logits, labels, *, weights=None, alpha=0.5, gamma=2.0, ignore_index=2):
    """Binary focal loss on labels 0/1; ignore pseudo/unknown class 2."""
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.shape != labels.shape:
        raise ValueError(f"logits {logits.shape} vs labels {labels.shape}")
    valid = (labels != ignore_index) & ((labels == 0) | (labels == 1))
    if not valid.any():
        return logits.sum() * 0.
    targets = labels[valid].float()
    bce = F.binary_cross_entropy_with_logits(logits[valid], targets, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = alpha * targets + (1-alpha) * (1-targets)
    loss = alpha_t * (1-pt).pow(gamma) * bce
    if weights is not None:
        if weights.shape != labels.shape:
            raise ValueError("weights must have label-mask shape")
        loss = loss * weights[valid]
    return loss.mean()


def gaussian_kl(mu, sigma):
    """KL[q=N(mu,diag(sigma**2)) || N(0,I)]."""
    sigma = sigma.float().clamp_min(1e-6)
    mu = mu.float()
    return (sigma.square() + mu.square() - 1 - 2*sigma.log()).sum(dim=-1).mul(.5).mean()


def combined_loss(logits_a, logits_b, labels, mu, sigma, *,
                  noise_weights=None, kl_weight=1e-4, alpha=0.5):
    return (focal_loss(logits_a, labels, weights=noise_weights, alpha=alpha)
            + focal_loss(logits_b, labels, weights=noise_weights, alpha=alpha)
            + kl_weight * gaussian_kl(mu, sigma))
