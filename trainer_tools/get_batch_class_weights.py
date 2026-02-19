import torch

def get_batch_class_weights(
    labels,
    num_classes,
    device,
    prev_weights=None,
    smoothing_eps=1e-5,
    clamp_min=0.2,
    clamp_max=5.0,
    momentum=0.8
):
    # Count samples per class
    counts = torch.bincount(labels, minlength=num_classes).float()

    # Inverse frequency weighting
    weights = 1.0 / (counts + smoothing_eps)

    # Handle missing classes in batch
    weights[counts == 0] = 1.0 / (counts.sum() + smoothing_eps)

    # Normalize so mean weight = 1
    weights = weights / weights.sum() * num_classes

    # Manual class scaling (adjust if num_classes changes)
    mu_k = torch.tensor([0.9, 1.5, 1.0, 0.7, 1.5], device=device)
    weights = weights * mu_k

    # Clamp to avoid extreme weights
    weights = weights.clamp(min=clamp_min, max=clamp_max)

    # Momentum smoothing across batches
    if prev_weights is not None:
        weights = prev_weights * momentum + weights * (1 - momentum)

    return weights.to(device)
