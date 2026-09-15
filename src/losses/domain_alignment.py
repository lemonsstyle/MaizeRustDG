from __future__ import annotations

from itertools import combinations

import torch


def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(features.shape[0] - 1, 1)


def deep_coral_alignment(features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    """Pairwise covariance alignment across source domains only."""
    losses = []
    for first, second in combinations(torch.unique(domains), 2):
        x = features[domains == first]
        y = features[domains == second]
        if x.shape[0] < 2 or y.shape[0] < 2:
            continue
        losses.append((_covariance(x) - _covariance(y)).square().mean())
    return torch.stack(losses).mean() if losses else features.sum() * 0.0
