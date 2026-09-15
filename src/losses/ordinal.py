from __future__ import annotations

import torch
import torch.nn.functional as F


def ordinal_levels(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=target.device).unsqueeze(0)
    return (target.unsqueeze(1) > thresholds).to(torch.float32)


def coral_ordinal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    levels = ordinal_levels(target, logits.shape[1] + 1)
    return F.binary_cross_entropy_with_logits(logits, levels)
