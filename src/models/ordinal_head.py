from __future__ import annotations

import torch
from torch import nn


class CoralOrdinalHead(nn.Module):
    """K classes represented by K-1 cumulative thresholds."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.score = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features) + self.bias

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)
