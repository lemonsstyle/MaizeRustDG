from __future__ import annotations

import torch
from torch import nn


class WeatherEncoder(nn.Module):
    """Mask-aware GRU baseline; replace with TST/InceptionTime in ablations."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, output_dim: int = 256) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        values = values * valid_mask.unsqueeze(-1).to(values.dtype)
        outputs, _ = self.gru(values)
        lengths = valid_mask.sum(dim=1).clamp_min(1) - 1
        batch = torch.arange(values.shape[0], device=values.device)
        return self.projection(outputs[batch, lengths])
