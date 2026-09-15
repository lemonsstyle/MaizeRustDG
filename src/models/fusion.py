from __future__ import annotations

import torch
from torch import nn


class MissingAwareGatedFusion(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(feature_dim * 2 + 1, feature_dim), nn.Sigmoid())
        self.weather_projection = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self, image_features: torch.Tensor, weather_features: torch.Tensor, weather_missing: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = weather_missing.to(image_features.dtype).reshape(-1, 1)
        gate = self.gate(torch.cat((image_features, weather_features, missing), dim=1))
        gate = gate * (1.0 - missing)
        fused = self.norm(image_features + gate * self.weather_projection(weather_features))
        return fused, gate
