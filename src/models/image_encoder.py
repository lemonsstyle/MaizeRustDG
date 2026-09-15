from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """Thin adapter around a backbone returning one vector per image."""

    def __init__(self, backbone: nn.Module, output_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(output_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU()
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.mean(dim=tuple(range(2, features.ndim)))
        return self.projection(features)
