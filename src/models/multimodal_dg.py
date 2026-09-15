from __future__ import annotations

import torch
from torch import nn

from .fusion import MissingAwareGatedFusion
from .ordinal_head import CoralOrdinalHead
from .weather_encoder import WeatherEncoder


class MultimodalDGModel(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        feature_dim: int = 256,
        weather_input_dim: int = 2,
        weather_hidden_dim: int = 128,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.weather_encoder = WeatherEncoder(weather_input_dim, weather_hidden_dim, feature_dim)
        self.fusion = MissingAwareGatedFusion(feature_dim)
        self.ordinal_head = CoralOrdinalHead(feature_dim, num_classes)
        self.auxiliary_head = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        image: torch.Tensor,
        weather: torch.Tensor,
        weather_mask: torch.Tensor,
        weather_missing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_features = self.image_encoder(image)
        weather_features = self.weather_encoder(weather, weather_mask)
        fused, gate = self.fusion(image_features, weather_features, weather_missing)
        return {
            "ordinal_logits": self.ordinal_head(fused),
            "auxiliary_logits": self.auxiliary_head(fused),
            "features": fused,
            "fusion_gate": gate,
        }
