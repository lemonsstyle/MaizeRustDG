from __future__ import annotations

from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .manifest import SampleRecord


class CornRustDataset(Dataset):
    """Dataset contract; weather loading is deliberately injected, not guessed."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        image_transform: Callable | None = None,
        weather_loader: Callable[[SampleRecord], tuple[torch.Tensor, torch.Tensor]] | None = None,
        validate: bool = True,
    ) -> None:
        self.records = list(records)
        self.image_transform = image_transform
        self.weather_loader = weather_loader
        if validate:
            for record in self.records:
                record.validate_for_training()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        image = self.image_transform(image) if self.image_transform else image

        if self.weather_loader and record.weather_path:
            weather, weather_mask = self.weather_loader(record)
            weather_missing = torch.tensor(False)
        else:
            weather = torch.zeros(1, 2, dtype=torch.float32)
            weather_mask = torch.zeros(1, dtype=torch.bool)
            weather_missing = torch.tensor(True)

        return {
            "image": image,
            "weather": weather,
            "weather_mask": weather_mask,
            "weather_missing": weather_missing,
            "target": torch.tensor(record.target, dtype=torch.long),
            "domain": record.site,
            "sample_id": record.sample_id,
        }
