from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


GRADE_ORDER = ("CK", "1", "3", "5", "7", "9")
GRADE_TO_INDEX = {grade: index for index, grade in enumerate(GRADE_ORDER)}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: Path
    site: str
    date: str
    plant_id: str
    plot_id: str
    grade: str
    weather_path: Path | None = None
    text_masked: bool = False
    label_verified: bool = False

    @property
    def target(self) -> int:
        try:
            return GRADE_TO_INDEX[str(self.grade)]
        except KeyError as exc:
            raise ValueError(f"Unsupported grade {self.grade!r}; expected {GRADE_ORDER}") from exc

    @property
    def group_id(self) -> str:
        return "|".join((self.site, self.date, self.plot_id, self.plant_id))

    @classmethod
    def from_mapping(cls, row: Mapping[str, str]) -> "SampleRecord":
        weather = row.get("weather_path", "").strip()
        return cls(
            sample_id=row["sample_id"].strip(),
            image_path=Path(row["image_path"].strip()),
            site=row["site"].strip(),
            date=row["date"].strip(),
            plant_id=row["plant_id"].strip(),
            plot_id=row["plot_id"].strip(),
            grade=row["grade"].strip(),
            weather_path=Path(weather) if weather else None,
            text_masked=row.get("text_masked", "false").lower() == "true",
            label_verified=row.get("label_verified", "false").lower() == "true",
        )

    def validate_for_training(self) -> None:
        if not self.image_path.exists():
            raise FileNotFoundError(self.image_path)
        if not self.text_masked:
            raise ValueError(f"{self.sample_id}: text_masked must be true before training")
        if not self.label_verified:
            raise ValueError(f"{self.sample_id}: label_verified must be true before training")
        _ = self.target
