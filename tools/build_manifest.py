from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FIELDS = (
    "sample_id", "image_path", "site", "date", "plant_id", "plot_id", "grade",
    "weather_path", "text_masked", "label_verified",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a conservative image audit manifest")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.image_root.resolve()
    if root.suffix.lower() == ".rar" or not root.is_dir():
        raise ValueError("--image-root must be the extracted image directory, never an archive")
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for path in images:
            relative = path.relative_to(root)
            sample_id = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:12]
            writer.writerow({
                "sample_id": sample_id,
                "image_path": str(path),
                "site": "", "date": "", "plant_id": "", "plot_id": "", "grade": "",
                "weather_path": "", "text_masked": "false", "label_verified": "false",
            })
    print(f"Wrote {len(images)} image rows to {args.output}")


if __name__ == "__main__":
    main()
