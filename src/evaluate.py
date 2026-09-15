from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dependency-light prediction audit")
    parser.add_argument("--predictions", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    confusion: dict[tuple[int, int], int] = defaultdict(int)
    absolute_error = 0
    count = 0
    with args.predictions.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            truth, prediction = int(row["target"]), int(row["prediction"])
            confusion[(truth, prediction)] += 1
            absolute_error += abs(truth - prediction)
            count += 1
    if count == 0:
        raise ValueError("Prediction file is empty")
    print(f"n={count} grade_index_mae={absolute_error / count:.4f}")
    print("confusion_entries=" + repr(dict(sorted(confusion.items()))))
    print("TODO: install scikit-learn to compute Macro-F1 and QWK for the paper table")


if __name__ == "__main__":
    main()
