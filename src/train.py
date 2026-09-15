from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the corn-rust multimodal DG model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--held-out-site", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    grade_order = [str(item) for item in config["data"]["grade_order"]]
    if grade_order != ["CK", "1", "3", "5", "7", "9"]:
        raise ValueError(f"Unexpected grade order: {grade_order}")
    if args.dry_run:
        print(f"Configuration valid; held-out target site={args.held_out_site}")
        return
    raise NotImplementedError(
        "Training loop intentionally waits for a verified manifest, weather field mapping, "
        "and an approved PyTorch environment. See docs/reproduction_plan.md."
    )


if __name__ == "__main__":
    main()
