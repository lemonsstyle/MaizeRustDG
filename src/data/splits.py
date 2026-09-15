from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence


class SplitRecord(Protocol):
    site: str
    group_id: str


@dataclass(frozen=True)
class Fold:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    held_out_site: str


def _stable_unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def leave_one_site_out(
    records: Sequence[SplitRecord],
    held_out_site: str,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> Fold:
    """Split by site, then assign complete source groups to train/validation.

    The held-out target site is never used to create or balance source splits.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if held_out_site not in {record.site for record in records}:
        raise ValueError(f"Unknown held-out site: {held_out_site}")

    source_groups: dict[str, list[int]] = defaultdict(list)
    test: list[int] = []
    for index, record in enumerate(records):
        if record.site == held_out_site:
            test.append(index)
        else:
            source_groups[record.group_id].append(index)

    train: list[int] = []
    validation: list[int] = []
    for group_id, indices in sorted(source_groups.items()):
        destination = validation if _stable_unit_interval(group_id, seed) < validation_fraction else train
        destination.extend(indices)

    if not train or not validation or not test:
        raise ValueError("Empty split; add groups or change validation_fraction/seed")
    return Fold(tuple(train), tuple(validation), tuple(test), held_out_site)


def assert_group_isolation(records: Sequence[SplitRecord], fold: Fold) -> None:
    partitions: Iterable[tuple[str, tuple[int, ...]]] = (
        ("train", fold.train), ("validation", fold.validation), ("test", fold.test)
    )
    owner: dict[str, str] = {}
    for partition, indices in partitions:
        for index in indices:
            group = records[index].group_id
            previous = owner.setdefault(group, partition)
            if previous != partition:
                raise AssertionError(f"Group {group!r} appears in {previous} and {partition}")
