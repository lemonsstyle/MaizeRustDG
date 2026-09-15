from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.data.splits import assert_group_isolation, leave_one_site_out


@dataclass(frozen=True)
class Record:
    site: str
    group_id: str


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            Record(site, f"{site}-g{group}")
            for site in ("Zhengzhou", "Changge", "Jiaozuo")
            for group in range(20)
            for _ in range(2)
        ]

    def test_target_site_is_test_only_and_groups_are_isolated(self) -> None:
        fold = leave_one_site_out(self.records, "Jiaozuo", validation_fraction=0.25, seed=42)
        self.assertTrue(all(self.records[i].site == "Jiaozuo" for i in fold.test))
        self.assertTrue(all(self.records[i].site != "Jiaozuo" for i in fold.train + fold.validation))
        assert_group_isolation(self.records, fold)

    def test_split_is_reproducible(self) -> None:
        first = leave_one_site_out(self.records, "Changge", seed=3407)
        second = leave_one_site_out(self.records, "Changge", seed=3407)
        self.assertEqual(first, second)

    def test_unknown_site_rejected(self) -> None:
        with self.assertRaises(ValueError):
            leave_one_site_out(self.records, "Unknown")


if __name__ == "__main__":
    unittest.main()
