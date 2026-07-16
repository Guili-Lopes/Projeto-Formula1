from __future__ import annotations

import unittest

from src.evaluation.additional_metrics import (
    mean_absolute_position_error,
    winner_correct,
)


class AdditionalMetricsTests(unittest.TestCase):
    def test_winner_correct(self) -> None:
        self.assertTrue(winner_correct(["VER", "NOR"], ["VER", "LEC"]))
        self.assertFalse(winner_correct(["NOR", "VER"], ["VER", "NOR"]))
        self.assertFalse(winner_correct([], ["VER"]))

    def test_mean_absolute_position_error(self) -> None:
        value = mean_absolute_position_error(
            ["VER", "NOR", "LEC"],
            ["VER", "LEC", "NOR"],
        )
        self.assertAlmostEqual(value, 2 / 3)

    def test_position_error_uses_only_common_drivers(self) -> None:
        value = mean_absolute_position_error(
            ["VER", "NOR", "LEC"],
            ["VER", "PIA"],
        )
        self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
