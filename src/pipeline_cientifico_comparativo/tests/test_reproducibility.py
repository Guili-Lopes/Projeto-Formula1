from __future__ import annotations

import unittest

from src.experiments.reproducibility import derive_seed, set_global_seed
from src.pipeline_cientifico_comparativo.adapters.m0_adapter import (
    evaluation_years_for_mode,
)
from src.pipeline_cientifico_comparativo.config_loader import load_resolved_config


class ReproducibilityTests(unittest.TestCase):
    def test_derived_seed_is_repeatable_and_race_specific(self) -> None:
        first = derive_seed(42, 2024, "Bahrain")
        second = derive_seed(42, 2024, "Bahrain")
        other = derive_seed(42, 2024, "Australia")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_global_seed_audit(self) -> None:
        audit = set_global_seed(42)

        self.assertTrue(audit["python_random"])
        self.assertTrue(audit["numpy"])
        self.assertFalse(audit["torch"])

    def test_validation_never_loads_test_year(self) -> None:
        config = load_resolved_config(model_config="m0_baseline.yaml")
        loaded, warmup, evaluated = evaluation_years_for_mode(
            config,
            mode="validation",
            allow_test=False,
        )

        self.assertNotIn(2025, loaded)
        self.assertEqual(warmup, [])
        self.assertEqual(evaluated, [2024])

    def test_final_test_requires_explicit_unlock(self) -> None:
        config = load_resolved_config(model_config="m0_baseline.yaml")

        with self.assertRaises(PermissionError):
            evaluation_years_for_mode(
                config,
                mode="final_test",
                allow_test=False,
            )

    def test_unlocked_final_test_uses_validation_as_warmup(self) -> None:
        config = load_resolved_config(model_config="m0_baseline.yaml")
        loaded, warmup, evaluated = evaluation_years_for_mode(
            config,
            mode="final_test",
            allow_test=True,
        )

        self.assertEqual(loaded, [2019, 2020, 2021, 2022, 2023, 2024, 2025])
        self.assertEqual(warmup, [2024])
        self.assertEqual(evaluated, [2025])


if __name__ == "__main__":
    unittest.main()
