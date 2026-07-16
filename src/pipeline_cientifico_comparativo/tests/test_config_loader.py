from __future__ import annotations

import unittest
from pathlib import Path

from src.pipeline_cientifico_comparativo.config_loader import (
    deep_merge,
    load_resolved_config,
)
from src.pipeline_cientifico_comparativo.config_schema import (
    ConfigError,
    validate_config,
)


class ConfigLoaderTests(unittest.TestCase):
    def test_deep_merge_preserves_base_and_overrides_nested_value(self) -> None:
        base = {"a": {"b": 1, "c": 2}, "items": [1, 2]}
        override = {"a": {"b": 9}, "items": [3]}
        merged = deep_merge(base, override)

        self.assertEqual(merged, {"a": {"b": 9, "c": 2}, "items": [3]})
        self.assertEqual(base["a"]["b"], 1)

    def test_real_phase0_config_loads_and_accepts_cli_overrides(self) -> None:
        config = load_resolved_config(
            model_config="m0_baseline.yaml",
            seed_override=123,
            artifact_root_override="artifacts/test_override",
        )

        self.assertEqual(config["experiment"]["model_id"], "M0")
        self.assertEqual(config["experiment"]["random_seed"], 123)
        self.assertEqual(config["mallows"]["n_clusters"], 2)
        self.assertEqual(config["mallows"]["refit_iterations"], 150)
        self.assertTrue(
            Path(config["_meta"]["artifact_root_resolved"])
            .as_posix()
            .endswith("artifacts/test_override")
        )

    def test_overlapping_splits_are_rejected(self) -> None:
        config = load_resolved_config(model_config="m0_baseline.yaml")
        config["data"]["test_years"] = [2024]

        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_invalid_score_source_is_rejected(self) -> None:
        config = load_resolved_config(model_config="m0_baseline.yaml")
        config["simulation"]["score_source"] = "unknown"

        with self.assertRaises(ConfigError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
