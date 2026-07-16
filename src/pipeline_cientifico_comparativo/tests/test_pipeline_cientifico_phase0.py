"""Testes rápidos e sintéticos da infraestrutura da Fase 0."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.engine.engine_predictor import predict as engine_predict
from src.engine.engine_trainer import initial_fit
from src.experiments.reproducibility import set_global_seed
from src.pipeline_cientifico_comparativo.adapters.m0_adapter import run_m0
from src.pipeline_cientifico_comparativo.config_loader import load_resolved_config


class PhaseZeroInfrastructureTests(unittest.TestCase):
    def test_engine_defaults_preserve_legacy_values(self) -> None:
        initial_signature = inspect.signature(initial_fit)
        predict_signature = inspect.signature(engine_predict)

        self.assertEqual(initial_signature.parameters["n_clusters"].default, 2)
        self.assertEqual(initial_signature.parameters["n_iter"].default, 150)
        self.assertEqual(initial_signature.parameters["alpha"].default, 0.5)
        self.assertEqual(initial_signature.parameters["pl_n_iter"].default, 200)
        self.assertEqual(
            predict_signature.parameters["cluster_weight"].default,
            0.7,
        )
        self.assertEqual(
            predict_signature.parameters["min_cluster_size"].default,
            5,
        )

    def test_m0_is_reproducible_on_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            drivers = [
                "Max Verstappen",
                "Lewis Hamilton",
                "Charles Leclerc",
                "Lando Norris",
            ]

            for year in range(2019, 2025):
                season_dir = data_dir / f"Season{year}"
                season_dir.mkdir(parents=True)
                rotation = (year - 2019) % len(drivers)
                ordered = drivers[rotation:] + drivers[:rotation]
                frame = pd.DataFrame(
                    {
                        "Track": [f"Race {year}"] * len(drivers),
                        "Position": [1, 2, 3, 4],
                        "Driver": ordered,
                    }
                )
                frame.to_csv(season_dir / "raceResults.csv", index=False)

            config = load_resolved_config(model_config="m0_baseline.yaml")
            config["_meta"]["data_dir_resolved"] = str(data_dir)
            config["data"]["top_k"] = 4
            config["mallows"]["n_iterations"] = 3
            config["mallows"]["refit_iterations"] = 3
            config["plackett_luce"]["n_iterations"] = 5
            config["simulation"]["n_simulations"] = 100
            config["simulation"]["n_positions"] = 4
            config["runtime"]["verbose"] = False

            set_global_seed(42)
            first = run_m0(config, mode="validation", allow_test=False)
            set_global_seed(42)
            second = run_m0(config, mode="validation", allow_test=False)

            self.assertEqual(first.summary["n_races"], 1)
            self.assertFalse(first.summary["test_years_loaded"])
            pd.testing.assert_frame_equal(
                first.race_metrics,
                second.race_metrics,
            )
            pd.testing.assert_frame_equal(
                first.position_probabilities,
                second.position_probabilities,
            )
            pd.testing.assert_frame_equal(
                first.parameter_history,
                second.parameter_history,
            )


if __name__ == "__main__":
    unittest.main()
