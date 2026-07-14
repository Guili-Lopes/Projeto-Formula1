from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.experiments.artifact_store import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_writes_open_formats_without_overwriting_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ArtifactStore(
                artifact_root=Path(temp_dir),
                mode="validation",
                model_id="M0",
                seed=42,
                run_id="fixed_run",
            )
            config_path = store.write_yaml(
                "config_resolved.yaml",
                {"seed": 42},
            )
            summary_path = store.write_json(
                "metrics_summary.json",
                {"rps": 0.12},
            )
            written = store.write_dataframe(
                "race_metrics",
                pd.DataFrame({"race": ["Bahrain"], "rps": [0.12]}),
                csv_mirror=True,
                parquet_required=False,
            )
            pointer = store.update_latest_pointer({"rps": 0.12})

            self.assertTrue(config_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(written["csv"].is_file())
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8"))["rps"],
                0.12,
            )
            self.assertEqual(
                json.loads(pointer.read_text(encoding="utf-8"))["run_id"],
                "fixed_run",
            )

            with self.assertRaises(FileExistsError):
                ArtifactStore(
                    artifact_root=Path(temp_dir),
                    mode="validation",
                    model_id="M0",
                    seed=42,
                    run_id="fixed_run",
                )

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ArtifactStore(
                artifact_root=Path(temp_dir),
                mode="validation",
                model_id="M0",
                seed=42,
                run_id="fixed_run",
            )
            with self.assertRaises(ValueError):
                store.path("../outside.json")


if __name__ == "__main__":
    unittest.main()
