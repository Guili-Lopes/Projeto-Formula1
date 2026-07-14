"""Standard result container for comparable pipeline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ExperimentResult:
    """Tables and objects produced by one model execution."""

    model_id: str
    mode: str
    seed: int
    summary: dict[str, Any]
    race_metrics: pd.DataFrame
    predictions: pd.DataFrame
    position_probabilities: pd.DataFrame
    parameter_history: pd.DataFrame
    state: Any
    extra_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def notebook_bundle(self, artifact_paths: dict[str, str]) -> dict[str, Any]:
        """Build a convenient, non-authoritative notebook bundle."""
        return {
            "model_id": self.model_id,
            "mode": self.mode,
            "seed": self.seed,
            "summary": self.summary,
            "race_metrics": self.race_metrics,
            "predictions": self.predictions,
            "position_probabilities": self.position_probabilities,
            "parameter_history": self.parameter_history,
            "extra_tables": self.extra_tables,
            "artifact_paths": artifact_paths,
        }
