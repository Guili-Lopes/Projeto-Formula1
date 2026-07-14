"""Tabular artifact builders for the M0 baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.data_pipeline import RaceRecord
from src.engine.engine_predictor import Prediction
from src.engine.engine_trainer import ModelState
from src.pipeline_score_rules.monte_carlo import RaceDistribution


def parameter_rows(
    state: ModelState,
    *,
    model_id: str,
    mode: str,
    seed: int,
    season: int,
    race: str,
    race_index: int,
) -> list[dict[str, Any]]:
    """Build pre-prediction global and cluster skill snapshots."""
    rows: list[dict[str, Any]] = []
    common = {
        "model_id": model_id,
        "mode": mode,
        "seed": seed,
        "season": season,
        "race": race,
        "race_index": race_index,
        "snapshot_timing": "pre_prediction",
    }

    for driver, value in state.pl.global_scores.items():
        rows.append(
            {
                **common,
                "parameter_type": "driver_skill_global",
                "group": "global",
                "entity": driver,
                "value": float(value),
            }
        )

    for cluster_id, scores in state.pl.cluster_scores.items():
        for driver, value in scores.items():
            rows.append(
                {
                    **common,
                    "parameter_type": "driver_skill_cluster",
                    "group": str(cluster_id),
                    "entity": driver,
                    "value": float(value),
                }
            )
    return rows


def distribution_rows(
    distribution: RaceDistribution,
    *,
    model_id: str,
    mode: str,
    seed: int,
    race_index: int,
) -> list[dict[str, Any]]:
    """Build one row per driver, race and simulated finishing position."""
    rows: list[dict[str, Any]] = []
    for driver, vector in distribution.vectors.items():
        expected_position = vector.expected_position()
        p_win = float(vector.probs[0]) if len(vector.probs) else 0.0
        p_top3 = vector.p_top_n(3)
        p_top5 = vector.p_top_n(5)
        p_top10 = vector.p_top_n(10)
        for position, probability in enumerate(vector.probs, start=1):
            rows.append(
                {
                    "model_id": model_id,
                    "mode": mode,
                    "seed": seed,
                    "season": distribution.season,
                    "race": distribution.race,
                    "race_index": race_index,
                    "cluster": distribution.cluster,
                    "driver": driver,
                    "position": position,
                    "probability": float(probability),
                    "expected_position": expected_position,
                    "p_win": p_win,
                    "p_top3": p_top3,
                    "p_top5": p_top5,
                    "p_top10": p_top10,
                }
            )
    return rows


def prediction_rows(
    prediction: Prediction,
    record: RaceRecord,
    distribution: RaceDistribution,
    *,
    model_id: str,
    mode: str,
    seed: int,
    race_index: int,
) -> list[dict[str, Any]]:
    """Build one row per predicted driver for a race."""
    actual_position = {
        driver: position for position, driver in enumerate(record.ranking, start=1)
    }
    rows: list[dict[str, Any]] = []
    for predicted_position, driver in enumerate(prediction.predicted_order, start=1):
        vector = distribution.vectors.get(driver)
        observed_position = actual_position.get(driver)
        rows.append(
            {
                "model_id": model_id,
                "mode": mode,
                "seed": seed,
                "season": record.season,
                "race": record.race,
                "race_index": race_index,
                "driver": driver,
                "predicted_position": predicted_position,
                "actual_position": observed_position,
                "in_observed_partial_ranking": observed_position is not None,
                "predicted_score": float(
                    prediction.scores_combined.get(driver, 0.0)
                ),
                "expected_position": (
                    vector.expected_position() if vector is not None else np.nan
                ),
                "p_win": (
                    float(vector.probs[0])
                    if vector is not None and len(vector.probs)
                    else np.nan
                ),
                "p_top3": vector.p_top_n(3) if vector is not None else np.nan,
                "p_top5": vector.p_top_n(5) if vector is not None else np.nan,
            }
        )
    return rows
