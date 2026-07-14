"""Shared prediction engine for Mallows + Plackett-Luce."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.engine.engine_trainer import ModelState
from src.models.models_mallows import cluster_probabilities, predict as mallows_predict


CLUSTER_WEIGHT_DEFAULT: float = 0.7
MIN_CLUSTER_SIZE: int = 5


@dataclass
class Prediction:
    season: int
    race: str
    predicted_order: list[str]
    cluster_used: int
    cluster_probs: np.ndarray
    scores_combined: dict[str, float]


def predict(
    state: ModelState,
    season: int,
    race: str,
    known_ranking: list[str] | None = None,
    cluster_weight: float = CLUSTER_WEIGHT_DEFAULT,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> Prediction:
    """Generate one integrated ranking prediction."""
    if not 0.0 <= cluster_weight <= 1.0:
        raise ValueError("cluster_weight must be between 0 and 1")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")

    if known_ranking is not None and len(known_ranking) >= 2:
        cluster_id = mallows_predict(state.mallows, known_ranking, weight=1.0)
        cluster_probs = cluster_probabilities(
            state.mallows,
            known_ranking,
            weight=1.0,
        )
    else:
        cluster_id = _cluster_by_circuit_history(state, race)
        cluster_probs = np.zeros(state.n_clusters)
        cluster_probs[cluster_id] = 1.0

    cluster_size = state.assignments.count(cluster_id)
    effective_weight = cluster_weight if cluster_size >= min_cluster_size else 0.3

    global_scores = state.pl.global_scores
    cluster_scores = state.pl.cluster_scores.get(cluster_id, global_scores)
    combined = {
        driver: (1 - effective_weight) * global_scores.get(driver, 1e-9)
        + effective_weight * cluster_scores.get(driver, 1e-9)
        for driver in state.all_drivers
    }

    total = sum(combined.values())
    if total > 1e-12:
        combined = {driver: value / total for driver, value in combined.items()}

    predicted_order = sorted(combined, key=lambda driver: -combined[driver])
    return Prediction(
        season=season,
        race=race,
        predicted_order=predicted_order,
        cluster_used=cluster_id,
        cluster_probs=cluster_probs,
        scores_combined=combined,
    )


def _cluster_by_circuit_history(state: ModelState, race: str) -> int:
    circuit_clusters = [
        state.assignments[i]
        for i, record in enumerate(state.seen_records)
        if record.race == race
    ]
    if not circuit_clusters:
        return max(
            range(state.n_clusters),
            key=lambda cluster_id: state.assignments.count(cluster_id),
        )
    return max(set(circuit_clusters), key=circuit_clusters.count)
