"""
src/engine/engine_predictor.py
===============================
Responsabilidade única: gerar previsões usando o pipeline integrado
Mallows → Plackett–Luce.

Compartilhado entre Pipeline 1 e Pipeline 2.
"""

import numpy as np
from dataclasses import dataclass

from src.engine.engine_trainer        import ModelState
from src.models.models_mallows        import predict as mallows_predict, cluster_probabilities

CLUSTER_WEIGHT_DEFAULT: float = 0.7
MIN_CLUSTER_SIZE:       int   = 5


@dataclass
class Prediction:
    season:          int
    race:            str
    predicted_order: list[str]
    cluster_used:    int
    cluster_probs:   np.ndarray
    scores_combined: dict[str, float]


def predict(
    state:          ModelState,
    season:         int,
    race:           str,
    known_ranking:  list[str] | None = None,
    cluster_weight: float = CLUSTER_WEIGHT_DEFAULT,
) -> Prediction:
    """
    Gera previsão integrada para uma corrida.
    Mallows define o contexto, Plackett–Luce gera os scores.
    """
    if known_ranking is not None and len(known_ranking) >= 2:
        cluster_id  = mallows_predict(state.mallows, known_ranking, weight=1.0)
        clust_probs = cluster_probabilities(state.mallows, known_ranking, weight=1.0)
    else:
        cluster_id  = _cluster_by_circuit_history(state, race)
        clust_probs = np.zeros(state.n_clusters)
        clust_probs[cluster_id] = 1.0

    cluster_size = state.assignments.count(cluster_id)
    effective_cw = cluster_weight if cluster_size >= MIN_CLUSTER_SIZE else 0.3

    global_scores  = state.pl.global_scores
    cluster_scores = state.pl.cluster_scores.get(cluster_id, global_scores)

    combined = {
        d: (1 - effective_cw) * global_scores.get(d, 1e-9)
         + effective_cw       * cluster_scores.get(d, 1e-9)
        for d in state.all_drivers
    }

    total = sum(combined.values())
    if total > 1e-12:
        combined = {d: v / total for d, v in combined.items()}

    predicted_order = sorted(combined.keys(), key=lambda d: -combined[d])

    return Prediction(
        season          = season,
        race            = race,
        predicted_order = predicted_order,
        cluster_used    = cluster_id,
        cluster_probs   = clust_probs,
        scores_combined = combined,
    )


def _cluster_by_circuit_history(state: ModelState, race: str) -> int:
    """Cluster mais frequente do circuito no histórico."""
    circuit_clusters = [
        state.assignments[i]
        for i, r in enumerate(state.seen_records)
        if r.race == race
    ]
    if not circuit_clusters:
        return max(range(state.n_clusters),
                   key=lambda c: state.assignments.count(c))
    return max(set(circuit_clusters), key=circuit_clusters.count)
