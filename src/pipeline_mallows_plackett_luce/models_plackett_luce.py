"""
src/models/plackett_luce.py
===========================
Responsabilidade única: estimar skill scores via Plackett–Luce
com algoritmo MM ponderado. Expõe scores globais e por cluster.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class PLModel:
    global_scores:  dict[str, float]
    cluster_scores: dict[int, dict[str, float]]
    all_drivers:    list[str]
    n_races_seen:   int

def _mm_iteration(
    rankings:    list[list[str]],
    weights:     list[float],
    lambdas:     dict[str, float],
    all_drivers: list[str],
) -> dict[str, float]:
    numerators   = defaultdict(float)
    denominators = defaultdict(float)

    for ranking, w in zip(rankings, weights):
        if w < 1e-10:
            continue
        for k in range(len(ranking)):
            tail      = ranking[k:]
            denom_sum = sum(lambdas.get(d, 1e-9) for d in tail)
            if denom_sum < 1e-12:
                continue
            numerators[ranking[k]] += w
            for d in tail:
                denominators[d]    += w / denom_sum

    new_lambdas = {}
    for d in all_drivers:
        num = numerators.get(d, 0.0)
        den = denominators.get(d, 1e-12)
        new_lambdas[d] = num / den if den > 1e-12 else 1e-9

    total = sum(new_lambdas.values())
    return {d: v / total for d, v in new_lambdas.items()} if total > 1e-12 else new_lambdas

def estimate(
    rankings:     list[list[str]],
    weights:      list[float],
    all_drivers:  list[str],
    n_iter:       int = 200,
    init_lambdas: dict[str, float] | None = None,
) -> dict[str, float]:
    """Estima skill scores via algoritmo MM ponderado."""
    if init_lambdas is not None:
        lambdas = dict(init_lambdas)
        for d in all_drivers:
            if d not in lambdas:
                lambdas[d] = 1.0 / len(all_drivers)
    else:
        lambdas = {d: 1.0 for d in all_drivers}

    for _ in range(n_iter):
        lambdas = _mm_iteration(rankings, weights, lambdas, all_drivers)

    return lambdas

def build(
    rankings:     list[list[str]],
    weights:      list[float],
    assignments:  list[int],
    all_drivers:  list[str],
    n_clusters:   int,
    n_iter:       int = 200,
    prev_model:   'PLModel | None' = None,
) -> PLModel:
    """Constrói PLModel com scores globais e por cluster."""
    init_global   = prev_model.global_scores if prev_model else None
    global_scores = estimate(rankings, weights, all_drivers,
                             n_iter=n_iter, init_lambdas=init_global)

    cluster_scores: dict[int, dict[str, float]] = {}
    for c in range(n_clusters):
        idx_c = [i for i in range(len(rankings)) if assignments[i] == c]
        if not idx_c:
            cluster_scores[c] = dict(global_scores)
            continue
        rk_c   = [rankings[i] for i in idx_c]
        wt_c   = [weights[i]  for i in idx_c]
        init_c = prev_model.cluster_scores.get(c) if prev_model else None
        cluster_scores[c] = estimate(rk_c, wt_c, all_drivers,
                                     n_iter=n_iter, init_lambdas=init_c)

    return PLModel(
        global_scores  = global_scores,
        cluster_scores = cluster_scores,
        all_drivers    = all_drivers,
        n_races_seen   = len(rankings),
    )

def ranked_drivers(
    model:      PLModel,
    cluster_id: int | None = None,
) -> list[tuple[str, float]]:
    """Retorna pilotos ordenados por skill score."""
    scores = (model.cluster_scores.get(cluster_id, model.global_scores)
              if cluster_id is not None else model.global_scores)
    return sorted(scores.items(), key=lambda x: -x[1])