"""
    Clustering de corridas via Modelo de Mallows
    e identificação do cluster de uma corrida nova.
"""

import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass

def kendall_distance(r1: list[str], r2: list[str]) -> int:
    set_r2 = set(r2)
    common = [x for x in r1 if x in set_r2]
    pos_r2 = {item: idx for idx, item in enumerate(r2)}
    pairs  = 0
    n      = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = common[i], common[j]
            if a in pos_r2 and b in pos_r2:
                if pos_r2[a] > pos_r2[b]:
                    pairs += 1
    return pairs

def _weighted_consensus(
    rankings:    list[list[str]],
    weights:     list[float],
    all_drivers: list[str],
) -> list[str]:
    scores = {d: 0.0 for d in all_drivers}
    n      = len(all_drivers)
    for ranking, w in zip(rankings, weights):
        for pos, driver in enumerate(ranking):
            scores[driver] += w * (n - pos)
    return sorted(all_drivers, key=lambda d: -scores[d])

def _gibbs_step(
    ranking:   list[str],
    weight:    float,
    consensos: list[list[str]],
    alpha:     float,
) -> int:
    distances  = np.array([kendall_distance(ranking, rho)
                           for rho in consensos], dtype=float)
    log_probs  = -alpha * weight * distances
    log_probs -= log_probs.max()
    probs      = np.exp(log_probs)
    probs     /= probs.sum()
    return int(np.random.choice(len(consensos), p=probs))

@dataclass
class MallowsModel:
    consensos:     list[list[str]]
    assignments:   list[int]
    cluster_sizes: list[int]
    all_drivers:   list[str]
    n_clusters:    int
    alpha:         float
    race_names:    list[str]

def fit(
    rankings:    list[list[str]],
    weights:     list[float],
    race_names:  list[str],
    all_drivers: list[str],
    n_clusters:  int   = 2,
    n_iter:      int   = 150,
    alpha:       float = 0.5,
    verbose:     bool  = True,
) -> MallowsModel:
    """Treina o modelo de clustering Mallows via MCMC (Algorithm 4)."""
    N           = len(rankings)
    assignments = np.random.choice(n_clusters, N).tolist()

    # Inicializar consensos
    consensos = []
    for c in range(n_clusters):
        idx_c = [i for i in range(N) if assignments[i] == c]
        if idx_c:
            rk = [rankings[i] for i in idx_c]
            wt = [weights[i]  for i in idx_c]
            consensos.append(_weighted_consensus(rk, wt, all_drivers))
        else:
            consensos.append(random.sample(all_drivers, len(all_drivers)))

    for iteration in range(n_iter):
        clusters    = defaultdict(list)
        cluster_wts = defaultdict(list)

        for i, (ranking, w) in enumerate(zip(rankings, weights)):
            c = _gibbs_step(ranking, w, consensos, alpha)
            assignments[i] = c
            clusters[c].append(ranking)
            cluster_wts[c].append(w)

        for c in range(n_clusters):
            if not clusters[c]:
                biggest = max(range(n_clusters), key=lambda x: len(clusters[x]))
                if clusters[biggest]:
                    clusters[c].append(clusters[biggest].pop())
                    cluster_wts[c].append(cluster_wts[biggest].pop())

        consensos = [
            _weighted_consensus(clusters[c], cluster_wts[c], all_drivers)
            if clusters[c]
            else random.sample(all_drivers, len(all_drivers))
            for c in range(n_clusters)
        ]

        if verbose and (iteration + 1) % 50 == 0:
            sizes = [assignments.count(c) for c in range(n_clusters)]
            print(f"    Iter {iteration+1:3d}/{n_iter} | Clusters: {sizes}")

    cluster_sizes = [assignments.count(c) for c in range(n_clusters)]

    return MallowsModel(
        consensos     = consensos,
        assignments   = assignments,
        cluster_sizes = cluster_sizes,
        all_drivers   = all_drivers,
        n_clusters    = n_clusters,
        alpha         = alpha,
        race_names    = race_names,
    )

def predict(
    model:   MallowsModel,
    ranking: list[str],
    weight:  float = 1.0,
) -> int:
    """Identifica o cluster mais provável de uma corrida nova."""
    distances  = np.array([kendall_distance(ranking, rho)
                           for rho in model.consensos], dtype=float)
    log_probs  = -model.alpha * weight * distances
    log_probs -= log_probs.max()
    probs      = np.exp(log_probs)
    probs     /= probs.sum()
    return int(np.argmax(probs))

def cluster_probabilities(
    model:   MallowsModel,
    ranking: list[str],
    weight:  float = 1.0,
) -> np.ndarray:
    distances  = np.array([kendall_distance(ranking, rho)
                           for rho in model.consensos], dtype=float)
    log_probs  = -model.alpha * weight * distances
    log_probs -= log_probs.max()
    probs      = np.exp(log_probs)
    return probs / probs.sum()