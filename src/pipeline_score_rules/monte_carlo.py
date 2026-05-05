"""
src/pipeline_score_rules/monte_carlo.py
=============================
Responsabilidade única: gerar o vetor de probabilidades de cada piloto
em cada posição via simulação Monte Carlo sobre os skill scores
do Plackett–Luce.
"""

import numpy as np
from dataclasses import dataclass


N_SIMULATIONS: int = 10_000


@dataclass
class ProbabilityVector:
    """Vetor de probabilidades de um piloto em cada posição."""
    driver:      str
    probs:       np.ndarray
    n_positions: int

    def p_top_n(self, n: int) -> float:
        """Probabilidade do piloto terminar entre os N primeiros."""
        return float(self.probs[:n].sum())

    def expected_position(self) -> float:
        """Posição esperada (média ponderada pelas probabilidades)."""
        positions = np.arange(1, self.n_positions + 1)
        return float(np.dot(positions, self.probs))


@dataclass
class RaceDistribution:
    """Distribuição completa de probabilidades de uma corrida."""
    season:  int
    race:    str
    cluster: int
    vectors: dict[str, ProbabilityVector]
    drivers: list[str]

    def get(self, driver: str) -> ProbabilityVector | None:
        return self.vectors.get(driver)

    def win_probabilities(self) -> dict[str, float]:
        """Retorna P(1º) de cada piloto, ordenado decrescente."""
        probs = {d: self.vectors[d].probs[0] for d in self.vectors}
        return dict(sorted(probs.items(), key=lambda x: -x[1]))


def simulate(
    skill_scores:  dict[str, float],
    season:        int,
    race:          str,
    cluster:       int,
    n_positions:   int = 10,
    n_simulations: int = N_SIMULATIONS,
    seed:          int | None = None,
) -> RaceDistribution:
    """
    Gera distribuição de probabilidades via Monte Carlo.

    Para cada simulação:
        1. Sorteia posição 1 com probabilidade ∝ λ_i
        2. Remove o piloto sorteado do pool
        3. Repete para posições 2, 3, ..., n_positions

    Com 10.000 simulações o erro de estimativa é ±1% por posição.
    """
    rng = np.random.default_rng(seed)

    drivers   = list(skill_scores.keys())
    n_drivers = len(drivers)
    lambdas   = np.array([skill_scores[d] for d in drivers])

    if lambdas.sum() > 1e-12:
        lambdas = lambdas / lambdas.sum()
    else:
        lambdas = np.ones(n_drivers) / n_drivers

    n_pos  = min(n_positions, n_drivers)
    counts = np.zeros((n_drivers, n_pos), dtype=np.int32)

    for _ in range(n_simulations):
        remaining_idx    = list(range(n_drivers))
        remaining_lambda = lambdas.copy()

        for pos in range(n_pos):
            w     = remaining_lambda[remaining_idx]
            w_sum = w.sum()
            if w_sum < 1e-12:
                break
            w = w / w_sum

            chosen_local  = rng.choice(len(remaining_idx), p=w)
            chosen_global = remaining_idx[chosen_local]
            counts[chosen_global, pos] += 1
            remaining_idx.pop(chosen_local)

    vectors: dict[str, ProbabilityVector] = {}
    for i, driver in enumerate(drivers):
        probs = counts[i] / n_simulations
        vectors[driver] = ProbabilityVector(
            driver=driver, probs=probs, n_positions=n_pos,
        )

    ordered_drivers = sorted(
        drivers, key=lambda d: vectors[d].expected_position()
    )

    return RaceDistribution(
        season=season, race=race, cluster=cluster,
        vectors=vectors, drivers=ordered_drivers,
    )


def uniform_baseline(
    drivers:     list[str],
    season:      int,
    race:        str,
    n_positions: int = 10,
) -> RaceDistribution:
    """
    Distribuição baseline uniforme — todos os pilotos com
    probabilidade igual em todas as posições (1 / n_drivers).

    Referência para avaliar se o modelo tem valor real:
    se o RPS do modelo for menor que o baseline, as probabilidades
    geradas têm valor informativo.
    """
    n_drivers = len(drivers)
    n_pos     = min(n_positions, n_drivers)
    uniform   = np.ones(n_pos) / n_drivers

    vectors = {
        d: ProbabilityVector(driver=d, probs=uniform.copy(), n_positions=n_pos)
        for d in drivers
    }

    return RaceDistribution(
        season=season, race=race, cluster=-1,
        vectors=vectors, drivers=list(drivers),
    )
