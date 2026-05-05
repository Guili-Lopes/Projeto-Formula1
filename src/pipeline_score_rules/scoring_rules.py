"""
src/pipeline2/scoring_rules.py
================================
Responsabilidade única: calcular o Ranked Probability Score (RPS)
e métricas de avaliação probabilística.

RPS = (1/K) × Σ_{k=1}^{K} (CDF_prevista(k) - CDF_real(k))²

    RPS = 0.0  → previsão perfeita
    RPS → 1.0  → previsão muito ruim
    RPS baseline uniforme ≈ 0.33 para 10 posições
"""

import numpy as np
from dataclasses import dataclass, field

from src.pipeline_score_rules.monte_carlo import RaceDistribution


@dataclass
class RPSResult:
    """RPS de uma corrida individual."""
    season:          int
    race:            str
    cluster:         int
    rps_model:       float
    rps_baseline:    float
    rps_per_driver:  dict[str, float]
    gain:            float


@dataclass
class RPSSummary:
    """Resumo do RPS agregado por temporada."""
    season:            int
    n_races:           int
    mean_rps_model:    float
    mean_rps_baseline: float
    mean_gain:         float
    per_race:          list[RPSResult] = field(default_factory=list)


def _build_cdf_real(actual_position: int, n_positions: int) -> np.ndarray:
    """CDF real: 0 antes da posição real, 1 a partir dela."""
    cdf = np.zeros(n_positions)
    if actual_position <= n_positions:
        cdf[actual_position - 1:] = 1.0
    return cdf


def rps_single_driver(
    predicted_probs: np.ndarray,
    actual_position: int,
) -> float:
    """
    RPS de um único piloto em uma corrida.

    predicted_probs : [P(1º), P(2º), ..., P(Kº)]
    actual_position : posição real (1-indexed). DNF → n_positions + 1
    """
    n        = len(predicted_probs)
    cdf_pred = np.cumsum(predicted_probs)
    cdf_real = _build_cdf_real(actual_position, n)
    return float(np.mean((cdf_pred - cdf_real) ** 2))


def compute_rps(
    distribution:   RaceDistribution,
    baseline:       RaceDistribution,
    actual_ranking: list[str],
) -> RPSResult:
    """
    Calcula o RPS da corrida para o modelo e para o baseline.

    Para cada piloto:
        1. Identifica sua posição real no ranking
        2. Calcula RPS individual com o vetor previsto
        3. Calcula RPS individual com o vetor baseline

    O RPS da corrida é a média dos RPS individuais.
    """
    rps_model_pd    = {}
    rps_baseline_pd = {}

    pos_map = {driver: pos + 1 for pos, driver in enumerate(actual_ranking)}

    for driver, vec in distribution.vectors.items():
        actual_pos = pos_map.get(driver, vec.n_positions + 1)

        rps_model_pd[driver] = rps_single_driver(vec.probs, actual_pos)

        base_vec = baseline.vectors.get(driver)
        if base_vec is not None:
            rps_baseline_pd[driver] = rps_single_driver(base_vec.probs, actual_pos)

    rps_model    = float(np.mean(list(rps_model_pd.values())))
    rps_baseline = float(np.mean(list(rps_baseline_pd.values()))) \
                   if rps_baseline_pd else 0.0
    gain         = rps_baseline - rps_model

    return RPSResult(
        season         = distribution.season,
        race           = distribution.race,
        cluster        = distribution.cluster,
        rps_model      = round(rps_model,    4),
        rps_baseline   = round(rps_baseline, 4),
        rps_per_driver = {d: round(v, 4) for d, v in rps_model_pd.items()},
        gain           = round(gain, 4),
    )


def rps_season_summary(results: list[RPSResult], season: int) -> RPSSummary:
    """Agrega o RPS de todas as corridas de uma temporada."""
    season_results = [r for r in results if r.season == season]
    if not season_results:
        return RPSSummary(season=season, n_races=0,
                          mean_rps_model=0.0, mean_rps_baseline=0.0,
                          mean_gain=0.0)
    return RPSSummary(
        season            = season,
        n_races           = len(season_results),
        mean_rps_model    = float(np.mean([r.rps_model    for r in season_results])),
        mean_rps_baseline = float(np.mean([r.rps_baseline for r in season_results])),
        mean_gain         = float(np.mean([r.gain         for r in season_results])),
        per_race          = season_results,
    )


def print_rps_table(results: list[RPSResult]) -> None:
    """Imprime tabela de RPS por corrida."""
    print(f"\n  {'Corrida':22s} {'Cluster':>8} {'RPS Modelo':>12} "
          f"{'RPS Baseline':>14} {'Ganho':>8}")
    print("  " + "-" * 68)
    for r in results:
        marker = " ✓" if r.gain > 0 else " ✗"
        print(f"  {r.race:22s} {r.cluster+1:>8d} {r.rps_model:>12.4f} "
              f"{r.rps_baseline:>14.4f} {r.gain:>8.4f}{marker}")


def print_rps_summary(val: RPSSummary, test: RPSSummary) -> None:
    """Comparação de RPS entre validação e teste."""
    print(f"\n  {'Split':20s} {'RPS Modelo':>12} {'RPS Baseline':>14} "
          f"{'Ganho Médio':>12} {'Corridas':>10}")
    print("  " + "-" * 72)
    print(f"  {'Validação ' + str(val.season):20s} {val.mean_rps_model:>12.4f} "
          f"{val.mean_rps_baseline:>14.4f} {val.mean_gain:>12.4f} "
          f"{val.n_races:>10}")
    print(f"  {'Teste ' + str(test.season):20s} {test.mean_rps_model:>12.4f} "
          f"{test.mean_rps_baseline:>14.4f} {test.mean_gain:>12.4f} "
          f"{test.n_races:>10}")
    print()
    pct_val  = (val.mean_gain  / val.mean_rps_baseline  * 100) \
               if val.mean_rps_baseline  > 0 else 0
    pct_test = (test.mean_gain / test.mean_rps_baseline * 100) \
               if test.mean_rps_baseline > 0 else 0
    print(f"  Melhoria sobre baseline:")
    print(f"    Validação {val.season}:  {pct_val:.1f}%")
    print(f"    Teste     {test.season}: {pct_test:.1f}%")
