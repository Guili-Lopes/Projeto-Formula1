"""
    Calcular o peso regulatório de cada corrida.
    w_i = era_weight(season_i) × exp(-λ × distância_até_corrida_mais_recente)
"""

import numpy as np
from dataclasses import dataclass

ERA_WEIGHTS: dict[int, float] = {
    2019: 0.40,
    2020: 0.40,
    2021: 0.50,
    2022: 1.00,
    2023: 1.00,
    2024: 1.00,
    2025: 1.00,
}

LAMBDA_DECAY: float = 0.015

@dataclass
class RaceWeight:
    season:       int
    race:         str
    era_weight:   float
    decay:        float
    final_weight: float

def compute(
    seasons:      list[int],
    races:        list[str],
    lambda_decay: float = LAMBDA_DECAY,
) -> list[RaceWeight]:
    """
    Calcula pesos regulatórios para uma lista de corridas.
    A corrida mais recente sempre recebe decay = 1.0.
    """
    assert len(seasons) == len(races)
    n      = len(seasons)
    result = []

    for idx, (season, race) in enumerate(zip(seasons, races)):
        era_w         = ERA_WEIGHTS.get(season, 1.0)
        dist_from_end = (n - 1) - idx
        decay         = float(np.exp(-lambda_decay * dist_from_end))
        final         = round(era_w * decay, 6)

        result.append(RaceWeight(
            season       = season,
            race         = race,
            era_weight   = era_w,
            decay        = round(decay, 6),
            final_weight = final,
        ))

    return result

def as_array(race_weights: list[RaceWeight]) -> np.ndarray:
    return np.array([rw.final_weight for rw in race_weights])