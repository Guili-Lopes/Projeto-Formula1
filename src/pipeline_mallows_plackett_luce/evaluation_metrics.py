"""
    Calcular e agregar métricas de avaliação.
"""

import numpy as np
from dataclasses import dataclass, field

@dataclass
class RaceEval:
    season:       int
    race:         str
    predicted:    list[str]
    actual:       list[str]
    cluster_used: int
    top3_acc:     float
    top5_acc:     float
    kendall_tau:  float

@dataclass
class SeasonSummary:
    season:       int
    n_races:      int
    mean_top3:    float
    mean_top5:    float
    mean_kendall: float
    per_race:     list[RaceEval] = field(default_factory=list)

def top_n_accuracy(predicted: list[str], actual: list[str], n: int) -> float:
    pred_top = set(predicted[:n])
    real_top = set(actual[:n])
    if not real_top:
        return 0.0
    return len(pred_top & real_top) / n

def kendall_tau(predicted: list[str], actual: list[str]) -> float:
    common  = [x for x in predicted if x in actual]
    pos_act = {item: idx for idx, item in enumerate(actual)}
    n       = len(common)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = common[i], common[j]
            if a in pos_act and b in pos_act:
                if pos_act[a] < pos_act[b]:
                    concordant += 1
                else:
                    discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0

def evaluate_race(
    season: int, race: str,
    predicted: list[str], actual: list[str],
    cluster_used: int,
) -> RaceEval:
    return RaceEval(
        season       = season,
        race         = race,
        predicted    = predicted,
        actual       = actual,
        cluster_used = cluster_used,
        top3_acc     = top_n_accuracy(predicted, actual, n=3),
        top5_acc     = top_n_accuracy(predicted, actual, n=5),
        kendall_tau  = kendall_tau(predicted, actual),
    )

def season_summary(evals: list[RaceEval], season: int) -> SeasonSummary:
    season_evals = [e for e in evals if e.season == season]
    if not season_evals:
        return SeasonSummary(season=season, n_races=0,
                             mean_top3=0.0, mean_top5=0.0, mean_kendall=0.0)
    return SeasonSummary(
        season       = season,
        n_races      = len(season_evals),
        mean_top3    = float(np.mean([e.top3_acc    for e in season_evals])),
        mean_top5    = float(np.mean([e.top5_acc    for e in season_evals])),
        mean_kendall = float(np.mean([e.kendall_tau for e in season_evals])),
        per_race     = season_evals,
    )

def print_race_table(evals: list[RaceEval]) -> None:
    print(f"\n  {'Corrida':22s} {'Cluster':>8} {'Top-3':>8} "
          f"{'Top-5':>8} {'Kendall τ':>10}")
    print("  " + "-" * 60)
    for e in evals:
        print(f"  {e.race:22s} {e.cluster_used+1:>8d} {e.top3_acc:>8.3f} "
              f"{e.top5_acc:>8.3f} {e.kendall_tau:>10.3f}")

def print_comparison(val: SeasonSummary, test: SeasonSummary) -> None:
    print(f"\n  {'Split':20s} {'Top-3':>8} {'Top-5':>8} "
          f"{'Kendall τ':>10} {'Corridas':>10}")
    print("  " + "-" * 60)
    print(f"  {'Validação ' + str(val.season):20s} {val.mean_top3:>8.3f} "
          f"{val.mean_top5:>8.3f} {val.mean_kendall:>10.3f} {val.n_races:>10}")
    print(f"  {'Teste ' + str(test.season):20s} {test.mean_top3:>8.3f} "
          f"{test.mean_top5:>8.3f} {test.mean_kendall:>10.3f} {test.n_races:>10}")