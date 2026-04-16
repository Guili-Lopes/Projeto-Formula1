"""
    Arquestrar o treino inicial e incremental
    do pipeline integrado Mallows → Plackett–Luce.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from collections import defaultdict

from data_pipeline        import RaceRecord
from models_weights       import compute as compute_weights
from models_mallows       import MallowsModel, fit as mallows_fit, predict as mallows_predict
from models_plackett_luce import PLModel, build as pl_build, ranked_drivers

@dataclass
class ModelState:
    """Estado completo do pipeline integrado."""
    mallows:        MallowsModel
    pl:             PLModel
    seen_records:   list[RaceRecord]
    seen_weights:   list[float]
    assignments:    list[int]
    n_clusters:     int
    alpha:          float
    all_drivers:    list[str]
    current_season: int = 0

def initial_fit(
    records:     list[RaceRecord],
    all_drivers: list[str],
    n_clusters:  int   = 2,
    n_iter:      int   = 150,
    alpha:       float = 0.5,
    verbose:     bool  = True,
) -> ModelState:
    """Treino inicial do pipeline integrado sobre o conjunto de base."""
    seasons    = [r.season  for r in records]
    races      = [r.race    for r in records]
    rankings   = [r.ranking for r in records]

    weight_objs = compute_weights(seasons, races)
    weights     = [w.final_weight for w in weight_objs]

    if verbose:
        print(f"\n{'=' * 60}")
        print("TREINO INICIAL")
        print(f"  Corridas: {len(records)} | Pilotos: {len(all_drivers)} | "
              f"Clusters: {n_clusters}")
        from collections import Counter
        for s, cnt in sorted(Counter(seasons).items()):
            ws = [w for w, se in zip(weights, seasons) if se == s]
            print(f"  {s}: {cnt} corridas | peso médio: {np.mean(ws):.4f}")
        print("=" * 60)

    if verbose:
        print("\n[1/2] Mallows clustering...")
    mallows_model = mallows_fit(
        rankings=rankings, weights=weights, race_names=races,
        all_drivers=all_drivers, n_clusters=n_clusters,
        n_iter=n_iter, alpha=alpha, verbose=verbose,
    )
    assignments = list(mallows_model.assignments)

    if verbose:
        print("\n[2/2] Plackett–Luce ponderado...")
    pl_model = pl_build(
        rankings=rankings, weights=weights, assignments=assignments,
        all_drivers=all_drivers, n_clusters=n_clusters, n_iter=200,
    )

    if verbose:
        print("\nSkill scores globais após treino (Top 10):")
        for i, (d, s) in enumerate(ranked_drivers(pl_model)[:10], 1):
            bar = '█' * int(s * 400)
            print(f"  {i:2d}. {d}  {s:.4f}  {bar}")
        print("\nClusters identificados:")
        for c in range(n_clusters):
            names = [races[i] for i, a in enumerate(assignments) if a == c]
            print(f"  Cluster {c+1} ({mallows_model.cluster_sizes[c]} corridas): "
                  f"{' > '.join(mallows_model.consensos[c][:8])}")
            print(f"    Corridas: {names[:5]}{'...' if len(names) > 5 else ''}")

    return ModelState(
        mallows        = mallows_model,
        pl             = pl_model,
        seen_records   = list(records),
        seen_weights   = weights,
        assignments    = assignments,
        n_clusters     = n_clusters,
        alpha          = alpha,
        all_drivers    = all_drivers,
        current_season = seasons[-1] if seasons else 0,
    )

def incremental_update(
    state:         ModelState,
    new_record:    RaceRecord,
    refit_mallows: bool = False,
    verbose:       bool = False,
) -> ModelState:

    """
    Incorpora uma nova corrida ao modelo e retreina.

    Fluxo:
        1. Mallows identifica o cluster da nova corrida
        2. Corrida é adicionada ao histórico
        3. Pesos regulatórios são recalculados (nova corrida = decay 1.0)
        4. Plackett–Luce retreina com histórico atualizado
        5. Mallows retreina se refit_mallows=True (virada de temporada)
    """

    # 1. Identificar cluster via Mallows
    cluster_new = mallows_predict(
        model=state.mallows, ranking=new_record.ranking, weight=1.0
    )

    if verbose:
        print(f"    {new_record.race}: → Cluster {cluster_new + 1}")

    # 2. Atualizar histórico
    new_records     = state.seen_records + [new_record]
    new_assignments = state.assignments  + [cluster_new]

    # 3. Recalcular pesos — nova corrida passa a ser a mais recente
    seasons  = [r.season  for r in new_records]
    races    = [r.race    for r in new_records]
    rankings = [r.ranking for r in new_records]

    weight_objs = compute_weights(seasons, races)
    new_weights = [w.final_weight for w in weight_objs]

    # 4. Retreinar Mallows se solicitado (virada de temporada)
    if refit_mallows:
        new_mallows = mallows_fit(
            rankings=rankings, weights=new_weights, race_names=races,
            all_drivers=state.all_drivers, n_clusters=state.n_clusters,
            n_iter=150, alpha=state.alpha, verbose=False,
        )
        new_assignments = list(new_mallows.assignments)
    else:
        new_mallows = state.mallows

    # 5. Retreinar Plackett–Luce (sempre, a cada corrida)
    new_pl = pl_build(
        rankings=rankings, weights=new_weights, assignments=new_assignments,
        all_drivers=state.all_drivers, n_clusters=state.n_clusters,
        n_iter=200, prev_model=state.pl,
    )

    return ModelState(
        mallows        = new_mallows,
        pl             = new_pl,
        seen_records   = new_records,
        seen_weights   = new_weights,
        assignments    = new_assignments,
        n_clusters     = state.n_clusters,
        alpha          = state.alpha,
        all_drivers    = state.all_drivers,
        current_season = new_record.season,
    )