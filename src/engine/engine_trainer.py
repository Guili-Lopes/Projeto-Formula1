"""
src/engine/engine_trainer.py
============================
Responsabilidade única: orquestrar treino inicial e incremental
do pipeline integrado Mallows -> Plackett-Luce.

Compartilhado entre os pipelines. Os parâmetros adicionados nesta versão
mantêm os mesmos valores padrão usados anteriormente, preservando a
compatibilidade com os entry points existentes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.data.data_pipeline import RaceRecord
from src.models.models_mallows import (
    MallowsModel,
    fit as mallows_fit,
    predict as mallows_predict,
)
from src.models.models_plackett_luce import (
    PLModel,
    build as pl_build,
    ranked_drivers,
)
from src.models.models_weights import compute as compute_weights


@dataclass
class ModelState:
    """Estado completo do pipeline integrado."""

    mallows: MallowsModel
    pl: PLModel
    seen_records: list[RaceRecord]
    seen_weights: list[float]
    assignments: list[int]
    n_clusters: int
    alpha: float
    all_drivers: list[str]
    current_season: int = 0
    mallows_n_iter: int = 150
    pl_n_iter: int = 200


def initial_fit(
    records: list[RaceRecord],
    all_drivers: list[str],
    n_clusters: int = 2,
    n_iter: int = 150,
    alpha: float = 0.5,
    verbose: bool = True,
    *,
    pl_n_iter: int = 200,
    refit_mallows_n_iter: int | None = None,
) -> ModelState:
    """Treino inicial do pipeline integrado sobre o conjunto de base.

    Parameters added for the scientific pipeline are backward compatible:
    ``pl_n_iter`` defaults to 200 and the Mallows refit defaults to ``n_iter``.
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters must be greater than zero")
    if n_iter <= 0:
        raise ValueError("n_iter must be greater than zero")
    if pl_n_iter <= 0:
        raise ValueError("pl_n_iter must be greater than zero")
    if refit_mallows_n_iter is not None and refit_mallows_n_iter <= 0:
        raise ValueError("refit_mallows_n_iter must be greater than zero")

    mallows_refit_iterations = refit_mallows_n_iter or n_iter
    seasons = [record.season for record in records]
    races = [record.race for record in records]
    rankings = [record.ranking for record in records]

    weight_objs = compute_weights(seasons, races)
    weights = [weight.final_weight for weight in weight_objs]

    if verbose:
        print(f"\n{'=' * 60}")
        print("TREINO INICIAL")
        print(
            f"  Corridas: {len(records)} | Pilotos: {len(all_drivers)} | "
            f"Clusters: {n_clusters}"
        )
        from collections import Counter

        for season, count in sorted(Counter(seasons).items()):
            season_weights = [
                weight
                for weight, record_season in zip(weights, seasons)
                if record_season == season
            ]
            print(
                f"  {season}: {count} corridas | "
                f"peso médio: {np.mean(season_weights):.4f}"
            )
        print("=" * 60)

    if verbose:
        print("\n[1/2] Mallows clustering...")
    mallows_model = mallows_fit(
        rankings=rankings,
        weights=weights,
        race_names=races,
        all_drivers=all_drivers,
        n_clusters=n_clusters,
        n_iter=n_iter,
        alpha=alpha,
        verbose=verbose,
    )
    assignments = list(mallows_model.assignments)

    if verbose:
        print("\n[2/2] Plackett-Luce ponderado...")
    pl_model = pl_build(
        rankings=rankings,
        weights=weights,
        assignments=assignments,
        all_drivers=all_drivers,
        n_clusters=n_clusters,
        n_iter=pl_n_iter,
    )

    if verbose:
        print("\nSkill scores globais após treino (Top 10):")
        for index, (driver, score) in enumerate(ranked_drivers(pl_model)[:10], 1):
            bar = "█" * int(score * 400)
            print(f"  {index:2d}. {driver}  {score:.4f}  {bar}")
        print("\nClusters identificados:")
        for cluster in range(n_clusters):
            names = [
                races[index]
                for index, assignment in enumerate(assignments)
                if assignment == cluster
            ]
            print(
                f"  Cluster {cluster + 1} "
                f"({mallows_model.cluster_sizes[cluster]} corridas): "
                f"{' > '.join(mallows_model.consensos[cluster][:8])}"
            )
            suffix = "..." if len(names) > 5 else ""
            print(f"    Corridas: {names[:5]}{suffix}")

    return ModelState(
        mallows=mallows_model,
        pl=pl_model,
        seen_records=list(records),
        seen_weights=weights,
        assignments=assignments,
        n_clusters=n_clusters,
        alpha=alpha,
        all_drivers=all_drivers,
        current_season=seasons[-1] if seasons else 0,
        mallows_n_iter=mallows_refit_iterations,
        pl_n_iter=pl_n_iter,
    )


def incremental_update(
    state: ModelState,
    new_record: RaceRecord,
    refit_mallows: bool = False,
    verbose: bool = False,
) -> ModelState:
    """Incorpora uma nova corrida ao modelo e retreina.

    Fluxo:
        1. Mallows identifica o cluster da nova corrida
        2. Corrida é adicionada ao histórico
        3. Pesos regulatórios são recalculados
        4. Plackett-Luce retreina com histórico atualizado
        5. Mallows retreina se refit_mallows=True
    """
    cluster_new = mallows_predict(
        model=state.mallows,
        ranking=new_record.ranking,
        weight=1.0,
    )

    if verbose:
        print(f"    {new_record.race}: -> Cluster {cluster_new + 1}")

    new_records = state.seen_records + [new_record]
    new_assignments = state.assignments + [cluster_new]

    seasons = [record.season for record in new_records]
    races = [record.race for record in new_records]
    rankings = [record.ranking for record in new_records]

    weight_objs = compute_weights(seasons, races)
    new_weights = [weight.final_weight for weight in weight_objs]

    # getattr keeps old pickled ModelState objects usable after this change.
    mallows_n_iter = int(getattr(state, "mallows_n_iter", 150))
    pl_n_iter = int(getattr(state, "pl_n_iter", 200))

    if refit_mallows:
        new_mallows = mallows_fit(
            rankings=rankings,
            weights=new_weights,
            race_names=races,
            all_drivers=state.all_drivers,
            n_clusters=state.n_clusters,
            n_iter=mallows_n_iter,
            alpha=state.alpha,
            verbose=False,
        )
        new_assignments = list(new_mallows.assignments)
    else:
        new_mallows = state.mallows

    new_pl = pl_build(
        rankings=rankings,
        weights=new_weights,
        assignments=new_assignments,
        all_drivers=state.all_drivers,
        n_clusters=state.n_clusters,
        n_iter=pl_n_iter,
        prev_model=state.pl,
    )

    return ModelState(
        mallows=new_mallows,
        pl=new_pl,
        seen_records=new_records,
        seen_weights=new_weights,
        assignments=new_assignments,
        n_clusters=state.n_clusters,
        alpha=state.alpha,
        all_drivers=state.all_drivers,
        current_season=new_record.season,
        mallows_n_iter=mallows_n_iter,
        pl_n_iter=pl_n_iter,
    )
