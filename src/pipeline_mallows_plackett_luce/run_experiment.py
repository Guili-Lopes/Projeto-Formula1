"""
src/pipeline_mallows_plackett_luce/run_experiment.py
=====================================================
Entry point do Pipeline 1 — Mallows + Plackett–Luce com pesos regulatórios.
Migrado na Etapa 4 da reestruturação.

Uso (a partir da raiz do projeto):
    python -m src.pipeline_mallows_plackett_luce.run_experiment
    python -m src.pipeline_mallows_plackett_luce.run_experiment --config <yaml>

Configuração:
    src/pipeline_mallows_plackett_luce/configs/default.yaml
    (split histórico: treino 2019–2022, validação 2023, teste 2024)

Dados:
    via carregador compartilhado src/data/repository.py
    (até 2022 → dataset histórico; 2023+ → OpenF1 processada)

Saída:
    artifacts/pipeline_mallows_plackett_luce/<run_id>/
        config_resolved.yaml, manifest.json, metrics_summary.json,
        race_metrics.parquet, predictions.parquet, skill_history.parquet,
        regulatory_weights.parquet, nb_data.pkl, run.log, runtime.json,
        plots/*.png
    artifacts/pipeline_mallows_plackett_luce/latest_run.json (ponteiro)
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

# Pipelines não chamam a API OpenF1: somente o sincronizador central pode.
os.environ.setdefault("OPENF1_OFFLINE", "1")

from src.data.repository import get_all_drivers, load_season_records
from src.engine.engine_trainer import initial_fit, incremental_update, ModelState
from src.engine.engine_predictor import predict as make_prediction
from src.evaluation.evaluation_metrics import (
    evaluate_race, season_summary, print_comparison, RaceEval,
)
from src.experiments.pipeline_config import load_pipeline_config
from src.experiments.run_store import PipelineRunStore, pipeline_run
from src.models.models_weights import compute as compute_weights
from src.models.models_plackett_luce import ranked_drivers
from src.data.data_pipeline import RaceRecord
from src.pipeline_mallows_plackett_luce.visualization_plots import (
    SkillSnapshot, plot_skill_evolution, plot_cluster_map,
    plot_regulatory_weights, plot_skill_ranking,
)

PIPELINE_NAME = "pipeline_mallows_plackett_luce"
PIPELINE_DIR = os.path.dirname(__file__)


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_incremental_phase(
    state:      ModelState,
    records:    list[RaceRecord],
    phase_name: str,
    season_label,
) -> tuple[ModelState, list[RaceEval], list[SkillSnapshot]]:
    """Executa fase incremental corrida a corrida com coleta de snapshots."""
    evals:     list[RaceEval]      = []
    snapshots: list[SkillSnapshot] = []

    _header(f"FASE {phase_name} — {season_label} (incremental)")
    print(f"\n  {'Corrida':22s} {'Cluster':>8} {'Top-3':>8} "
          f"{'Top-5':>8} {'Kendall τ':>10}")
    print("  " + "-" * 60)

    for i, record in enumerate(records):

        snapshots.append(SkillSnapshot(
            season = record.season,
            race   = record.race,
            label  = f"{record.season} {record.race[:3]}",
            scores = dict(state.pl.global_scores),
        ))

        pred = make_prediction(state, record.season, record.race)

        ev = evaluate_race(
            season       = record.season,
            race         = record.race,
            predicted    = pred.predicted_order,
            actual       = record.ranking,
            cluster_used = pred.cluster_used,
        )
        evals.append(ev)

        print(f"  {record.race:22s} {pred.cluster_used+1:>8d} "
              f"{ev.top3_acc:>8.3f} {ev.top5_acc:>8.3f} "
              f"{ev.kendall_tau:>10.3f}")

        is_last       = (i == len(records) - 1)
        season_turn   = (i < len(records) - 1 and
                         records[i + 1].season != record.season)
        refit_mallows = is_last or season_turn

        state = incremental_update(
            state=state, new_record=record,
            refit_mallows=refit_mallows, verbose=False,
        )

    summary = season_summary(evals, season_label)
    print("  " + "-" * 60)
    print(f"  {'MÉDIA':22s} {'':>8} "
          f"{summary.mean_top3:>8.3f} "
          f"{summary.mean_top5:>8.3f} "
          f"{summary.mean_kendall:>10.3f}")

    return state, evals, snapshots


def _evals_dataframe(evals: list[RaceEval], phase: str):
    import pandas as pd
    return pd.DataFrame([{
        "phase": phase,
        "season": e.season,
        "race": e.race,
        "cluster_used": e.cluster_used,
        "top3_acc": e.top3_acc,
        "top5_acc": e.top5_acc,
        "kendall_tau": e.kendall_tau,
    } for e in evals])


def _predictions_dataframe(evals: list[RaceEval], phase: str):
    import json
    import pandas as pd
    return pd.DataFrame([{
        "phase": phase,
        "season": e.season,
        "race": e.race,
        "cluster_used": e.cluster_used,
        "predicted": json.dumps(e.predicted, ensure_ascii=False),
        "actual": json.dumps(e.actual, ensure_ascii=False),
    } for e in evals])


def _snapshots_dataframe(snapshots: list[SkillSnapshot], phase: str):
    import pandas as pd
    rows = []
    for order, snap in enumerate(snapshots):
        for driver, score in snap.scores.items():
            rows.append({
                "phase": phase,
                "order": order,
                "season": snap.season,
                "race": snap.race,
                "driver": driver,
                "skill_score": score,
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    config = load_pipeline_config(PIPELINE_DIR, argv,
                                  description="Pipeline 1 — Mallows + Plackett–Luce")
    seed          = int(config["seed"])
    top_k         = int(config["data"]["top_k"])
    source_policy = str(config["data"].get("source_policy", "prefer_openf1"))
    train_seasons = list(config["splits"]["train"])
    val_seasons   = list(config["splits"]["validation"])
    test_seasons  = list(config["splits"]["test"])
    all_seasons   = train_seasons + val_seasons + test_seasons
    n_clusters    = int(config["model"]["n_clusters"])
    n_iter        = int(config["model"]["n_iter"])
    alpha         = float(config["model"]["alpha"])
    viz1_drivers  = list(config["visualization"]["viz1_drivers"])
    top_n_ranking = int(config["visualization"].get("skill_ranking_top_n", 15))

    np.random.seed(seed)
    random.seed(seed)

    store = PipelineRunStore(
        PIPELINE_NAME, artifacts_root=config.get("artifacts", {}).get("root"))

    with pipeline_run(store):
        _header("PIPELINE 1 — Mallows + Plackett–Luce  |  F1 TCC")
        print(f"\n  Execução : {store.run_id}")
        print(f"  Config   : {config['_config_path']}")

        # 1. Carregar dados (regra de fontes da reestruturação)
        print("\n[Dados] Carregando temporadas via repositório compartilhado...")
        all_records, provenance = load_season_records(
            all_seasons, top_k=top_k, source_policy=source_policy)
        all_drivers   = get_all_drivers(all_records)
        train_records = [r for r in all_records if r.season in train_seasons]
        val_records   = [r for r in all_records if r.season in val_seasons]
        test_records  = [r for r in all_records if r.season in test_seasons]

        print(f"\n  Treino:    {len(train_records)} corridas  {train_seasons}")
        print(f"  Validação: {len(val_records)} corridas  {val_seasons}")
        print(f"  Teste:     {len(test_records)} corridas  {test_seasons}")
        print(f"  Pilotos:   {len(all_drivers)} únicos")
        print(f"  Fontes:    {provenance}")

        # 2. Treino inicial
        _header(f"FASE 1 — TREINO INICIAL {train_seasons}")
        state = initial_fit(
            records=train_records, all_drivers=all_drivers,
            n_clusters=n_clusters, n_iter=n_iter, alpha=alpha, verbose=True,
        )

        # 3. Validação e teste incrementais
        state, val_evals, val_snapshots = run_incremental_phase(
            state=state, records=val_records,
            phase_name="VALIDAÇÃO", season_label=val_seasons[0],
        )
        state, test_evals, test_snapshots = run_incremental_phase(
            state=state, records=test_records,
            phase_name="TESTE", season_label=test_seasons[0],
        )

        # 4. Resumo final
        _header("RESUMO FINAL")
        val_summary  = season_summary(val_evals,  val_seasons[0])
        test_summary = season_summary(test_evals, test_seasons[0])
        print_comparison(val_summary, test_summary)

        print(f"\n  Skill scores finais (Top 10):")
        print(f"\n  {'#':>3} {'Piloto':>6}  {'Score':>10}")
        print("  " + "-" * 24)
        for i, (d, s) in enumerate(ranked_drivers(state.pl)[:10], 1):
            bar = '█' * int(s * 300)
            print(f"  {i:>3}. {d}  {s:.4f}  {bar}")

        print(f"\n  Clusters finais do Mallows:")
        for c in range(state.n_clusters):
            names = [r.race for r, a in
                     zip(state.seen_records, state.assignments) if a == c]
            print(f"\n  Cluster {c+1} ({state.assignments.count(c)} corridas)")
            print(f"    Consenso: {' > '.join(state.mallows.consensos[c][:8])}")
            print(f"    Exemplos: {names[-5:]}")

        # 5. Visualizações (na pasta da execução)
        _header("GERANDO VISUALIZAÇÕES")
        all_snapshots = val_snapshots + test_snapshots
        train_seasons_list = [r.season for r in train_records]
        train_races        = [r.race   for r in train_records]

        print("\n[1/4] Evolução dos Skill Scores...")
        plot_skill_evolution(
            snapshots   = all_snapshots,
            top_drivers = viz1_drivers,
            save_path   = store.plot_path('viz1_skill_evolution.png'),
        )
        print("[2/4] Mapa de Clusters...")
        plot_cluster_map(
            race_names  = train_races,
            assignments = state.assignments[:len(train_records)],
            consensos   = state.mallows.consensos,
            n_clusters  = state.n_clusters,
            seasons     = train_seasons_list,
            save_path   = store.plot_path('viz3_cluster_map.png'),
        )
        print("[3/4] Pesos Regulatórios...")
        weight_objs = compute_weights(train_seasons_list, train_races)
        plot_regulatory_weights(
            seasons   = train_seasons_list,
            races     = train_races,
            weights   = [w.final_weight for w in weight_objs],
            save_path = store.plot_path('viz4_regulatory_weights.png'),
        )
        print("[4/4] Ranking Final de Skill Scores...")
        plot_skill_ranking(
            skill_scores = state.pl.global_scores,
            top_n        = top_n_ranking,
            save_path    = store.plot_path('viz5_skill_ranking.png'),
        )

        # 6. Artefatos tabulares + resumo
        _header("SALVANDO ARTEFATOS DA EXECUÇÃO")
        race_metrics = pd.concat([
            _evals_dataframe(val_evals, "validation"),
            _evals_dataframe(test_evals, "test"),
        ], ignore_index=True)
        predictions = pd.concat([
            _predictions_dataframe(val_evals, "validation"),
            _predictions_dataframe(test_evals, "test"),
        ], ignore_index=True)
        skill_history = pd.concat([
            _snapshots_dataframe(val_snapshots, "validation"),
            _snapshots_dataframe(test_snapshots, "test"),
        ], ignore_index=True)
        weights_df = pd.DataFrame({
            "season": train_seasons_list,
            "race": train_races,
            "final_weight": [w.final_weight for w in weight_objs],
        })

        store.write_dataframe("race_metrics", race_metrics)
        store.write_dataframe("predictions", predictions)
        store.write_dataframe("skill_history", skill_history, csv_mirror=False)
        store.write_dataframe("regulatory_weights", weights_df)

        metrics_summary = {
            "validation": {
                "season": val_seasons[0],
                "n_races": val_summary.n_races,
                "mean_top3": val_summary.mean_top3,
                "mean_top5": val_summary.mean_top5,
                "mean_kendall": val_summary.mean_kendall,
            },
            "test": {
                "season": test_seasons[0],
                "n_races": test_summary.n_races,
                "mean_top3": test_summary.mean_top3,
                "mean_top5": test_summary.mean_top5,
                "mean_kendall": test_summary.mean_kendall,
            },
            "data_provenance": provenance,
        }
        store.write_json("metrics_summary.json", metrics_summary)

        # 7. PKL auxiliar (compatível com o formato pré-reestruturação)
        nb_data = {
            'state':          state,
            'val_evals':      val_evals,
            'test_evals':     test_evals,
            'all_snapshots':  all_snapshots,
            'train_records':  train_records,
            'train_seasons':  train_seasons_list,
            'train_races':    train_races,
            'weights_arr':    [w.final_weight for w in weight_objs],
            'all_drivers':    all_drivers,
        }
        store.write_pickle("nb_data.pkl", nb_data)

        config_resolved = {k: v for k, v in config.items()
                           if not k.startswith("_")}
        store.write_yaml("config_resolved.yaml", config_resolved)
        store.write_manifest(config=config_resolved, extra={
            "data_provenance": provenance,
            "n_races": {
                "train": len(train_records),
                "validation": len(val_records),
                "test": len(test_records),
            },
        })
        store.finalize(summary=metrics_summary)

        print(f"\n  Artefatos em: {store.run_dir}")
        print(f"\n{'=' * 60}")
        print("  Pipeline 1 concluído.")
        print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
