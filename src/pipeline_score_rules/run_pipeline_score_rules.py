"""
src/pipeline_score_rules/run_pipeline_score_rules.py
======================================================
Entry point do Pipeline Score Rules — Monte Carlo + RPS.
Migrado na Etapa 5 da reestruturação.

Uso (a partir da raiz do projeto):
    python -m src.pipeline_score_rules.run_pipeline_score_rules
    python -m src.pipeline_score_rules.run_pipeline_score_rules --config <yaml>

Configuração:
    src/pipeline_score_rules/configs/default.yaml
    (split histórico: treino 2019–2022, validação 2023, teste 2024)

Saída:
    artifacts/pipeline_score_rules/<run_id>/
        config_resolved.yaml, manifest.json, metrics_summary.json,
        race_metrics.parquet, rps_metrics.parquet, predictions.parquet,
        position_probabilities.parquet, nb_data_p2.pkl, run.log,
        runtime.json, plots/*.png
    artifacts/pipeline_score_rules/latest_run.json (ponteiro)
"""

from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

# Pipelines não chamam a API OpenF1: somente o sincronizador central pode.
os.environ.setdefault("OPENF1_OFFLINE", "1")

from src.data.repository import get_all_drivers, load_season_records
from src.data.data_pipeline import RaceRecord
from src.engine.engine_trainer import initial_fit, incremental_update, ModelState
from src.engine.engine_predictor import predict as make_prediction
from src.evaluation.evaluation_metrics import (
    evaluate_race, season_summary, print_comparison, RaceEval,
)
from src.experiments.pipeline_config import load_pipeline_config
from src.experiments.reproducibility import derive_seed
from src.experiments.run_store import PipelineRunStore, pipeline_run
from src.pipeline_mallows_plackett_luce.visualization_plots import SkillSnapshot
from src.pipeline_score_rules.monte_carlo import (
    simulate, uniform_baseline, RaceDistribution,
)
from src.pipeline_score_rules.scoring_rules import (
    compute_rps, rps_season_summary,
    print_rps_summary, RPSResult,
)
from src.pipeline_score_rules.visualization_plots_p2 import (
    plot_probability_heatmap,
    plot_rps_evolution,
    plot_win_probabilities,
    plot_rps_gain,
)

PIPELINE_NAME = "pipeline_score_rules"
PIPELINE_DIR = os.path.dirname(__file__)


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_incremental_phase(
    state:       ModelState,
    records:     list[RaceRecord],
    phase_name:  str,
    season_label,
    *,
    n_positions:   int,
    n_simulations: int,
    mc_base_seed:  int | None,
) -> tuple[ModelState, list[RaceEval], list[RPSResult],
           list[RaceDistribution], list[SkillSnapshot]]:
    """Fase incremental com avaliação pontual + probabilística."""
    evals:         list[RaceEval]          = []
    rps_results:   list[RPSResult]         = []
    distributions: list[RaceDistribution]  = []
    snapshots:     list[SkillSnapshot]     = []

    _header(f"FASE {phase_name} — {season_label}")
    print(f"\n  {'Corrida':22s} {'Cl':>3} {'Top-3':>7} "
          f"{'Kendall':>8} {'RPS':>8} {'Baseline':>9} {'Ganho':>7}")
    print("  " + "-" * 70)

    for i, record in enumerate(records):

        snapshots.append(SkillSnapshot(
            season = record.season,
            race   = record.race,
            label  = f"{record.season} {record.race[:3]}",
            scores = dict(state.pl.global_scores),
        ))

        pred = make_prediction(state, record.season, record.race)

        cluster_scores = state.pl.cluster_scores.get(
            pred.cluster_used, state.pl.global_scores
        )
        mc_seed = (derive_seed(mc_base_seed, record.season, record.race)
                   if mc_base_seed is not None else None)
        dist = simulate(
            skill_scores  = cluster_scores,
            season        = record.season,
            race          = record.race,
            cluster       = pred.cluster_used,
            n_positions   = n_positions,
            n_simulations = n_simulations,
            seed          = mc_seed,
        )
        distributions.append(dist)

        baseline = uniform_baseline(
            drivers     = state.all_drivers,
            season      = record.season,
            race        = record.race,
            n_positions = n_positions,
        )

        ev = evaluate_race(
            season       = record.season,
            race         = record.race,
            predicted    = pred.predicted_order,
            actual       = record.ranking,
            cluster_used = pred.cluster_used,
        )
        evals.append(ev)

        rps = compute_rps(dist, baseline, record.ranking)
        rps_results.append(rps)

        marker = "✓" if rps.gain > 0 else "✗"
        print(f"  {record.race:22s} {pred.cluster_used+1:>3d} "
              f"{ev.top3_acc:>7.3f} {ev.kendall_tau:>8.3f} "
              f"{rps.rps_model:>8.4f} {rps.rps_baseline:>9.4f} "
              f"{rps.gain:>7.4f} {marker}")

        is_last       = (i == len(records) - 1)
        season_turn   = (i < len(records) - 1 and
                         records[i + 1].season != record.season)
        refit_mallows = is_last or season_turn

        state = incremental_update(
            state=state, new_record=record,
            refit_mallows=refit_mallows, verbose=False,
        )

    summary     = season_summary(evals, season_label)
    rps_summary = rps_season_summary(rps_results, season_label)
    print("  " + "-" * 70)
    print(f"  {'MÉDIA':22s} {'':>3} "
          f"{summary.mean_top3:>7.3f} {summary.mean_kendall:>8.3f} "
          f"{rps_summary.mean_rps_model:>8.4f} "
          f"{rps_summary.mean_rps_baseline:>9.4f} "
          f"{rps_summary.mean_gain:>7.4f}")

    return state, evals, rps_results, distributions, snapshots


# ── serialização tabular ─────────────────────────────────────────────────────

def _rps_dataframe(rps_results: list[RPSResult], phase: str):
    import pandas as pd
    return pd.DataFrame([{
        "phase": phase,
        "season": r.season,
        "race": r.race,
        "cluster": r.cluster,
        "rps_model": r.rps_model,
        "rps_baseline": r.rps_baseline,
        "gain": r.gain,
    } for r in rps_results])


def _distributions_dataframe(dists: list[RaceDistribution], phase: str):
    import pandas as pd
    rows = []
    for dist in dists:
        for driver, vec in dist.vectors.items():
            for pos_idx, prob in enumerate(vec.probs, start=1):
                rows.append({
                    "phase": phase,
                    "season": dist.season,
                    "race": dist.race,
                    "cluster": dist.cluster,
                    "driver": driver,
                    "position": pos_idx,
                    "probability": float(prob),
                })
    return pd.DataFrame(rows)


def _evals_dataframe(evals: list[RaceEval], phase: str):
    import pandas as pd
    return pd.DataFrame([{
        "phase": phase, "season": e.season, "race": e.race,
        "cluster_used": e.cluster_used, "top3_acc": e.top3_acc,
        "top5_acc": e.top5_acc, "kendall_tau": e.kendall_tau,
    } for e in evals])


def _predictions_dataframe(evals: list[RaceEval], phase: str):
    import pandas as pd
    return pd.DataFrame([{
        "phase": phase, "season": e.season, "race": e.race,
        "predicted": json.dumps(e.predicted, ensure_ascii=False),
        "actual": json.dumps(e.actual, ensure_ascii=False),
    } for e in evals])


def _summary_dict(det, rps):
    return {
        "season": det.season,
        "n_races": det.n_races,
        "mean_top3": det.mean_top3,
        "mean_top5": det.mean_top5,
        "mean_kendall": det.mean_kendall,
        "mean_rps_model": rps.mean_rps_model,
        "mean_rps_baseline": rps.mean_rps_baseline,
        "mean_gain": rps.mean_gain,
    }


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    config = load_pipeline_config(
        PIPELINE_DIR, argv, description="Pipeline Score Rules — Monte Carlo + RPS")
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
    n_simulations = int(config["monte_carlo"]["n_simulations"])
    n_positions   = int(config["monte_carlo"]["n_positions"])
    mc_seed_raw   = config["monte_carlo"].get("seed")
    mc_base_seed  = None if mc_seed_raw in (None, "null") else int(mc_seed_raw)
    top5_drivers  = list(config["visualization"]["top5_drivers"])

    np.random.seed(seed)
    random.seed(seed)

    store = PipelineRunStore(
        PIPELINE_NAME, artifacts_root=config.get("artifacts", {}).get("root"))

    with pipeline_run(store):
        _header("PIPELINE SCORE RULES — Monte Carlo + RPS  |  F1 TCC")
        print(f"\n  Execução : {store.run_id}")
        print(f"  Config   : {config['_config_path']}")

        print("\n[Dados] Carregando temporadas via repositório compartilhado...")
        all_records, provenance = load_season_records(
            all_seasons, top_k=top_k, source_policy=source_policy,
            data_dir=config["data"].get("dir"))
        all_drivers   = get_all_drivers(all_records)
        train_records = [r for r in all_records if r.season in train_seasons]
        val_records   = [r for r in all_records if r.season in val_seasons]
        test_records  = [r for r in all_records if r.season in test_seasons]

        print(f"\n  Treino:    {len(train_records)} corridas  {train_seasons}")
        print(f"  Validação: {len(val_records)} corridas  {val_seasons}")
        print(f"  Teste:     {len(test_records)} corridas  {test_seasons}")
        print(f"  Pilotos:   {len(all_drivers)} únicos")
        print(f"  Fontes:    {provenance}")
        print(f"  Simulações Monte Carlo por corrida: {n_simulations:,}"
              + ("" if mc_base_seed is None
                 else f" (seed base {mc_base_seed})"))

        _header(f"FASE 1 — TREINO INICIAL {train_seasons}")
        state = initial_fit(
            records=train_records, all_drivers=all_drivers,
            n_clusters=n_clusters, n_iter=n_iter, alpha=alpha, verbose=True,
        )

        mc_kwargs = dict(n_positions=n_positions,
                         n_simulations=n_simulations,
                         mc_base_seed=mc_base_seed)
        state, val_evals, val_rps, val_dists, val_snaps = run_incremental_phase(
            state=state, records=val_records,
            phase_name="VALIDAÇÃO", season_label=val_seasons[0], **mc_kwargs)
        state, test_evals, test_rps, test_dists, test_snaps = run_incremental_phase(
            state=state, records=test_records,
            phase_name="TESTE", season_label=test_seasons[0], **mc_kwargs)

        _header("RESUMO FINAL")
        val_summary  = season_summary(val_evals,  val_seasons[0])
        test_summary = season_summary(test_evals, test_seasons[0])
        print("\n  [Pipeline 1] Métricas pontuais:")
        print_comparison(val_summary, test_summary)

        val_rps_sum  = rps_season_summary(val_rps,  val_seasons[0])
        test_rps_sum = rps_season_summary(test_rps, test_seasons[0])
        print("\n  [Pipeline Score Rules] RPS:")
        print_rps_summary(val_rps_sum, test_rps_sum)

        _header("GERANDO VISUALIZAÇÕES")
        worst_test = max(range(len(test_rps)),
                         key=lambda i: test_rps[i].rps_model)

        print("\n[1/4] Mapa de Calor de Probabilidades...")
        plot_probability_heatmap(
            distributions = test_dists,
            records       = test_records,
            race_indices  = [0, worst_test],
            save_path     = store.plot_path('viz1_probability_heatmap.png'),
        )
        print("[2/4] Evolução do RPS...")
        plot_rps_evolution(
            val_rps   = val_rps,
            test_rps  = test_rps,
            save_path = store.plot_path('viz2_rps_evolution.png'),
        )
        print("[3/4] Probabilidades de Vitória...")
        plot_win_probabilities(
            distributions = test_dists,
            records       = test_records,
            top_drivers   = top5_drivers,
            season        = test_seasons[0],
            save_path     = store.plot_path('viz3_win_probabilities.png'),
        )
        print("[4/4] Ganho sobre o Baseline...")
        plot_rps_gain(
            val_rps   = val_rps,
            test_rps  = test_rps,
            save_path = store.plot_path('viz4_rps_gain.png'),
        )

        _header("SALVANDO ARTEFATOS DA EXECUÇÃO")
        store.write_dataframe("race_metrics", pd.concat([
            _evals_dataframe(val_evals, "validation"),
            _evals_dataframe(test_evals, "test")], ignore_index=True))
        store.write_dataframe("rps_metrics", pd.concat([
            _rps_dataframe(val_rps, "validation"),
            _rps_dataframe(test_rps, "test")], ignore_index=True))
        store.write_dataframe("predictions", pd.concat([
            _predictions_dataframe(val_evals, "validation"),
            _predictions_dataframe(test_evals, "test")], ignore_index=True))
        store.write_dataframe("position_probabilities", pd.concat([
            _distributions_dataframe(val_dists, "validation"),
            _distributions_dataframe(test_dists, "test")],
            ignore_index=True), csv_mirror=False)

        metrics_summary = {
            "validation": _summary_dict(val_summary, val_rps_sum),
            "test": _summary_dict(test_summary, test_rps_sum),
            "monte_carlo": {
                "n_simulations": n_simulations,
                "n_positions": n_positions,
                "seed": mc_base_seed,
            },
            "data_provenance": provenance,
        }
        store.write_json("metrics_summary.json", metrics_summary)

        nb_data = {
            'state':             state,
            'val_evals':         val_evals,
            'test_evals':        test_evals,
            'val_rps':           val_rps,
            'test_rps':          test_rps,
            'val_rps_summary':   val_rps_sum,
            'test_rps_summary':  test_rps_sum,
            'val_distributions': val_dists,
            'test_distributions': test_dists,
            'val_snapshots':     val_snaps,
            'test_snapshots':    test_snaps,
            'val_records':       val_records,
            'test_records':      test_records,
            'all_drivers':       all_drivers,
            'train_records':     train_records,
        }
        store.write_pickle("nb_data_p2.pkl", nb_data)

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
        print("  Pipeline Score Rules concluído.")
        print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
