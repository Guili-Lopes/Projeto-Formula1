"""
src/pipeline_openf1/run_pipeline_openf1.py
===========================================
Entry point do Pipeline 3 — Mallows + Plackett–Luce + contexto OpenF1.
Migrado na Etapa 6 da reestruturação.

Features contextuais (pós-corrida, não usadas na previsão):
    race_control  → sc_count, vsc_count, red_flag_count, yellow_flag_count
    starting_grid → grid_<SIGLA>
    session_result→ dnf_<SIGLA>

Uso (a partir da raiz do projeto):
    python -m src.pipeline_openf1.run_pipeline_openf1
    python -m src.pipeline_openf1.run_pipeline_openf1 --config <yaml>

Regra da reestruturação: o pipeline NÃO chama a API OpenF1
(OPENF1_OFFLINE=1); consome somente dados sincronizados em data/openf1/.

Saída:
    artifacts/pipeline_openf1/<run_id>/  (tabelas, plots, nb_data_p3.pkl,
    run.log, manifest, metrics_summary) + latest_run.json (ponteiro)
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

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
from src.pipeline_score_rules.monte_carlo import (
    simulate, uniform_baseline, RaceDistribution,
)
from src.pipeline_score_rules.scoring_rules import (
    compute_rps, rps_season_summary, print_rps_summary, RPSResult,
)
from src.pipeline_mallows_plackett_luce.visualization_plots import SkillSnapshot
from src.data_openf1.feature_builder import load_race_context
from src.data_openf1.race_mapping import canonical_race_key
from src.data_openf1.coverage_report import (
    create_coverage_report, print_coverage_summary,
)
from src.pipeline_openf1.visualization_plots_p3 import (
    plot_rps_evolution_p3,
    plot_rps_gain_p3,
    plot_context_impact,
    plot_win_probabilities_p3,
    plot_pipeline_comparison,
)

PIPELINE_NAME = "pipeline_openf1"
PIPELINE_DIR = os.path.dirname(__file__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline3")


def _header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def _extract_context(ctx_df: pd.DataFrame, season: int, race: str) -> dict:
    """Extrai as features contextuais de uma corrida via race_key canônico."""
    default = {
        "has_context": False,
        "race_key": canonical_race_key(race),
        "openf1_race": "",
        "grid_source": "missing_context",
        "sc_count": 0,
        "vsc_count": 0,
        "red_flag_count": 0,
        "yellow_flag_count": 0,
        "grid": {},
        "dnf": {},
    }

    if ctx_df is None or ctx_df.empty:
        return default

    df = ctx_df.copy()
    if "race_key" not in df.columns and "race" in df.columns:
        df["race_key"] = df["race"].map(canonical_race_key)

    race_key = canonical_race_key(race)
    row = df[(df["season"] == season) & (df["race_key"] == race_key)]

    if row.empty and "race" in df.columns:
        row = df[
            (df["season"] == season)
            & (df["race"].fillna("").astype(str).str.lower() == str(race).lower())
        ]

    if row.empty:
        return default

    r = row.iloc[0]

    grid_skip = {"grid_source", "grid_driver_count"}
    grid = {
        col[5:]: int(float(r[col]))
        for col in r.index
        if str(col).startswith("grid_") and col not in grid_skip and pd.notna(r[col])
    }
    dnf = {
        col[4:]: int(float(r[col]))
        for col in r.index
        if str(col).startswith("dnf_") and col != "dnf_driver_count" and pd.notna(r[col])
    }

    return {
        "has_context": True,
        "race_key": race_key,
        "openf1_race": str(r.get("race", "")),
        "grid_source": str(r.get("grid_source", "unavailable")),
        "sc_count": int(float(r.get("sc_count", 0) or 0)),
        "vsc_count": int(float(r.get("vsc_count", 0) or 0)),
        "red_flag_count": int(float(r.get("red_flag_count", 0) or 0)),
        "yellow_flag_count": int(float(r.get("yellow_flag_count", 0) or 0)),
        "grid": grid,
        "dnf": dnf,
    }


def _compute_dnf_rates(records: list[RaceRecord]) -> dict:
    """Taxas históricas de DNF por piloto e circuito (sobre dados de treino)."""
    by_driver:  dict[str, list[int]]   = defaultdict(list)
    by_circuit: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        n_total = len(rec.ranking)
        n_dnf   = rec.n_dnf
        dnf_set = set(rec.ranking[n_total - n_dnf:]) if n_dnf > 0 else set()

        for drv in rec.ranking:
            by_driver[drv].append(1 if drv in dnf_set else 0)

        by_circuit[rec.race].append(n_dnf / n_total if n_total > 0 else 0.0)

    return {
        "by_driver":  {d: float(np.mean(v)) for d, v in by_driver.items()},
        "by_circuit": {c: float(np.mean(v)) for c, v in by_circuit.items()},
    }


def run_incremental_phase(
    state:       ModelState,
    records:     list[RaceRecord],
    ctx_df:      pd.DataFrame,
    phase_name:  str,
    season_label,
    *,
    n_positions:   int,
    n_simulations: int,
    mc_base_seed:  int | None,
) -> tuple[ModelState, list[RaceEval], list[RPSResult],
           list[RaceDistribution], list[SkillSnapshot]]:
    """Fase incremental corrida a corrida com contexto OpenF1."""
    evals:         list[RaceEval]         = []
    rps_results:   list[RPSResult]        = []
    distributions: list[RaceDistribution] = []
    snapshots:     list[SkillSnapshot]    = []

    _header(f"FASE {phase_name} — {season_label}")
    print(f"\n  {'Corrida':22s} {'Cl':>3} {'Top-3':>7} {'Kendall':>8} "
          f"{'RPS':>8} {'Baseline':>9} {'Ganho':>7} "
          f"{'SC':>3} {'VSC':>4} {'RF':>3} {'YF':>4}")
    print("  " + "-" * 85)

    for i, record in enumerate(records):

        snapshots.append(SkillSnapshot(
            season=record.season,
            race=record.race,
            label=f"{record.season} {record.race[:3]}",
            scores=dict(state.pl.global_scores),
        ))

        pred = make_prediction(state, record.season, record.race)

        ctx = _extract_context(ctx_df, record.season, record.race)

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
        rps.__dict__["context"] = ctx
        rps_results.append(rps)

        marker = "OK" if rps.gain > 0 else "--"
        print(
            f"  {record.race:22s} {pred.cluster_used+1:>3d} "
            f"{ev.top3_acc:>7.3f} {ev.kendall_tau:>8.3f} "
            f"{rps.rps_model:>8.4f} {rps.rps_baseline:>9.4f} "
            f"{rps.gain:>6.4f}{marker:>2s} "
            f"{ctx['sc_count']:>3d} {ctx['vsc_count']:>4d} "
            f"{ctx['red_flag_count']:>3d} {ctx['yellow_flag_count']:>4d}"
        )

        is_last     = (i == len(records) - 1)
        season_turn = (i < len(records) - 1 and
                       records[i + 1].season != record.season)
        state = incremental_update(
            state=state, new_record=record,
            refit_mallows=(is_last or season_turn),
            verbose=False,
        )

    summ     = season_summary(evals, season_label)
    rps_summ = rps_season_summary(rps_results, season_label)
    print("  " + "-" * 85)
    print(
        f"  {'MÉDIA':22s} {'':>3} "
        f"{summ.mean_top3:>7.3f} {summ.mean_kendall:>8.3f} "
        f"{rps_summ.mean_rps_model:>8.4f} "
        f"{rps_summ.mean_rps_baseline:>9.4f} "
        f"{rps_summ.mean_gain:>7.4f}"
    )

    return state, evals, rps_results, distributions, snapshots


# ── serialização tabular ─────────────────────────────────────────────────────

def _evals_dataframe(evals: list[RaceEval], phase: str):
    return pd.DataFrame([{
        "phase": phase, "season": e.season, "race": e.race,
        "cluster_used": e.cluster_used, "top3_acc": e.top3_acc,
        "top5_acc": e.top5_acc, "kendall_tau": e.kendall_tau,
    } for e in evals])


def _rps_dataframe(rps_results: list[RPSResult], phase: str):
    rows = []
    for r in rps_results:
        ctx = r.__dict__.get("context", {})
        rows.append({
            "phase": phase, "season": r.season, "race": r.race,
            "cluster": r.cluster, "rps_model": r.rps_model,
            "rps_baseline": r.rps_baseline, "gain": r.gain,
            "has_context": bool(ctx.get("has_context", False)),
            "grid_source": ctx.get("grid_source", ""),
            "sc_count": ctx.get("sc_count", 0),
            "vsc_count": ctx.get("vsc_count", 0),
            "red_flag_count": ctx.get("red_flag_count", 0),
            "yellow_flag_count": ctx.get("yellow_flag_count", 0),
        })
    return pd.DataFrame(rows)


def _predictions_dataframe(evals: list[RaceEval], phase: str):
    return pd.DataFrame([{
        "phase": phase, "season": e.season, "race": e.race,
        "predicted": json.dumps(e.predicted, ensure_ascii=False),
        "actual": json.dumps(e.actual, ensure_ascii=False),
    } for e in evals])


def _distributions_dataframe(dists: list[RaceDistribution], phase: str):
    rows = []
    for dist in dists:
        for driver, vec in dist.vectors.items():
            for pos_idx, prob in enumerate(vec.probs, start=1):
                rows.append({
                    "phase": phase, "season": dist.season, "race": dist.race,
                    "cluster": dist.cluster, "driver": driver,
                    "position": pos_idx, "probability": float(prob),
                })
    return pd.DataFrame(rows)


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
    config = load_pipeline_config(
        PIPELINE_DIR, argv,
        description="Pipeline 3 — Mallows + Plackett–Luce + contexto OpenF1")
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
    ctx_first     = int(config["context"]["first_year"])
    allow_partial = bool(config["context"].get("allow_partial", True))
    top5_drivers  = list(config["visualization"]["top5_drivers"])

    np.random.seed(seed)
    random.seed(seed)

    store = PipelineRunStore(
        PIPELINE_NAME, artifacts_root=config.get("artifacts", {}).get("root"))

    with pipeline_run(store):
        _header("PIPELINE 3 — Mallows + Plackett–Luce + OpenF1  |  F1 TCC")
        print(f"\n  Execução : {store.run_id}")
        print(f"  Config   : {config['_config_path']}")

        # 1. Rankings via repositório compartilhado (regra de fontes)
        all_records, provenance = load_season_records(
            all_seasons, top_k=top_k, source_policy=source_policy)
        all_drivers   = get_all_drivers(all_records)
        train_records = [r for r in all_records if r.season in train_seasons]
        val_records   = [r for r in all_records if r.season in val_seasons]
        test_records  = [r for r in all_records if r.season in test_seasons]

        # 2. Features contextuais da OpenF1 (dados já sincronizados em disco)
        openf1_years = [y for y in all_seasons if y >= ctx_first]
        min_races_by_year = {
            y: sum(1 for r in all_records if r.season == y)
            for y in openf1_years
        }
        logger.info("Carregando contexto OpenF1 para: %s", openf1_years)
        ctx_df = load_race_context(
            years=openf1_years,
            allow_partial=allow_partial,
            min_races_by_year=min_races_by_year,
        )
        logger.info("Features contextuais: %d corridas", len(ctx_df))

        if not ctx_df.empty and "grid_source" in ctx_df.columns:
            print("\n  [Grid source breakdown]:")
            for source, count in ctx_df["grid_source"].value_counts(dropna=False).items():
                print(f"    {str(source):28s}: {count}")

        print(f"\n  Treino    : {len(train_records)} corridas  {train_seasons}")
        print(f"  Validacao : {len(val_records)} corridas  {val_seasons}")
        print(f"  Teste     : {len(test_records)} corridas  {test_seasons}")
        print(f"  Pilotos   : {len(all_drivers)}")
        print(f"  Fontes    : {provenance}")

        coverage_report = create_coverage_report(
            records=val_records + test_records,
            ctx_df=ctx_df,
            years=val_seasons + test_seasons,
            save=False,  # pipelines não escrevem em data/; vai para artifacts
        )
        print_coverage_summary(coverage_report)

        # 3. Taxas históricas de DNF (sobre treino — para análise no NB07)
        dnf_analysis = _compute_dnf_rates(train_records)

        # 4. Treino inicial
        _header(f"TREINO INICIAL {train_seasons}")
        state = initial_fit(
            records     = train_records,
            all_drivers = all_drivers,
            n_clusters  = n_clusters,
            n_iter      = n_iter,
            alpha       = alpha,
            verbose     = True,
        )

        # 5–6. Validação e teste incrementais
        mc_kwargs = dict(n_positions=n_positions,
                         n_simulations=n_simulations,
                         mc_base_seed=mc_base_seed)
        state, val_evals, val_rps, val_dists, val_snaps = run_incremental_phase(
            state=state, records=val_records, ctx_df=ctx_df,
            phase_name="VALIDAÇÃO", season_label=val_seasons[0], **mc_kwargs)
        state, test_evals, test_rps, test_dists, test_snaps = run_incremental_phase(
            state=state, records=test_records, ctx_df=ctx_df,
            phase_name="TESTE", season_label=test_seasons[0], **mc_kwargs)

        # 7. Resumo
        val_sum      = season_summary(val_evals,  val_seasons[0])
        test_sum     = season_summary(test_evals, test_seasons[0])
        val_rps_sum  = rps_season_summary(val_rps,  val_seasons[0])
        test_rps_sum = rps_season_summary(test_rps, test_seasons[0])

        _header("RESUMO FINAL")
        print("\n  [Métricas determinísticas]:")
        print_comparison(val_sum, test_sum)
        print("\n  [RPS]:")
        print_rps_summary(val_rps_sum, test_rps_sum)

        # 8. Grid vs chegada (para análise no NB07)
        grid_vs_finish = []
        for rec in val_records + test_records:
            ctx = _extract_context(ctx_df, rec.season, rec.race)
            for pos_real, driver in enumerate(rec.ranking, start=1):
                grid_pos = ctx["grid"].get(driver)
                if grid_pos is not None:
                    grid_vs_finish.append({
                        "season":     rec.season,
                        "race":       rec.race,
                        "driver":     driver,
                        "grid_pos":   grid_pos,
                        "finish_pos": pos_real,
                        "dnf":        ctx["dnf"].get(driver, 0),
                    })

        # 9. Visualizações
        _header("GERANDO VISUALIZAÇÕES")
        print("\n[1/5] Evolução do RPS...")
        plot_rps_evolution_p3(
            val_rps=val_rps, test_rps=test_rps,
            save_path=store.plot_path('viz1_rps_evolution_p3.png'))
        print("[2/5] Ganho RPS sobre baseline...")
        plot_rps_gain_p3(
            val_rps=val_rps, test_rps=test_rps,
            save_path=store.plot_path('viz2_rps_gain_p3.png'))
        print("[3/5] Impacto de eventos de corrida no RPS...")
        plot_context_impact(
            rps_results=val_rps + test_rps,
            save_path=store.plot_path('viz3_context_impact_p3.png'))
        print("[4/5] Probabilidades de vitória — Teste...")
        plot_win_probabilities_p3(
            distributions=test_dists, records=test_records,
            top_drivers=top5_drivers, season=test_seasons[0],
            save_path=store.plot_path('viz4_win_probabilities_p3.png'))
        print("[5/5] Comparação de RPS entre pipelines...")
        plot_pipeline_comparison(
            val_rps_p3=val_rps_sum, test_rps_p3=test_rps_sum,
            save_path=store.plot_path('viz5_pipeline_comparison.png'))

        # 10. Artefatos tabulares + resumo + PKL auxiliar
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
        if not ctx_df.empty:
            store.write_dataframe("race_context", ctx_df, csv_mirror=False)
        if grid_vs_finish:
            store.write_dataframe(
                "grid_vs_finish", pd.DataFrame(grid_vs_finish))
        if coverage_report is not None and not coverage_report.empty:
            store.write_dataframe("openf1_coverage_report", coverage_report)
        store.write_json("dnf_analysis.json", dnf_analysis)

        metrics_summary = {
            "validation": _summary_dict(val_sum, val_rps_sum),
            "test": _summary_dict(test_sum, test_rps_sum),
            "monte_carlo": {
                "n_simulations": n_simulations,
                "n_positions": n_positions,
                "seed": mc_base_seed,
            },
            "data_provenance": provenance,
            "context_races": int(len(ctx_df)),
        }
        store.write_json("metrics_summary.json", metrics_summary)

        nb_data = {
            "state":              state,
            "train_records":      train_records,
            "val_records":        val_records,
            "test_records":       test_records,
            "all_drivers":        all_drivers,
            "val_evals":          val_evals,
            "test_evals":         test_evals,
            "val_rps":            val_rps,
            "test_rps":           test_rps,
            "val_rps_summary":    val_rps_sum,
            "test_rps_summary":   test_rps_sum,
            "val_distributions":  val_dists,
            "test_distributions": test_dists,
            "val_snapshots":      val_snaps,
            "test_snapshots":     test_snaps,
            "dnf_analysis":       dnf_analysis,
            "grid_vs_finish":     grid_vs_finish,
            "ctx_df":             ctx_df,
            "coverage_report":    coverage_report,
            "config": {
                "TRAIN_SEASONS":  train_seasons,
                "VAL_SEASONS":    val_seasons,
                "TEST_SEASONS":   test_seasons,
                "TOP_K":          top_k,
                "N_CLUSTERS":     n_clusters,
                "N_SIMULATIONS":  n_simulations,
                "SEED":           seed,
                "features_extracted": [
                    "grid_<SIGLA>",
                    "dnf_observed_<SIGLA>",
                    "sc_count", "vsc_count", "red_flag_count", "yellow_flag_count",
                ],
                "features_used_for_prediction": [
                    "skill_scores", "cluster", "historical_rankings"
                ],
                "post_race_features_not_used_for_prediction": [
                    "dnf_observed_<SIGLA>", "sc_count", "vsc_count",
                    "red_flag_count", "yellow_flag_count"
                ],
            },
        }
        store.write_pickle("nb_data_p3.pkl", nb_data)

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
        print(f"\n{'=' * 65}")
        print("  Pipeline 3 concluído.")
        print(f"{'=' * 65}\n")


if __name__ == '__main__':
    main()
