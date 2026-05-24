"""
src/pipeline_openf1/run_pipeline_openf1.py
===========================================
Entry point do Pipeline 3 — Mallows + Plackett–Luce + OpenF1.

Features contextuais utilizadas (via src/data_openf1/):
    race_control  → sc_count, vsc_count, red_flag_count, yellow_flag_count
    starting_grid → grid_<SIGLA>   (posição de largada por piloto)
    session_result→ dnf_<SIGLA>    (1 se abandonou, 0 caso contrário)

Divisão temporal:
    Treino     : 2019–2023
    Validação  : 2024
    Teste      : 2025

Uso (a partir da raiz do projeto):
    python -m src.pipeline_openf1.run_pipeline_openf1

Saída:
    src/pipeline_openf1/outputs/nb_data_p3.pkl
    src/pipeline_openf1/outputs/viz*.png
"""

from __future__ import annotations

import os
import sys
import random
import pickle
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

# ── caminhos relativos à raiz do repositório ──────────────────────────────────
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR    = os.path.join(ROOT_DIR, 'data')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
sys.path.insert(0, ROOT_DIR)

# ── configuração ──────────────────────────────────────────────────────────────
SEED          = 42
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
VAL_SEASONS   = [2024]
TEST_SEASONS  = [2025]
ALL_SEASONS   = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS

N_CLUSTERS    = 2
N_ITER        = 150
ALPHA         = 0.5
TOP_K         = 10
N_SIMULATIONS = 10_000
N_POSITIONS   = 20

TOP5_DRIVERS  = ['VER', 'NOR', 'LEC', 'PIA', 'HAM']

np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline3")

# ── imports dos módulos compartilhados ────────────────────────────────────────
from src.data.data_pipeline              import load_seasons, get_all_drivers, RaceRecord
from src.engine.engine_trainer           import initial_fit, incremental_update, ModelState
from src.engine.engine_predictor         import predict as make_prediction
from src.evaluation.evaluation_metrics   import evaluate_race, season_summary, print_comparison, RaceEval
from src.pipeline_score_rules.monte_carlo    import simulate, uniform_baseline, RaceDistribution
from src.pipeline_score_rules.scoring_rules  import (
    compute_rps, rps_season_summary, print_rps_summary, RPSResult, RPSSummary,
)
from src.pipeline_mallows_plackett_luce.visualization_plots import SkillSnapshot

# ── imports do módulo de dados OpenF1 (compartilhado) ────────────────────────
from src.data_openf1.feature_builder import load_race_context

# ── imports de visualizações do Pipeline 3 ───────────────────────────────────
from src.pipeline_openf1.visualization_plots_p3 import (
    plot_rps_evolution_p3,
    plot_rps_gain_p3,
    plot_context_impact,
    plot_win_probabilities_p3,
    plot_pipeline_comparison,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def _extract_context(ctx_df: pd.DataFrame, season: int, race: str) -> dict:
    """
    Extrai as features contextuais de uma corrida específica.

    Features retornadas:
        sc_count, vsc_count, red_flag_count, yellow_flag_count
        grid_<SIGLA> (via dicionário aninhado 'grid')
        dnf_<SIGLA>  (via dicionário aninhado 'dnf')

    Retorna valores padrão se a corrida não estiver na tabela.
    """
    default = {
        "sc_count": 0, "vsc_count": 0,
        "red_flag_count": 0, "yellow_flag_count": 0,
        "grid": {}, "dnf": {},
    }

    if ctx_df is None or ctx_df.empty:
        return default

    row = ctx_df[(ctx_df["season"] == season) & (ctx_df["race"] == race)]
    if row.empty:
        row = ctx_df[
            (ctx_df["season"] == season) &
            ctx_df["race"].str.lower().str.contains(race.lower()[:6], na=False)
        ]
    if row.empty:
        return default

    r = row.iloc[0]

    # Colunas dinâmicas grid_* e dnf_*
    grid = {
        col[5:]: int(r[col])
        for col in r.index
        if col.startswith("grid_") and pd.notna(r[col])
    }
    dnf = {
        col[4:]: int(r[col])
        for col in r.index
        if col.startswith("dnf_") and pd.notna(r[col])
    }

    return {
        "sc_count":           int(r.get("sc_count",           0)),
        "vsc_count":          int(r.get("vsc_count",          0)),
        "red_flag_count":     int(r.get("red_flag_count",     0)),
        "yellow_flag_count":  int(r.get("yellow_flag_count",  0)),
        "grid": grid,
        "dnf":  dnf,
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


# ── loop incremental ──────────────────────────────────────────────────────────

def run_incremental_phase(
    state:      ModelState,
    records:    list[RaceRecord],
    ctx_df:     pd.DataFrame,
    phase_name: str,
    season:     int,
) -> tuple[ModelState, list[RaceEval], list[RPSResult],
           list[RaceDistribution], list[SkillSnapshot]]:
    """
    Executa a fase incremental corrida a corrida.

    Para cada corrida:
        1. Snapshot do estado atual (antes da previsão)
        2. Previsão determinística + simulação Monte Carlo
        3. Extração do contexto OpenF1 (sc_count, vsc_count,
           red_flag_count, yellow_flag_count, grid_*, dnf_*)
        4. Avaliação: RaceEval + RPSResult
        5. Incorporação do resultado e atualização do modelo
    """
    evals:         list[RaceEval]         = []
    rps_results:   list[RPSResult]        = []
    distributions: list[RaceDistribution] = []
    snapshots:     list[SkillSnapshot]    = []

    _header(f"FASE {phase_name} — {season}")
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

        # Contexto OpenF1 — apenas as features definidas
        ctx = _extract_context(ctx_df, record.season, record.race)

        cluster_scores = state.pl.cluster_scores.get(
            pred.cluster_used, state.pl.global_scores
        )
        dist = simulate(
            skill_scores  = cluster_scores,
            season        = record.season,
            race          = record.race,
            cluster       = pred.cluster_used,
            n_positions   = N_POSITIONS,
            n_simulations = N_SIMULATIONS,
        )
        distributions.append(dist)

        baseline = uniform_baseline(
            drivers     = state.all_drivers,
            season      = record.season,
            race        = record.race,
            n_positions = N_POSITIONS,
        )

        ev  = evaluate_race(
            season       = record.season,
            race         = record.race,
            predicted    = pred.predicted_order,
            actual       = record.ranking,
            cluster_used = pred.cluster_used,
        )
        evals.append(ev)

        rps = compute_rps(dist, baseline, record.ranking)
        # Anexar contexto ao RPSResult para análise no NB07
        rps.__dict__["context"] = ctx
        rps_results.append(rps)

        marker = "✓" if rps.gain > 0 else "✗"
        print(
            f"  {record.race:22s} {pred.cluster_used+1:>3d} "
            f"{ev.top3_acc:>7.3f} {ev.kendall_tau:>8.3f} "
            f"{rps.rps_model:>8.4f} {rps.rps_baseline:>9.4f} "
            f"{rps.gain:>6.4f}{marker} "
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

    summ     = season_summary(evals, season)
    rps_summ = rps_season_summary(rps_results, season)
    print("  " + "-" * 85)
    print(
        f"  {'MÉDIA':22s} {'':>3} "
        f"{summ.mean_top3:>7.3f} {summ.mean_kendall:>8.3f} "
        f"{rps_summ.mean_rps_model:>8.4f} "
        f"{rps_summ.mean_rps_baseline:>9.4f} "
        f"{rps_summ.mean_gain:>7.4f}"
    )

    return state, evals, rps_results, distributions, snapshots


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    _header("PIPELINE 3 — Mallows + Plackett–Luce + OpenF1  |  F1 TCC")

    # 1. Features contextuais da OpenF1 (apenas 2023–2025 têm cobertura)
    openf1_years = [y for y in ALL_SEASONS if y >= 2023]
    logger.info("Carregando contexto OpenF1 para: %s", openf1_years)
    ctx_df = load_race_context(openf1_years)
    logger.info("Features contextuais: %d corridas", len(ctx_df))

    # 2. Rankings históricos dos CSVs
    all_records   = load_seasons(DATA_DIR, ALL_SEASONS, top_k=TOP_K)
    all_drivers   = get_all_drivers(all_records)
    train_records = [r for r in all_records if r.season in TRAIN_SEASONS]
    val_records   = [r for r in all_records if r.season in VAL_SEASONS]
    test_records  = [r for r in all_records if r.season in TEST_SEASONS]

    print(f"\n  Treino    : {len(train_records)} corridas  {TRAIN_SEASONS}")
    print(f"  Validação : {len(val_records)} corridas  {VAL_SEASONS}")
    print(f"  Teste     : {len(test_records)} corridas  {TEST_SEASONS}")
    print(f"  Pilotos   : {len(all_drivers)}")

    # 3. Taxas históricas de DNF (sobre treino — para análise no NB07)
    dnf_analysis = _compute_dnf_rates(train_records)

    # 4. Treino inicial (2019–2023)
    _header("TREINO INICIAL (2019–2023)")
    state = initial_fit(
        records     = train_records,
        all_drivers = all_drivers,
        n_clusters  = N_CLUSTERS,
        n_iter      = N_ITER,
        alpha       = ALPHA,
        verbose     = True,
    )

    # 5. Validação incremental (2024)
    state, val_evals, val_rps, val_dists, val_snaps = run_incremental_phase(
        state=state, records=val_records, ctx_df=ctx_df,
        phase_name="VALIDAÇÃO", season=2024,
    )

    # 6. Teste incremental (2025)
    state, test_evals, test_rps, test_dists, test_snaps = run_incremental_phase(
        state=state, records=test_records, ctx_df=ctx_df,
        phase_name="TESTE", season=2025,
    )

    # 7. Resumo
    val_sum  = season_summary(val_evals,  2024)
    test_sum = season_summary(test_evals, 2025)
    val_rps_sum  = rps_season_summary(val_rps,  2024)
    test_rps_sum = rps_season_summary(test_rps, 2025)

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
        val_rps   = val_rps,
        test_rps  = test_rps,
        save_path = os.path.join(OUTPUTS_DIR, 'viz1_rps_evolution_p3.png'),
    )

    print("[2/5] Ganho RPS sobre baseline...")
    plot_rps_gain_p3(
        val_rps   = val_rps,
        test_rps  = test_rps,
        save_path = os.path.join(OUTPUTS_DIR, 'viz2_rps_gain_p3.png'),
    )

    print("[3/5] Impacto de eventos de corrida no RPS...")
    plot_context_impact(
        rps_results = val_rps + test_rps,
        save_path   = os.path.join(OUTPUTS_DIR, 'viz3_context_impact_p3.png'),
    )

    print("[4/5] Probabilidades de vitória — Teste 2025...")
    plot_win_probabilities_p3(
        distributions = test_dists,
        records       = test_records,
        top_drivers   = TOP5_DRIVERS,
        season        = 2025,
        save_path     = os.path.join(OUTPUTS_DIR, 'viz4_win_probabilities_p3.png'),
    )

    print("[5/5] Comparação de RPS entre pipelines...")
    plot_pipeline_comparison(
        val_rps_p3   = val_rps_sum,
        test_rps_p3  = test_rps_sum,
        save_path    = os.path.join(OUTPUTS_DIR, 'viz5_pipeline_comparison.png'),
    )

    # 10. Serializar para o NB07
    _header("SALVANDO DADOS PARA O NOTEBOOK 07")
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
        "config": {
            "TRAIN_SEASONS":  TRAIN_SEASONS,
            "VAL_SEASONS":    VAL_SEASONS,
            "TEST_SEASONS":   TEST_SEASONS,
            "TOP_K":          TOP_K,
            "N_CLUSTERS":     N_CLUSTERS,
            "N_SIMULATIONS":  N_SIMULATIONS,
            "SEED":           SEED,
            "features_used":  [
                "sc_count", "vsc_count", "red_flag_count", "yellow_flag_count",
                "grid_<SIGLA>", "dnf_<SIGLA>",
            ],
        },
    }

    pkl_out = os.path.join(OUTPUTS_DIR, "nb_data_p3.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(nb_data, f)

    print(f"\n  Gráficos : {OUTPUTS_DIR}")
    print(f"  Dados    : {pkl_out}")
    print(f"\n{'=' * 65}")
    print("  Pipeline 3 concluído.")
    print(f"{'=' * 65}\n")


if __name__ == '__main__':
    main()
