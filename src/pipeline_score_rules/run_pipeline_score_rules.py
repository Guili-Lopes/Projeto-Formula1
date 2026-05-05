"""
src/pipeline2/run_pipeline2.py
================================
Entry point do Pipeline 2 — Monte Carlo + RPS.

Incrementa o Pipeline 1 adicionando:
    - Vetor de probabilidades via Monte Carlo (10.000 simulações)
    - Avaliação via Ranked Probability Score (RPS)
    - Comparação contra baseline uniforme

Uso:
    Executar a partir da raiz do projeto:
    python -m src.pipeline2.run_pipeline2

Saída:
    - Métricas no terminal (Top-3, Kendall τ, RPS)
    - Dados serializados em src/pipeline2/outputs/nb_data_p2.pkl
"""

import os
import sys
import random
import pickle
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR    = os.path.join(ROOT_DIR, 'data')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

SEED          = 42
TRAIN_SEASONS = [2019, 2020, 2021, 2022]
VAL_SEASONS   = [2023]
TEST_SEASONS  = [2024]
ALL_SEASONS   = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS

N_CLUSTERS    = 2
N_ITER        = 150
ALPHA         = 0.5
TOP_K         = 10
N_SIMULATIONS = 10_000
N_POSITIONS   = 10

np.random.seed(SEED)
random.seed(SEED)

sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------------
# IMPORTS — compartilhados com Pipeline 1
# ---------------------------------------------------------------------------

from src.data.data_pipeline              import load_seasons, get_all_drivers, RaceRecord
from src.engine.engine_trainer           import initial_fit, incremental_update, ModelState
from src.engine.engine_predictor         import predict as make_prediction
from src.evaluation.evaluation_metrics   import (
    evaluate_race, season_summary, print_comparison, RaceEval,
)
from src.models.models_plackett_luce     import ranked_drivers

# ---------------------------------------------------------------------------
# IMPORTS — exclusivos do Pipeline 2
# ---------------------------------------------------------------------------

from src.pipeline2.monte_carlo   import simulate, uniform_baseline
from src.pipeline2.scoring_rules import (
    compute_rps, rps_season_summary,
    print_rps_table, print_rps_summary, RPSResult,
)


# ---------------------------------------------------------------------------
# AUXILIARES
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# FASE INCREMENTAL COM MONTE CARLO + RPS
# ---------------------------------------------------------------------------

def run_incremental_phase(
    state:      ModelState,
    records:    list[RaceRecord],
    phase_name: str,
    season:     int,
) -> tuple[ModelState, list[RaceEval], list[RPSResult]]:
    """
    Executa fase incremental com dois níveis de avaliação:

    Nível 1 — Previsão pontual (Pipeline 1):
        Top-3 Accuracy e Kendall τ

    Nível 2 — Previsão probabilística (Pipeline 2):
        Monte Carlo → vetor de probabilidades → RPS
    """
    evals:       list[RaceEval]  = []
    rps_results: list[RPSResult] = []

    _header(f"FASE {phase_name} — {season}")
    print(f"\n  {'Corrida':22s} {'Cl':>3} {'Top-3':>7} "
          f"{'Kendall':>8} {'RPS':>8} {'Baseline':>9} {'Ganho':>7}")
    print("  " + "-" * 70)

    for i, record in enumerate(records):

        # 1. Previsão pontual — Pipeline 1
        pred = make_prediction(state, record.season, record.race)

        # 2. Monte Carlo — usa skill scores do cluster identificado
        cluster_scores = state.pl.cluster_scores.get(
            pred.cluster_used, state.pl.global_scores
        )
        distribution = simulate(
            skill_scores  = cluster_scores,
            season        = record.season,
            race          = record.race,
            cluster       = pred.cluster_used,
            n_positions   = N_POSITIONS,
            n_simulations = N_SIMULATIONS,
        )

        # 3. Baseline uniforme
        baseline = uniform_baseline(
            drivers     = state.all_drivers,
            season      = record.season,
            race        = record.race,
            n_positions = N_POSITIONS,
        )

        # 4. Avaliação pontual
        ev = evaluate_race(
            season       = record.season,
            race         = record.race,
            predicted    = pred.predicted_order,
            actual       = record.ranking,
            cluster_used = pred.cluster_used,
        )
        evals.append(ev)

        # 5. RPS
        rps = compute_rps(
            distribution   = distribution,
            baseline       = baseline,
            actual_ranking = record.ranking,
        )
        rps_results.append(rps)

        marker = "✓" if rps.gain > 0 else "✗"
        print(f"  {record.race:22s} {pred.cluster_used+1:>3d} "
              f"{ev.top3_acc:>7.3f} {ev.kendall_tau:>8.3f} "
              f"{rps.rps_model:>8.4f} {rps.rps_baseline:>9.4f} "
              f"{rps.gain:>7.4f} {marker}")

        # 6. Incorporar resultado e atualizar modelo
        is_last       = (i == len(records) - 1)
        season_turn   = (i < len(records) - 1 and
                         records[i + 1].season != record.season)
        refit_mallows = is_last or season_turn

        state = incremental_update(
            state=state, new_record=record,
            refit_mallows=refit_mallows, verbose=False,
        )

    summary     = season_summary(evals, season)
    rps_summary = rps_season_summary(rps_results, season)
    print("  " + "-" * 70)
    print(f"  {'MÉDIA':22s} {'':>3} "
          f"{summary.mean_top3:>7.3f} {summary.mean_kendall:>8.3f} "
          f"{rps_summary.mean_rps_model:>8.4f} "
          f"{rps_summary.mean_rps_baseline:>9.4f} "
          f"{rps_summary.mean_gain:>7.4f}")

    return state, evals, rps_results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    _header("PIPELINE 2 — Monte Carlo + RPS  |  F1 TCC")

    # 1. Carregar dados
    print("\n[Dados] Carregando temporadas...")
    all_records   = load_seasons(DATA_DIR, ALL_SEASONS, top_k=TOP_K)
    all_drivers   = get_all_drivers(all_records)
    train_records = [r for r in all_records if r.season in TRAIN_SEASONS]
    val_records   = [r for r in all_records if r.season in VAL_SEASONS]
    test_records  = [r for r in all_records if r.season in TEST_SEASONS]

    print(f"\n  Treino:    {len(train_records)} corridas  {TRAIN_SEASONS}")
    print(f"  Validação: {len(val_records)} corridas  {VAL_SEASONS}")
    print(f"  Teste:     {len(test_records)} corridas  {TEST_SEASONS}")
    print(f"  Pilotos:   {len(all_drivers)} únicos")
    print(f"  Simulações Monte Carlo por corrida: {N_SIMULATIONS:,}")

    # 2. Treino inicial
    _header("FASE 1 — TREINO INICIAL (2019–2022)")
    state = initial_fit(
        records=train_records, all_drivers=all_drivers,
        n_clusters=N_CLUSTERS, n_iter=N_ITER, alpha=ALPHA, verbose=True,
    )

    # 3. Validação incremental
    state, val_evals, val_rps = run_incremental_phase(
        state=state, records=val_records,
        phase_name="VALIDAÇÃO", season=2023,
    )

    # 4. Teste incremental
    state, test_evals, test_rps = run_incremental_phase(
        state=state, records=test_records,
        phase_name="TESTE", season=2024,
    )

    # 5. Resumo final
    _header("RESUMO FINAL — PIPELINE 2")

    val_summary  = season_summary(val_evals,  2023)
    test_summary = season_summary(test_evals, 2024)
    print("\n  [Pipeline 1] Métricas pontuais:")
    print_comparison(val_summary, test_summary)

    val_rps_summary  = rps_season_summary(val_rps,  2023)
    test_rps_summary = rps_season_summary(test_rps, 2024)
    print("\n  [Pipeline 2] Ranked Probability Score:")
    print_rps_summary(val_rps_summary, test_rps_summary)

    print(f"\n  Skill scores finais (Top 10):")
    print(f"\n  {'#':>3} {'Piloto':>6}  {'Score':>10}")
    print("  " + "-" * 24)
    for i, (d, s) in enumerate(ranked_drivers(state.pl)[:10], 1):
        bar = '█' * int(s * 300)
        print(f"  {i:>3}. {d}  {s:.4f}  {bar}")

    # 6. Salvar dados
    _header("SALVANDO DADOS")
    nb_data = {
        'state':            state,
        'val_evals':        val_evals,
        'test_evals':       test_evals,
        'val_rps':          val_rps,
        'test_rps':         test_rps,
        'val_rps_summary':  val_rps_summary,
        'test_rps_summary': test_rps_summary,
        'all_drivers':      all_drivers,
        'train_records':    train_records,
    }
    nb_path = os.path.join(OUTPUTS_DIR, 'nb_data_p2.pkl')
    with open(nb_path, 'wb') as f:
        pickle.dump(nb_data, f)

    print(f"\n  Dados salvos em: {nb_path}")
    print(f"\n{'=' * 60}")
    print("  Pipeline 2 concluído.")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
