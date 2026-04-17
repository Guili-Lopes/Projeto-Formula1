"""
    Entry point único do experimento F1 TCC.

    Executa o pipeline completo em uma única execução:

        FASE 1 — Treino inicial        (2019–2022)
        FASE 2 — Validação incremental (2023)
        FASE 3 — Teste incremental     (2024)
        FASE 4 — Geração de gráficos   (automática)

    Uso:
        python run_experiment.py

    Saída:
        - Métricas impressas no terminal
        - 4 gráficos salvos em outputs/plots/
"""

import os
import sys
import random
import numpy as np

# --------------------
# CONFIGURAÇÃO
# --------------------
DATA_DIR = r"C:\Users\guiga\OneDrive\Documentos\Projeto-Formula1\data"
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'plots')

SEED          = 42
TRAIN_SEASONS = [2019, 2020, 2021, 2022]
VAL_SEASONS   = [2023]
TEST_SEASONS  = [2024]
ALL_SEASONS   = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS

N_CLUSTERS = 2
N_ITER     = 150
ALPHA      = 0.5
TOP_K      = 10

# Pilotos exibidos na Viz 1 — os que disputaram o campeonato em 2023/2024
VIZ1_DRIVERS = ['VER', 'NOR', 'LEC', 'HAM', 'PIA', 'RUS', 'SAI', 'PER', 'ALO', 'STR']

np.random.seed(SEED)
random.seed(SEED)

sys.path.insert(0, os.path.dirname(__file__))

# --------------------
# IMPORTS
# --------------------
from data_pipeline      import load_seasons, get_all_drivers, RaceRecord
from engine_trainer     import initial_fit, incremental_update, ModelState
from engine_predictor   import predict as make_prediction
from evaluation_metrics import (
    evaluate_race, season_summary,
    print_race_table, print_comparison, RaceEval,
)
from models_weights          import compute as compute_weights
from models_plackett_luce    import ranked_drivers
from visualization_plots     import (
    SkillSnapshot,
    plot_skill_evolution,
    plot_cluster_map,
    plot_regulatory_weights,
    plot_skill_ranking,
)

# --------------------
# AUXILIARES
# --------------------
def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

# --------------------
# FASE INCREMENTAL
# --------------------
def run_incremental_phase(
    state:      ModelState,
    records:    list[RaceRecord],
    phase_name: str,
    season:     int,
) -> tuple[ModelState, list[RaceEval], list[SkillSnapshot]]:
    """
    Executa uma fase incremental corrida a corrida.

    Para cada corrida:
        1. Captura snapshot dos skill scores ANTES de incorporar
        2. Faz a previsão com o modelo atual
        3. Avalia contra o resultado real
        4. Incorpora o resultado e atualiza o modelo

    Retorna o estado atualizado, as avaliações e os snapshots.
    """
    evals:     list[RaceEval]      = []
    snapshots: list[SkillSnapshot] = []

    _header(f"FASE {phase_name} — {season} (incremental)")

    print(f"\n  {'Corrida':22s} {'Cluster':>8} {'Top-3':>8} "
          f"{'Top-5':>8} {'Kendall τ':>10}")
    print("  " + "-" * 60)

    for i, record in enumerate(records):

        # 1. Snapshot dos skill scores ANTES da previsão
        snapshots.append(SkillSnapshot(
            season = record.season,
            race   = record.race,
            label  = f"{record.season} {record.race[:3]}",
            scores = dict(state.pl.global_scores),
        ))

        # 2. Previsão (Mallows → Plackett–Luce)
        pred = make_prediction(
            state  = state,
            season = record.season,
            race   = record.race,
        )

        # 3. Avaliação
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

        # 4. Incorporar resultado e atualizar modelo
        is_last        = (i == len(records) - 1)
        season_turning = (i < len(records) - 1 and
                          records[i + 1].season != record.season)
        refit_mallows  = is_last or season_turning

        state = incremental_update(
            state         = state,
            new_record    = record,
            refit_mallows = refit_mallows,
            verbose       = False,
        )

    # Resumo da fase
    summary = season_summary(evals, season)
    print("  " + "-" * 60)
    print(f"  {'MÉDIA':22s} {'':>8} "
          f"{summary.mean_top3:>8.3f} "
          f"{summary.mean_top5:>8.3f} "
          f"{summary.mean_kendall:>10.3f}")

    return state, evals, snapshots


# --------------------
# MAIN
# --------------------
def main() -> None:

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    _header("PROJETO F1 — TCC  |  Mallows + Plackett–Luce Integrados")

    # --------------------
    # 1. CARREGAR DADOS
    # --------------------
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

    # --------------------
    # 2. TREINO INICIAL (2019–2022)
    # --------------------
    _header("FASE 1 — TREINO INICIAL (2019–2022)")

    state = initial_fit(
        records     = train_records,
        all_drivers = all_drivers,
        n_clusters  = N_CLUSTERS,
        n_iter      = N_ITER,
        alpha       = ALPHA,
        verbose     = True,
    )

    # --------------------
    # 3. VALIDAÇÃO INCREMENTAL (2023)
    # --------------------
    state, val_evals, val_snapshots = run_incremental_phase(
        state      = state,
        records    = val_records,
        phase_name = "VALIDAÇÃO",
        season     = 2023,
    )

    # --------------------
    # 4. TESTE INCREMENTAL (2024)
    # --------------------
    state, test_evals, test_snapshots = run_incremental_phase(
        state      = state,
        records    = test_records,
        phase_name = "TESTE",
        season     = 2024,
    )

    # --------------------
    # 5. RESUMO FINAL
    # --------------------
    _header("RESUMO FINAL DO EXPERIMENTO")

    val_summary  = season_summary(val_evals,  2023)
    test_summary = season_summary(test_evals, 2024)
    print_comparison(val_summary, test_summary)

    print(f"\n  Skill scores finais (Top 10 — após 2019–2024):")
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

    # --------------------
    # 6. VISUALIZAÇÕES
    # --------------------
    _header("GERANDO VISUALIZAÇÕES")

    all_snapshots = val_snapshots + test_snapshots

    # Viz 1 — Evolução dos skill scores
    print("\n[1/4] Evolução dos Skill Scores...")
    plot_skill_evolution(
        snapshots   = all_snapshots,
        top_drivers = VIZ1_DRIVERS,
        save_path   = os.path.join(OUTPUTS_DIR, 'viz1_skill_evolution.png'),
    )

    # Viz 2 — Mapa de clusters (conjunto de treino)
    print("[2/4] Mapa de Clusters...")
    train_seasons = [r.season for r in train_records]
    train_races   = [r.race   for r in train_records]
    train_assignments = state.assignments[:len(train_records)]

    plot_cluster_map(
        race_names  = train_races,
        assignments = train_assignments,
        consensos   = state.mallows.consensos,
        n_clusters  = state.n_clusters,
        seasons     = train_seasons,
        save_path   = os.path.join(OUTPUTS_DIR, 'viz2_cluster_map.png'),
    )

    # Viz 3 — Pesos regulatórios
    print("[3/4] Pesos Regulatórios...")
    weight_objs = compute_weights(train_seasons, train_races)
    weights_arr = [w.final_weight for w in weight_objs]

    plot_regulatory_weights(
        seasons   = train_seasons,
        races     = train_races,
        weights   = weights_arr,
        save_path = os.path.join(OUTPUTS_DIR, 'viz3_regulatory_weights.png'),
    )

    # Viz 4 — Ranking final de skill scores
    print("[4/4] Ranking Final de Skill Scores...")
    plot_skill_ranking(
        skill_scores = state.pl.global_scores,
        top_n        = 15,
        save_path    = os.path.join(OUTPUTS_DIR, 'viz4_skill_ranking.png'),
    )

    print(f"\n  Gráficos salvos em: {OUTPUTS_DIR}")
    print(f"\n{'=' * 60}")
    print("  Experimento concluído.")
    print(f"{'=' * 60}\n")

if __name__ == '__main__':
    main()