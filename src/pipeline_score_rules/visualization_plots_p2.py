"""
src/pipeline_score_rules/visualization_plots_p2.py
====================================================
Responsabilidade única: gerar as 4 visualizações do Pipeline 2
(Monte Carlo + RPS).

    Viz 1 — Mapa de calor de probabilidades por corrida
    Viz 2 — Evolução do RPS ao longo das corridas
    Viz 3 — Probabilidades de vitória P(1º) por corrida
    Viz 4 — Ganho sobre o baseline por corrida
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch

from src.pipeline_mallows_plackett_luce.visualization_plots import TEAM_COLORS, CLUSTER_COLORS

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO VISUAL
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.edgecolor':   '#444466',
    'axes.labelcolor':  '#ccccdd',
    'axes.titlecolor':  '#ffffff',
    'xtick.color':      '#aaaacc',
    'ytick.color':      '#aaaacc',
    'grid.color':       '#2a2a4a',
    'grid.linewidth':   0.6,
    'text.color':       '#ccccdd',
    'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#444466',
    'font.family':      'monospace',
})


# ---------------------------------------------------------------------------
# VIZ 1 — MAPA DE CALOR DE PROBABILIDADES
# ---------------------------------------------------------------------------

def plot_probability_heatmap(
    distributions:  list,
    records:        list,
    race_indices:   list[int] | None = None,
    save_path:      str | None = None,
) -> plt.Figure:
    """
    Mapa de calor: pilotos × posições para corridas selecionadas.

    Cada célula representa P(piloto i = posição k).
    A borda branca marca a posição real do piloto.

    Parâmetros
    ----------
    distributions : list[RaceDistribution]
        Distribuições geradas pelo Monte Carlo.
    records : list[RaceRecord]
        Corridas correspondentes (mesmo comprimento).
    race_indices : list[int], opcional
        Índices das corridas a exibir. Se None, exibe as 2 primeiras.
    save_path : str, opcional
        Caminho para salvar a figura.
    """
    if race_indices is None:
        race_indices = [0, 1]

    n_plots = len(race_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 7))
    fig.patch.set_facecolor('#1a1a2e')

    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, race_indices):
        dist   = distributions[idx]
        record = records[idx]

        top_drivers = dist.drivers[:10]
        n_pos       = dist.vectors[top_drivers[0]].n_positions
        matrix      = np.array([dist.vectors[d].probs for d in top_drivers])

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=matrix.max())
        plt.colorbar(im, ax=ax, label='Probabilidade')

        ax.set_xticks(range(n_pos))
        ax.set_xticklabels([f'{i+1}º' for i in range(n_pos)], fontsize=9)
        ax.set_yticks(range(len(top_drivers)))
        ax.set_yticklabels(top_drivers, fontsize=9)

        # Marcar posição real com borda branca
        pos_map = {d: i for i, d in enumerate(record.ranking)}
        for row, driver in enumerate(top_drivers):
            if driver in pos_map:
                col = pos_map[driver]
                if col < n_pos:
                    ax.add_patch(plt.Rectangle(
                        (col - 0.5, row - 0.5), 1, 1,
                        fill=False, edgecolor='white', linewidth=2.5
                    ))

        ax.set_title(
            f'Mapa de Probabilidades — {record.race} {record.season}\n'
            f'(borda branca = posição real)',
            fontsize=11, fontweight='bold', color='white',
        )
        ax.set_xlabel('Posição', fontsize=9)
        ax.set_ylabel('Piloto', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Salvo: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# VIZ 2 — EVOLUÇÃO DO RPS AO LONGO DAS CORRIDAS
# ---------------------------------------------------------------------------

def plot_rps_evolution(
    val_rps:   list,
    test_rps:  list,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Gráfico de linhas: RPS do modelo e do baseline corrida a corrida
    em 2023 e 2024. Área verde = modelo melhor, vermelha = baseline melhor.

    Parâmetros
    ----------
    val_rps : list[RPSResult]
        Resultados RPS da validação (2023).
    test_rps : list[RPSResult]
        Resultados RPS do teste (2024).
    save_path : str, opcional
        Caminho para salvar a figura.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, rps_list, season, color in zip(
        axes,
        [val_rps, test_rps],
        [2023, 2024],
        ['#FF8000', '#3671C6'],
    ):
        races    = [r.race[:8]     for r in rps_list]
        rps_mod  = [r.rps_model    for r in rps_list]
        rps_base = [r.rps_baseline for r in rps_list]
        x        = range(len(races))

        ax.plot(x, rps_base, color='#aaaacc', linewidth=1.5,
                label='Baseline', linestyle='--', alpha=0.7)
        ax.plot(x, rps_mod, color=color, linewidth=2.0,
                label='Modelo', marker='o', markersize=4)

        ax.fill_between(x, rps_mod, rps_base,
                        where=[m < b for m, b in zip(rps_mod, rps_base)],
                        alpha=0.2, color='#52E252', label='Ganho')
        ax.fill_between(x, rps_mod, rps_base,
                        where=[m >= b for m, b in zip(rps_mod, rps_base)],
                        alpha=0.2, color='#E8002D', label='Perda')

        ax.set_xticks(list(x))
        ax.set_xticklabels(races, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'RPS por Corrida — {season}',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('RPS', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Salvo: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# VIZ 3 — PROBABILIDADES DE VITÓRIA P(1º) POR CORRIDA
# ---------------------------------------------------------------------------

def plot_win_probabilities(
    distributions: list,
    records:       list,
    top_drivers:   list[str] | None = None,
    season:        int = 2024,
    save_path:     str | None = None,
) -> plt.Figure:
    """
    Barras agrupadas: P(1º) dos top-N pilotos para cada corrida.

    Parâmetros
    ----------
    distributions : list[RaceDistribution]
        Distribuições geradas pelo Monte Carlo.
    records : list[RaceRecord]
        Corridas correspondentes.
    top_drivers : list[str], opcional
        Pilotos a exibir. Se None, usa os 5 com maior P(1º) média.
    season : int
        Temporada (para o título).
    save_path : str, opcional
        Caminho para salvar a figura.
    """
    if top_drivers is None:
        top_drivers = ['VER', 'NOR', 'LEC', 'PIA', 'HAM']

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')

    races = [r.race[:8] for r in records]
    x     = np.arange(len(races))
    width = 0.15

    for i, driver in enumerate(top_drivers):
        win_probs = []
        for dist in distributions:
            vec = dist.vectors.get(driver)
            win_probs.append(vec.probs[0] if vec is not None else 0.0)
        color = TEAM_COLORS.get(driver, '#888899')
        ax.bar(x + i * width, win_probs, width, label=driver,
               color=color, alpha=0.85, edgecolor='#1a1a2e')

    ax.set_xticks(x + width * (len(top_drivers) - 1) / 2)
    ax.set_xticklabels(races, rotation=45, ha='right', fontsize=7)
    ax.set_title(f'Probabilidade de Vitória P(1º) por Corrida — {season}',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('P(1º)', fontsize=9)
    ax.legend(fontsize=9, framealpha=0.6)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Salvo: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# VIZ 4 — GANHO SOBRE O BASELINE POR CORRIDA
# ---------------------------------------------------------------------------

def plot_rps_gain(
    val_rps:   list,
    test_rps:  list,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Barras: ganho RPS sobre o baseline corrida a corrida.

    Verde = modelo melhor que o baseline.
    Vermelho = baseline melhor que o modelo.
    Linha tracejada = ganho médio.

    Parâmetros
    ----------
    val_rps : list[RPSResult]
        Resultados RPS da validação (2023).
    test_rps : list[RPSResult]
        Resultados RPS do teste (2024).
    save_path : str, opcional
        Caminho para salvar a figura.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, rps_list, season in zip(axes, [val_rps, test_rps], [2023, 2024]):
        races  = [r.race[:10] for r in rps_list]
        gains  = [r.gain      for r in rps_list]
        x      = range(len(races))
        colors = ['#52E252' if g > 0 else '#E8002D' for g in gains]

        ax.bar(x, gains, color=colors, alpha=0.85,
               edgecolor='#1a1a2e', linewidth=0.8)
        ax.axhline(y=0, color='#ffffff', linewidth=1.0, alpha=0.5)
        ax.axhline(y=np.mean(gains), color='#FF8000', linewidth=1.5,
                   linestyle='--', alpha=0.8,
                   label=f'Média: {np.mean(gains):.4f}')

        # Anotar corridas com ganho negativo
        for i, (race, gain) in enumerate(zip(races, gains)):
            if gain < -0.005:
                ax.annotate(race, xy=(i, gain), xytext=(0, -15),
                            textcoords='offset points', ha='center',
                            fontsize=7, color='#E8002D')

        ax.set_xticks(list(x))
        ax.set_xticklabels(races, rotation=45, ha='right', fontsize=7)
        ax.set_title(
            f'Ganho sobre Baseline — {season}  '
            f'(verde = modelo melhor | vermelho = baseline melhor)',
            fontsize=11, fontweight='bold',
        )
        ax.set_ylabel('Ganho (RPS baseline − RPS modelo)', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=9, framealpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Salvo: {save_path}")
    return fig
