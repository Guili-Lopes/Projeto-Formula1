"""
src/pipeline_mallows_plackett_luce/visualization_plots.py
==========================================================
Responsabilidade única: gerar as 4 visualizações do Pipeline 1.

    Viz 1 — Evolução dos skill scores corrida a corrida (2023–2024)
    Viz 3 — Mapa de clusters do Mallows
    Viz 4 — Pesos regulatórios ao longo do tempo
    Viz 5 — Ranking final de skill scores
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# CONFIGURAÇÃO VISUAL
# ---------------------------------------------------------------------------

TEAM_COLORS = {
    'VER': '#3671C6', 'PER': '#3671C6',
    'HAM': '#27F4D2', 'RUS': '#27F4D2',
    'LEC': '#E8002D', 'SAI': '#E8002D',
    'NOR': '#FF8000', 'PIA': '#FF8000',
    'ALO': '#358C75', 'STR': '#358C75',
    'GAS': '#0093CC', 'OCO': '#0093CC',
    'ALB': '#64C4FF', 'SAR': '#64C4FF',
    'TSU': '#6692FF', 'RIC': '#6692FF',
    'HUL': '#B6BABD', 'MAG': '#B6BABD',
    'BOT': '#52E252', 'ZHO': '#52E252',
}

CLUSTER_COLORS = ['#E8002D', '#3671C6']

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
# DATACLASS — snapshot de skill score por corrida
# ---------------------------------------------------------------------------

@dataclass
class SkillSnapshot:
    """Captura os skill scores do modelo em um dado momento."""
    season:  int
    race:    str
    label:   str
    scores:  dict[str, float]


# ---------------------------------------------------------------------------
# VIZ 1 — EVOLUÇÃO DOS SKILL SCORES
# ---------------------------------------------------------------------------

def plot_skill_evolution(
    snapshots:   list[SkillSnapshot],
    top_drivers: list[str] | None = None,
    save_path:   str | None = None,
) -> plt.Figure:
    """Evolução dos skill scores corrida a corrida durante 2023 e 2024."""
    if not snapshots:
        raise ValueError("snapshots está vazio.")

    if top_drivers is None:
        all_d = list(snapshots[0].scores.keys())
        means = {d: np.mean([s.scores.get(d, 0) for s in snapshots])
                 for d in all_d}
        top_drivers = sorted(means, key=lambda x: -means[x])[:10]

    seasons = [s.season for s in snapshots]
    x       = list(range(len(snapshots)))

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#1a1a2e')

    for driver in top_drivers:
        scores = [s.scores.get(driver, 0) for s in snapshots]
        color  = TEAM_COLORS.get(driver, '#ffffff')
        ax.plot(x, scores, color=color, linewidth=2.0,
                label=driver, marker='o', markersize=3, alpha=0.9)
        ax.annotate(driver, xy=(x[-1], scores[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    color=color, fontsize=9, va='center', fontweight='bold')

    for i in range(1, len(seasons)):
        if seasons[i] != seasons[i - 1]:
            ax.axvline(x=i - 0.5, color='#ffffff', linewidth=1.2,
                       linestyle='--', alpha=0.4)
            ax.text(i - 0.3, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0.17,
                    f'{seasons[i]} →', color='#aaaacc',
                    fontsize=8, ha='left', va='top', alpha=0.8)

    step = max(1, len(x) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(
        [f"{s.race[:6]}\n{s.season}" for s in snapshots[::step]],
        fontsize=7, rotation=30, ha='right',
    )

    ax.set_title('Evolução dos Skill Scores — 2023 e 2024',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Skill Score (Plackett–Luce)', fontsize=10)
    ax.set_xlabel('Corridas (ordem cronológica)', fontsize=10)
    ax.grid(True, axis='y', alpha=0.4)
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Salvo: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# VIZ 3 — MAPA DE CLUSTERS
# ---------------------------------------------------------------------------

def plot_cluster_map(
    race_names:  list[str],
    assignments: list[int],
    consensos:   list[list[str]],
    n_clusters:  int,
    seasons:     list[int],
    save_path:   str | None = None,
) -> plt.Figure:
    """Mapa visual: quais corridas foram para qual cluster."""
    season_races = defaultdict(list)
    for race, asgn, season in zip(race_names, assignments, seasons):
        season_races[season].append((race, asgn))

    all_seasons = sorted(season_races.keys())
    max_races   = max(len(v) for v in season_races.values())

    fig, ax = plt.subplots(figsize=(16, len(all_seasons) * 1.5 + 2.5))
    fig.patch.set_facecolor('#1a1a2e')

    for row_idx, season in enumerate(all_seasons):
        for col_idx, (race, cluster) in enumerate(season_races[season]):
            color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
            rect  = mpatches.FancyBboxPatch(
                (col_idx * 1.08, row_idx * 1.4), 1.0, 1.1,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='#1a1a2e',
                linewidth=1.5, alpha=0.85,
            )
            ax.add_patch(rect)
            ax.text(col_idx * 1.08 + 0.50, row_idx * 1.4 + 0.55,
                    race[:11], ha='center', va='center',
                    fontsize=6.5, color='white', fontweight='bold')

        ax.text(-0.4, row_idx * 1.4 + 0.55, str(season),
                ha='right', va='center',
                fontsize=11, color='#aaaacc', fontweight='bold')

    legend_patches = [
        mpatches.Patch(
            color=CLUSTER_COLORS[c], alpha=0.85,
            label=f'Cluster {c+1}:  {" > ".join(consensos[c][:5])}'
        )
        for c in range(n_clusters)
    ]
    ax.legend(handles=legend_patches, loc='lower center',
              fontsize=9, framealpha=0.7,
              bbox_to_anchor=(0.5, -0.06), ncol=n_clusters)

    ax.set_xlim(-0.8, max_races * 1.08 + 0.2)
    ax.set_ylim(-0.5, len(all_seasons) * 1.4 + 0.3)
    ax.set_title('Mapa de Clusters do Mallows — Padrões de Resultado por Corrida',
                 fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Salvo: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# VIZ 4 — PESOS REGULATÓRIOS
# ---------------------------------------------------------------------------

def plot_regulatory_weights(
    seasons:   list[int],
    races:     list[str],
    weights:   list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """Pesos regulatórios de cada corrida do conjunto de treino."""
    x      = list(range(len(weights)))
    colors = [CLUSTER_COLORS[0] if s <= 2021 else CLUSTER_COLORS[1]
              for s in seasons]

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')

    ax.bar(x, weights, color=colors, alpha=0.85, width=0.85,
           edgecolor='#1a1a2e', linewidth=0.5)

    change_idx = next((i for i, s in enumerate(seasons) if s == 2022), None)
    if change_idx:
        ax.axvline(x=change_idx - 0.5, color='#ffffff',
                   linewidth=1.5, linestyle='--', alpha=0.6)
        ax.text(change_idx + 0.5, max(weights) * 0.92,
                'Mudança de\nRegulamento\n(2022)',
                color='#ffffff', fontsize=8, alpha=0.8, va='top')

    season_starts = {}
    for i, s in enumerate(seasons):
        if s not in season_starts:
            season_starts[s] = i
    for season, idx in season_starts.items():
        ax.text(idx, -0.055, str(season), ha='left', va='top',
                fontsize=9, color='#aaaacc',
                transform=ax.get_xaxis_transform())

    patches = [
        mpatches.Patch(color=CLUSTER_COLORS[0], alpha=0.85,
                       label='Era 1 — 2019–2021 (regulamento antigo)'),
        mpatches.Patch(color=CLUSTER_COLORS[1], alpha=0.85,
                       label='Era 2 — 2022 em diante (efeito solo)'),
    ]
    ax.legend(handles=patches, fontsize=9, framealpha=0.6, loc='upper left')
    ax.set_title('Pesos Regulatórios por Corrida — Conjunto de Treino (2019–2022)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Peso Final  (era × decaimento temporal)', fontsize=10)
    ax.set_xlabel('Corridas em ordem cronológica', fontsize=10)
    ax.set_xticks([])
    ax.set_ylim(0, max(weights) * 1.15)
    ax.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Salvo: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# VIZ 5 — RANKING FINAL DE SKILL SCORES
# ---------------------------------------------------------------------------

def plot_skill_ranking(
    skill_scores: dict[str, float],
    top_n:        int = 15,
    title:        str = 'Skill Scores Finais — Plackett–Luce (2019–2024)',
    save_path:    str | None = None,
) -> plt.Figure:
    """Ranking final de skill scores dos pilotos."""
    sorted_drivers = sorted(skill_scores.items(), key=lambda x: x[1])
    sorted_drivers = sorted_drivers[-top_n:]

    drivers = [d for d, _ in sorted_drivers]
    scores  = [s for _, s in sorted_drivers]
    colors  = [TEAM_COLORS.get(d, '#888899') for d in drivers]

    fig, ax = plt.subplots(figsize=(11, top_n * 0.55 + 2.0))
    fig.patch.set_facecolor('#1a1a2e')

    bars = ax.barh(drivers, scores, color=colors, alpha=0.88,
                   edgecolor='#1a1a2e', linewidth=0.8, height=0.7)

    for bar, score in zip(bars, scores):
        ax.text(score + max(scores) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}', va='center', ha='left',
                fontsize=8, color='#ccccdd')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Skill Score (Plackett–Luce)', fontsize=10)
    ax.set_xlim(0, max(scores) * 1.18)
    ax.grid(True, axis='x', alpha=0.4)
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Salvo: {save_path}")
    return fig
