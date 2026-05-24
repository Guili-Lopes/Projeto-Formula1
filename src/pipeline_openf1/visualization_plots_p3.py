"""
src/pipeline_openf1/visualization_plots_p3.py
===============================================
Gráficos exclusivos do Pipeline 3.

Contexto esperado em cada RPSResult (campo context):
    sc_count           int
    vsc_count          int
    red_flag_count     int
    yellow_flag_count  int
    grid               dict[str, int]   ex: {"VER": 1, "NOR": 2}
    dnf                dict[str, int]   ex: {"PER": 1, "VER": 0}
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.pipeline_score_rules.scoring_rules import RPSResult, RPSSummary
from src.pipeline_score_rules.monte_carlo   import RaceDistribution
from src.data.data_pipeline                 import RaceRecord

_BLUE   = "#2196F3"
_ORANGE = "#FF9800"
_RED    = "#F44336"
_GREEN  = "#4CAF50"
_GREY   = "#9E9E9E"
_PURPLE = "#9C27B0"
_YELLOW = "#FFC107"


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


def _get_ctx(r: RPSResult) -> dict:
    """Extrai o contexto de um RPSResult com valores padrão seguros."""
    ctx = getattr(r, "context", {}) or {}
    return {
        "sc_count":          ctx.get("sc_count",          0),
        "vsc_count":         ctx.get("vsc_count",         0),
        "red_flag_count":    ctx.get("red_flag_count",    0),
        "yellow_flag_count": ctx.get("yellow_flag_count", 0),
        "grid":              ctx.get("grid",              {}),
        "dnf":               ctx.get("dnf",               {}),
    }


# ── 1. Evolução do RPS por corrida ────────────────────────────────────────────

def plot_rps_evolution_p3(
    val_rps:   list[RPSResult],
    test_rps:  list[RPSResult],
    save_path: str,
) -> None:
    """Linha: RPS modelo e baseline por corrida — validação e teste."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor('#1a1a2e')

    for ax, rps_list, label, color in [
        (axes[0], val_rps,  "Validação 2024", _BLUE),
        (axes[1], test_rps, "Teste 2025",     _RED),
    ]:
        ax.set_facecolor('#1a1a2e')
        x         = range(len(rps_list))
        models    = [r.rps_model    for r in rps_list]
        baselines = [r.rps_baseline for r in rps_list]
        names     = [r.race[:10]    for r in rps_list]

        ax.plot(x, models,    marker="o", ms=5, color=color,
                label="Modelo",   alpha=0.9, linewidth=1.8)
        ax.plot(x, baselines, marker="s", ms=4, color=_GREY,
                label="Baseline", alpha=0.7, linestyle="--", linewidth=1.4)
        ax.axhline(np.mean(models), linestyle=":", color=color, alpha=0.5,
                   label=f"Média = {np.mean(models):.4f}")

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right",
                           fontsize=7, color="#ccccdd")
        ax.set_title(label, fontsize=12, fontweight="bold", color="white")
        ax.set_ylabel("RPS (↓ melhor)", color="#ccccdd")
        ax.tick_params(colors="#ccccdd")
        ax.legend(fontsize=8, framealpha=0.4)
        ax.grid(alpha=0.3)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    fig.suptitle(
        "Evolução do RPS — Pipeline 3 (OpenF1)\n"
        "Treino 2019–2023 | Val 2024 | Teste 2025",
        fontsize=12, color="white"
    )
    plt.tight_layout()
    _save(fig, save_path)


# ── 2. Ganho sobre o baseline por corrida ────────────────────────────────────

def plot_rps_gain_p3(
    val_rps:   list[RPSResult],
    test_rps:  list[RPSResult],
    save_path: str,
) -> None:
    """Barras horizontais: ganho (baseline – modelo) por corrida."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, rps_list, label in [
        (axes[0], val_rps,  "Validação 2024"),
        (axes[1], test_rps, "Teste 2025"),
    ]:
        ax.set_facecolor('#1a1a2e')
        gains  = [r.gain         for r in rps_list]
        names  = [r.race[:14]    for r in rps_list]
        colors = [_GREEN if g > 0 else _RED for g in gains]

        ax.barh(names, gains, color=colors, alpha=0.85, edgecolor='#1a1a2e')
        ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)

        pct = 100 * sum(1 for g in gains if g > 0) / len(gains)
        ax.set_title(
            f"{label}\n{pct:.0f}% das corridas acima do baseline",
            fontsize=11, fontweight="bold", color="white"
        )
        ax.set_xlabel("Ganho RPS (baseline – modelo)", color="#ccccdd")
        ax.tick_params(colors="#ccccdd")
        ax.grid(axis="x", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    fig.suptitle("Ganho sobre Baseline Uniforme — Pipeline 3",
                 fontsize=12, color="white")
    plt.tight_layout()
    _save(fig, save_path)


# ── 3. Impacto de eventos de corrida no RPS ───────────────────────────────────

def plot_context_impact(
    rps_results: list[RPSResult],
    save_path:   str,
) -> None:
    """
    Três painéis:
        - RPS médio por nº de Safety Cars
        - RPS médio por nº de Yellow Flags
        - RPS médio por presença de bandeira vermelha
    """
    sc_rps:    dict[int, list[float]] = {}
    yf_rps:    dict[int, list[float]] = {}
    rf_rps:    dict[str, list[float]] = {"Sem Red Flag": [], "Com Red Flag": []}

    for r in rps_results:
        ctx = _get_ctx(r)
        sc_rps.setdefault(ctx["sc_count"],          []).append(r.rps_model)
        # Agrupar yellow flags em faixas para não fragmentar demais
        yf_bin = min(ctx["yellow_flag_count"], 5)   # cap em 5+
        yf_rps.setdefault(yf_bin, []).append(r.rps_model)
        rf_key = "Com Red Flag" if ctx["red_flag_count"] > 0 else "Sem Red Flag"
        rf_rps[rf_key].append(r.rps_model)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')

    # Painel 1 — Safety Car
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    sc_keys  = sorted(sc_rps.keys())
    sc_means = [np.mean(sc_rps[k]) for k in sc_keys]
    sc_ns    = [len(sc_rps[k])     for k in sc_keys]
    bars = ax.bar([str(k) for k in sc_keys], sc_means,
                  color=_ORANGE, alpha=0.85, edgecolor='#1a1a2e')
    for bar, m, n in zip(bars, sc_means, sc_ns):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.0005,
                f"{m:.4f}\n(n={n})", ha="center", fontsize=7, color="#ccccdd")
    ax.set_xlabel("Nº de Safety Cars", color="#ccccdd")
    ax.set_ylabel("RPS médio (↓ melhor)", color="#ccccdd")
    ax.set_title("RPS por Safety Car", fontsize=11,
                 fontweight="bold", color="white")
    ax.tick_params(colors="#ccccdd")
    ax.grid(axis="y", alpha=0.3)

    # Painel 2 — Yellow Flags
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    yf_keys  = sorted(yf_rps.keys())
    yf_means = [np.mean(yf_rps[k]) for k in yf_keys]
    yf_ns    = [len(yf_rps[k])     for k in yf_keys]
    xlabels  = [str(k) if k < 5 else "5+" for k in yf_keys]
    bars2 = ax2.bar(xlabels, yf_means,
                    color=_YELLOW, alpha=0.85, edgecolor='#1a1a2e')
    for bar, m, n in zip(bars2, yf_means, yf_ns):
        ax2.text(bar.get_x() + bar.get_width()/2, m + 0.0005,
                 f"{m:.4f}\n(n={n})", ha="center", fontsize=7, color="#333333")
    ax2.set_xlabel("Nº de Yellow Flags", color="#ccccdd")
    ax2.set_ylabel("RPS médio (↓ melhor)", color="#ccccdd")
    ax2.set_title("RPS por Yellow Flags", fontsize=11,
                  fontweight="bold", color="white")
    ax2.tick_params(colors="#ccccdd")
    ax2.grid(axis="y", alpha=0.3)

    # Painel 3 — Red Flag
    ax3 = axes[2]
    ax3.set_facecolor('#1a1a2e')
    rf_labels = list(rf_rps.keys())
    rf_means  = [np.mean(v) if v else np.nan for v in rf_rps.values()]
    rf_ns     = [len(v)                       for v in rf_rps.values()]
    colors_rf = [_GREEN, _RED]
    bars3 = ax3.bar(rf_labels, rf_means, color=colors_rf,
                    alpha=0.85, edgecolor='#1a1a2e')
    for bar, m, n in zip(bars3, rf_means, rf_ns):
        if not np.isnan(m):
            ax3.text(bar.get_x() + bar.get_width()/2, m + 0.0005,
                     f"{m:.4f}\n(n={n})", ha="center", fontsize=8,
                     color="#ccccdd")
    ax3.set_ylabel("RPS médio (↓ melhor)", color="#ccccdd")
    ax3.set_title("RPS por Bandeira Vermelha", fontsize=11,
                  fontweight="bold", color="white")
    ax3.tick_params(colors="#ccccdd")
    ax3.grid(axis="y", alpha=0.3)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    fig.suptitle(
        "Impacto de Eventos de Corrida no RPS — Pipeline 3 (OpenF1)",
        fontsize=12, color="white"
    )
    plt.tight_layout()
    _save(fig, save_path)


# ── 4. Probabilidades de vitória — Teste 2025 ────────────────────────────────

def plot_win_probabilities_p3(
    distributions: list[RaceDistribution],
    records:       list[RaceRecord],
    top_drivers:   list[str],
    season:        int,
    save_path:     str,
) -> None:
    """P(vitória) dos top_drivers ao longo das corridas de teste."""
    race_labels = [r.race[:10] for r in records]
    palette     = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    plotted = 0
    for driver in top_drivers:
        win_probs = []
        for dist in distributions:
            vec = dist.vectors.get(driver)
            win_probs.append(float(vec.probs[0]) if vec is not None else np.nan)

        if all(np.isnan(p) for p in win_probs):
            continue

        ax.plot(range(len(win_probs)), win_probs,
                marker="o", ms=5, label=driver,
                color=palette[plotted % 10], alpha=0.9, linewidth=1.8)
        plotted += 1

    ax.set_xticks(range(len(race_labels)))
    ax.set_xticklabels(race_labels, rotation=45, ha="right",
                       fontsize=8, color="#ccccdd")
    ax.set_ylabel("P(vitória)", fontsize=11, color="#ccccdd")
    ax.set_title(
        f"Probabilidade de Vitória — Teste {season}\nPipeline 3 (OpenF1)",
        fontsize=12, color="white"
    )
    ax.legend(title="Piloto", bbox_to_anchor=(1.01, 1),
              loc="upper left", framealpha=0.4)
    ax.grid(alpha=0.3)
    ax.tick_params(colors="#ccccdd")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    plt.tight_layout()
    _save(fig, save_path)


# ── 5. Comparação de RPS entre pipelines ─────────────────────────────────────

def plot_pipeline_comparison(
    val_rps_p3:  RPSSummary,
    test_rps_p3: RPSSummary,
    save_path:   str,
    val_rps_p2:  RPSSummary | None = None,
    test_rps_p2: RPSSummary | None = None,
) -> None:
    """Barras agrupadas: RPS de P2, P3 e baseline."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    phases    = ["Validação", "Teste"]
    p3_vals   = [val_rps_p3.mean_rps_model,    test_rps_p3.mean_rps_model]
    base_vals = [val_rps_p3.mean_rps_baseline,  test_rps_p3.mean_rps_baseline]
    x = np.arange(len(phases))
    w = 0.25

    if val_rps_p2 and test_rps_p2:
        p2_vals = [val_rps_p2.mean_rps_model, test_rps_p2.mean_rps_model]
        ax.bar(x - w, p2_vals,   w, label="Pipeline 2 (CSV)",    color=_BLUE,   alpha=0.85)
        ax.bar(x,     p3_vals,   w, label="Pipeline 3 (OpenF1)", color=_ORANGE, alpha=0.85)
        ax.bar(x + w, base_vals, w, label="Baseline uniforme",   color=_GREY,   alpha=0.7)
    else:
        ax.bar(x - w/2, p3_vals,   w, label="Pipeline 3 (OpenF1)", color=_ORANGE, alpha=0.85)
        ax.bar(x + w/2, base_vals, w, label="Baseline uniforme",   color=_GREY,   alpha=0.7)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=8, padding=2, color="#ccccdd")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Validação\n({val_rps_p3.season})", f"Teste\n({test_rps_p3.season})"],
        color="#ccccdd"
    )
    ax.set_ylabel("RPS médio (↓ melhor)", fontsize=11, color="#ccccdd")
    ax.set_title(
        "Comparação de RPS entre Pipelines\nMenor RPS = previsão probabilística melhor",
        fontsize=12, color="white"
    )
    ax.legend(fontsize=9, framealpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(colors="#ccccdd")
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    plt.tight_layout()
    _save(fig, save_path)
