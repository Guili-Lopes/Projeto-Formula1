"""
src/data_openf1/coverage_report.py
===================================
Relatorio de cobertura do contexto OpenF1 contra as corridas historicas usadas
no Pipeline 3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.data_openf1.race_mapping import canonical_race_key

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "openf1" / "processed"


def _as_int(value: object, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _match_context(ctx_df: pd.DataFrame, season: int, race: str) -> pd.Series | None:
    if ctx_df is None or ctx_df.empty:
        return None

    key = canonical_race_key(race)
    df = ctx_df.copy()
    if "race_key" not in df.columns and "race" in df.columns:
        df["race_key"] = df["race"].map(canonical_race_key)

    exact = df[(df["season"] == season) & (df["race_key"] == key)]
    if not exact.empty:
        return exact.iloc[0]

    # Fallback conservador para arquivos antigos sem race_key confiavel.
    if "race" in df.columns:
        old = df[
            (df["season"] == season)
            & (df["race"].fillna("").astype(str).str.lower() == str(race).lower())
        ]
        if not old.empty:
            return old.iloc[0]

    return None


def create_coverage_report(
    records: Sequence[Any],
    ctx_df: pd.DataFrame,
    years: Iterable[int] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Cria e opcionalmente salva um CSV de cobertura."""
    years_set = set(years) if years is not None else None
    rows: list[dict[str, object]] = []

    for rec in records:
        if years_set is not None and rec.season not in years_set:
            continue

        row_ctx = _match_context(ctx_df, rec.season, rec.race)
        race_key = canonical_race_key(rec.race)

        base = {
            "season": rec.season,
            "race": rec.race,
            "race_key": race_key,
            "has_context": 0,
            "openf1_race": "",
            "grid_source": "missing_context",
            "has_grid": 0,
            "grid_driver_count": 0,
            "has_session_result": 0,
            "has_race_control": 0,
            "dnf_driver_count": 0,
            "missing_reason": "no_context_row",
        }

        if row_ctx is None:
            rows.append(base)
            continue

        grid_cols = [c for c in row_ctx.index if str(c).startswith("grid_") and c not in {"grid_source", "grid_driver_count"}]
        grid_count = int(sum(pd.notna(row_ctx[c]) for c in grid_cols))
        dnf_cols = [c for c in row_ctx.index if str(c).startswith("dnf_") and c != "dnf_driver_count"]
        dnf_count = int(sum(pd.notna(row_ctx[c]) for c in dnf_cols))
        grid_source = str(row_ctx.get("grid_source", "unavailable"))

        if grid_source == "unavailable" or grid_count == 0:
            reason = "context_without_grid"
        else:
            reason = "matched"

        base.update({
            "has_context": 1,
            "openf1_race": str(row_ctx.get("race", "")),
            "openf1_race_key": str(row_ctx.get("race_key", race_key)),
            "grid_source": grid_source,
            "has_grid": int(grid_count > 0 and grid_source != "unavailable"),
            "grid_driver_count": _as_int(row_ctx.get("grid_driver_count", grid_count), grid_count),
            "has_session_result": _as_int(row_ctx.get("has_session_result", 0)),
            "has_race_control": _as_int(row_ctx.get("has_race_control", 0)),
            "dnf_driver_count": _as_int(row_ctx.get("dnf_driver_count", dnf_count), dnf_count),
            "missing_reason": reason,
        })
        rows.append(base)

    report = pd.DataFrame(rows)

    if save and not report.empty:
        _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        min_year = int(report["season"].min())
        max_year = int(report["season"].max())
        path = _PROCESSED_DIR / f"openf1_coverage_report_{min_year}_{max_year}.csv"
        report.to_csv(path, index=False)
        logger.info("Coverage report salvo em %s (%d corridas)", path, len(report))

    return report


def print_coverage_summary(report: pd.DataFrame) -> None:
    """Imprime resumo de cobertura no terminal."""
    if report is None or report.empty:
        print("\n  Coverage report vazio.")
        return

    total = len(report)
    with_context = int(report["has_context"].sum())
    with_grid = int(report["has_grid"].sum())
    official_grid = int((report["grid_source"] == "starting_grid_quali").sum())
    fallback_grid = int((report["grid_source"] == "qualifying_fallback").sum())
    unavailable = total - with_grid

    print("\n" + "=" * 60)
    print("  COBERTURA DO CONTEXTO OPENF1")
    print("=" * 60)
    print(f"  Corridas historicas        : {total}")
    print(f"  Com contexto OpenF1        : {with_context} ({with_context / total:.0%})")
    print(f"  Com grid de piloto         : {with_grid} ({with_grid / total:.0%})")
    print(f"    |- starting_grid oficial : {official_grid}")
    print(f"    |- qualifying fallback   : {fallback_grid}")
    print(f"    |- indisponivel/sem match: {unavailable}")

    reason_counts = report["missing_reason"].value_counts().to_dict()
    if reason_counts:
        print("\n  Status:")
        for reason, count in reason_counts.items():
            print(f"    {reason:22s}: {count}")

    missing = report[report["missing_reason"] != "matched"]
    if not missing.empty:
        print("\n  Corridas que exigem atencao:")
        for _, row in missing.iterrows():
            print(f"    {int(row['season'])} - {row['race']} ({row['missing_reason']})")
    print("=" * 60)
