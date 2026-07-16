"""
src/data/repository.py
=======================
Carregador compartilhado de dados (Etapa 2 da reestruturação).

Regra de fonte oficial por período:

    Até 2022  → dataset histórico do projeto (data/Season<ano>/)
    2023+     → dados OpenF1 processados (data/openf1/processed/)

Os pipelines não decidem qual arquivo carregar; essa responsabilidade fica
neste módulo. Antes de usar anos 2023+ é preciso sincronizar:

    python -m src.data_openf1.sync --start-year 2023 --end-year 2025 --from-local-cache
    # ou, com rede:
    python -m src.data_openf1.sync --start-year 2023 --end-year 2026 --profile core

Políticas de fonte (`source_policy`):

    prefer_openf1  (padrão) → 2023+ via OpenF1; se um ano não tiver dados
                              processados, cai para o legado com aviso.
    strict_openf1           → 2023+ somente via OpenF1; ausência de dados
                              gera erro.
    legacy_only             → tudo via CSVs históricos (reprodução do
                              comportamento pré-reestruturação).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.data.data_pipeline import (
    DRIVER_ABBREV,
    RaceRecord,
    get_all_drivers,
    load_seasons,
)
from src.data_openf1.race_mapping import canonical_race_key

logger = logging.getLogger("data.repository")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
PROCESSED_DIR = DEFAULT_DATA_DIR / "openf1" / "processed"

#: Primeiro ano em que a OpenF1 passa a ser a fonte oficial.
OPENF1_FIRST_YEAR = 2023

SOURCE_POLICIES = {"prefer_openf1", "strict_openf1", "legacy_only"}

__all__ = [
    "OPENF1_FIRST_YEAR",
    "SOURCE_POLICIES",
    "RaceRecord",
    "get_all_drivers",
    "load_race_results",
    "load_season_records",
]


# ─────────────────────────────────────────────────────────────────────────────
# Mapeamento chave canônica → rótulo histórico da corrida
# ─────────────────────────────────────────────────────────────────────────────

_label_cache: dict[str, dict[str, str]] = {}


def _legacy_label_map(data_dir: Path) -> dict[str, str]:
    """
    Constrói o mapa canonical_race_key → rótulo usado nos CSVs históricos
    (ex.: 'sakhir' → 'Bahrain'). Garante que corridas vindas da OpenF1
    recebam o mesmo nome usado no histórico, preservando o casamento por
    circuito feito pelo preditor de clusters.
    """
    cache_key = str(data_dir)
    if cache_key in _label_cache:
        return _label_cache[cache_key]

    mapping: dict[str, str] = {}
    roots = [data_dir, data_dir / "legacy" / "reference_2023_2025"]
    for root in roots:
        if not root.is_dir():
            continue
        for season_dir in sorted(root.glob("Season*")):
            for fname in sorted(os.listdir(season_dir)):
                lower = fname.lower()
                if lower.endswith("raceresults.csv") and "sprint" not in lower:
                    try:
                        df = pd.read_csv(season_dir / fname, usecols=["Track"])
                    except Exception:
                        continue
                    for track in df["Track"].dropna().astype(str).unique():
                        key = canonical_race_key(track)
                        if key:
                            # Temporadas mais recentes sobrescrevem rótulos
                            # antigos (glob ordenado por ano).
                            mapping[key] = track.strip()
    _label_cache[cache_key] = mapping
    return mapping


def _display_label(circuit_short_name: str, meeting_name: str,
                   data_dir: Path) -> str:
    key = canonical_race_key(circuit_short_name or meeting_name)
    label = _legacy_label_map(data_dir).get(key)
    if label:
        return label
    fallback = str(meeting_name or circuit_short_name).strip()
    fallback = fallback.replace(" Grand Prix", "").strip()
    return fallback or str(circuit_short_name)


# ─────────────────────────────────────────────────────────────────────────────
# Fonte OpenF1 (2023+)
# ─────────────────────────────────────────────────────────────────────────────

def _openf1_race_results_path(year: int) -> Path:
    return PROCESSED_DIR / "race_results" / f"race_results_{year}.parquet"


def load_openf1_race_results(year: int) -> pd.DataFrame:
    """Lê a tabela processada de resultados de corrida de um ano."""
    path = _openf1_race_results_path(year)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _openf1_records_for_year(
    year: int,
    top_k: int,
    data_dir: Path,
) -> list[RaceRecord]:
    df = load_openf1_race_results(year)
    if df.empty:
        return []

    records: list[RaceRecord] = []
    df = df.copy()
    df["date_start"] = df["date_start"].astype(str)
    session_order = (
        df.groupby("session_key")["date_start"].min().sort_values().index
    )

    for sk in session_order:
        race_df = df[df["session_key"] == sk]
        first = race_df.iloc[0]
        label = _display_label(
            str(first.get("circuit_short_name", "")),
            str(first.get("meeting_name", "")),
            data_dir,
        )

        has_status = race_df["dnf"] | race_df["dns"] | race_df["dsq"]
        # Semântica idêntica ao loader histórico: classificado = posição
        # numérica oficial (mesmo que a flag dnf seja verdadeira, como em
        # abandonos com >90% da prova); cauda = sem posição numérica
        # (DNF/DNS/DSQ não classificados), ordenada pela distância completada.
        classified = (
            race_df[race_df["position_num"].notna()]
            .sort_values("position_num")
        )
        tail = (
            race_df[race_df["position_num"].isna()]
            .sort_values(
                ["number_of_laps", "driver_number"],
                ascending=[False, True],
                na_position="last",
            )
        )

        top_drivers = [d for d in classified["driver"].tolist() if d][:top_k]
        tail_drivers = [d for d in tail["driver"].tolist() if d]
        ranking = top_drivers + tail_drivers

        if len(ranking) >= 2:
            records.append(RaceRecord(
                season=year,
                race=label,
                ranking=ranking,
                n_classified=len(top_drivers),
                n_dnf=len(tail_drivers),
            ))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def load_season_records(
    seasons: list[int],
    *,
    top_k: int = 10,
    source_policy: str = "prefer_openf1",
    data_dir: str | Path | None = None,
) -> tuple[list[RaceRecord], dict[int, str]]:
    """
    Carrega temporadas aplicando a regra de fontes por período.

    Retorna (records, provenance), onde provenance registra a fonte usada
    por ano: 'legacy' ou 'openf1'.
    """
    if source_policy not in SOURCE_POLICIES:
        raise ValueError(
            f"source_policy inválida: {source_policy!r}. "
            f"Opções: {sorted(SOURCE_POLICIES)}"
        )
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    records: list[RaceRecord] = []
    provenance: dict[int, str] = {}

    for year in sorted(seasons):
        use_openf1 = (
            year >= OPENF1_FIRST_YEAR and source_policy != "legacy_only"
        )
        if use_openf1:
            year_records = _openf1_records_for_year(year, top_k, data_dir)
            if year_records:
                provenance[year] = "openf1"
                records.extend(year_records)
                continue
            msg = (
                f"Sem dados OpenF1 processados para {year}. Execute "
                "'python -m src.data_openf1.sync' antes dos pipelines."
            )
            if source_policy == "strict_openf1":
                raise FileNotFoundError(msg)
            logger.warning("%s Usando dataset legado como fallback.", msg)

        year_records = load_seasons(str(data_dir), [year], top_k=top_k)
        provenance[year] = "legacy"
        records.extend(year_records)

    logger.info(
        "Fontes por ano: %s",
        {y: provenance[y] for y in sorted(provenance)},
    )
    return records, provenance


def load_race_results(
    year: int,
    *,
    source_policy: str = "prefer_openf1",
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Interface tabular do plano (seção 4.1): devolve os resultados de corrida
    de um ano como DataFrame, decidindo internamente a fonte.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if year >= OPENF1_FIRST_YEAR and source_policy != "legacy_only":
        df = load_openf1_race_results(year)
        if not df.empty:
            return df
        if source_policy == "strict_openf1":
            raise FileNotFoundError(
                f"Sem race_results processados da OpenF1 para {year}."
            )
        logger.warning("Fallback para CSV legado em %d.", year)

    from src.data.data_pipeline import _find_race_file  # uso interno

    path = _find_race_file(str(data_dir), year)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["season"] = year
    return df
