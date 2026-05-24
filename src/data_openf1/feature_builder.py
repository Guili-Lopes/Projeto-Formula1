"""
src/data_openf1/feature_builder.py
====================================
Responsabilidade única: construir a tabela de features contextuais
por corrida a partir dos dados da API OpenF1.

Features geradas (uma linha por corrida):

    Da corrida (race_control):
        sc_count            int   — períodos de Safety Car
        vsc_count           int   — períodos de Virtual Safety Car
        red_flag_count      int   — bandeiras vermelhas
        yellow_flag_count   int   — ocorrências de bandeira amarela (flag="YELLOW")

    Da grade de largada (starting_grid):
        grid_<SIGLA>        int   — posição de largada de cada piloto
                                    ex: grid_VER=1, grid_NOR=2

    Do resultado da sessão (session_result):
        dnf_<SIGLA>         0/1   — 1 se o piloto não completou a corrida
                                    ex: dnf_PER=1

Cobertura:
    OpenF1 tem dados completos a partir de 2023.
    Para 2019-2022, as features não estão disponíveis.

Saída:
    data/openf1/processed/race_context_<ano_ini>_<ano_fim>.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data_openf1 import cache

logger = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "openf1" / "processed"

# Statuses que indicam corrida concluída (não DNF)
_FINISHED_STATUSES = frozenset({
    "FINISHED", "1 LAP", "+1 LAP", "2 LAPS", "+2 LAPS",
    "1LAP", "2LAPS", "+1LAP", "+2LAPS",
})


# ── helpers internos ──────────────────────────────────────────────────────────

def _driver_map(session_key: int) -> dict[int, str]:
    """Retorna {driver_number: name_acronym}."""
    df = cache.get_drivers(session_key)
    if df.empty or "driver_number" not in df.columns:
        return {}
    return {
        int(row["driver_number"]): str(row["name_acronym"])
        for _, row in df.iterrows()
        if pd.notna(row.get("name_acronym"))
    }


def _summarize_race_control(session_key: int) -> dict:
    """
    Conta eventos de controle de corrida a partir do endpoint race_control.

    Contabiliza apenas entradas de evento (não END/CLEAR/ENDING) para
    evitar dupla contagem.

    Retorna:
        sc_count           — períodos de Safety Car
        vsc_count          — períodos de Virtual Safety Car
        red_flag_count     — bandeiras vermelhas
        yellow_flag_count  — ocorrências de bandeira amarela (flag=YELLOW)
    """
    df = cache.get_race_control(session_key)
    if df.empty:
        return {"sc_count": 0, "vsc_count": 0,
                "red_flag_count": 0, "yellow_flag_count": 0}

    msg  = df["message"].str.upper() if "message" in df.columns \
           else pd.Series(dtype=str)
    flag = df["flag"].str.upper()    if "flag"    in df.columns \
           else pd.Series(dtype=str)
    cat  = df["category"].str.upper() if "category" in df.columns \
           else pd.Series(dtype=str)

    sc_count = int(
        (msg.str.contains("SAFETY CAR", na=False)
         & ~msg.str.contains("CLEAR|END|ENDING|VIRTUAL", na=False)).sum()
    )
    vsc_count = int(
        (msg.str.contains("VIRTUAL", na=False)
         & ~msg.str.contains("ENDING|CLEAR", na=False)).sum()
    )
    red_flag_count = int(
        ((cat == "FLAG") & flag.str.contains("RED", na=False)).sum()
    )
    # Bandeiras amarelas: flag = "YELLOW" no endpoint race_control
    yellow_flag_count = int(
        flag.str.contains("YELLOW", na=False).sum()
    )

    return {
        "sc_count":           sc_count,
        "vsc_count":          vsc_count,
        "red_flag_count":     red_flag_count,
        "yellow_flag_count":  yellow_flag_count,
    }


def _get_grid_positions(session_key: int,
                        drv_map: dict[int, str]) -> dict[str, int]:
    """
    Retorna {grid_<SIGLA>: posição_de_largada}.
    Ex: {"grid_VER": 1, "grid_NOR": 2, ...}
    """
    df = cache.get_starting_grid(session_key)
    if df.empty or "driver_number" not in df.columns:
        return {}
    result = {}
    for _, row in df.iterrows():
        sigla = drv_map.get(int(row["driver_number"]))
        if sigla and pd.notna(row.get("grid_position")):
            result[f"grid_{sigla}"] = int(row["grid_position"])
    return result


def _get_dnf_flags(session_key: int,
                   drv_map: dict[int, str]) -> dict[str, int]:
    """
    Retorna {dnf_<SIGLA>: 0 ou 1}.
    1 se o piloto não completou a corrida (DNF, DNS, DSQ).
    Ex: {"dnf_PER": 1, "dnf_VER": 0, ...}
    """
    df = cache.get_session_result(session_key)
    if df.empty or "driver_number" not in df.columns:
        return {}
    result = {}
    for _, row in df.iterrows():
        sigla = drv_map.get(int(row["driver_number"]))
        if sigla:
            status = str(row.get("status", "")).strip().upper()
            result[f"dnf_{sigla}"] = 0 if status in _FINISHED_STATUSES else 1
    return result


def _get_race_session_key(meeting_key: int) -> int | None:
    """Retorna o session_key da corrida principal de um GP."""
    sessions = cache.get_sessions(meeting_key)
    if sessions.empty:
        return None
    race_sess = sessions[
        sessions["session_type"].str.lower().str.contains("race", na=False)
        & ~sessions["session_name"].str.lower().str.contains("sprint", na=False)
    ]
    if race_sess.empty:
        return None
    return int(race_sess.iloc[0]["session_key"])


# ── função principal ──────────────────────────────────────────────────────────

def build_race_context(
    years:         Sequence[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Constrói a tabela de features contextuais para os anos informados.

    Uma linha por corrida. Colunas:
        season, race, meeting_key, session_key
        sc_count, vsc_count, red_flag_count, yellow_flag_count
        grid_<SIGLA>  (uma coluna por piloto)
        dnf_<SIGLA>   (uma coluna por piloto)

    Parameters
    ----------
    years : Sequence[int]
        Ex: [2023, 2024, 2025]
    force_refresh : bool
        Se True, ignora cache e refaz todas as requisições.

    Returns
    -------
    pd.DataFrame  — salvo automaticamente em
        data/openf1/processed/race_context_<ano_ini>_<ano_fim>.csv
    """
    rows = []

    for year in years:
        meetings = cache.get_meetings(year, force_refresh=force_refresh)
        if meetings.empty:
            logger.warning("Sem meetings para %d.", year)
            continue

        for _, mtg in meetings.iterrows():
            circuit     = str(mtg.get("circuit_short_name", ""))
            meeting_key = int(mtg["meeting_key"])

            sk = _get_race_session_key(meeting_key)
            if sk is None:
                continue

            drv_map = _driver_map(sk)

            row: dict = {
                "season":      year,
                "race":        circuit,
                "meeting_key": meeting_key,
                "session_key": sk,
            }

            # ── features de race_control ──────────────────────────────────
            row.update(_summarize_race_control(sk))

            # ── grid de largada: grid_<SIGLA> ─────────────────────────────
            row.update(_get_grid_positions(sk, drv_map))

            # ── status de abandono: dnf_<SIGLA> ───────────────────────────
            row.update(_get_dnf_flags(sk, drv_map))

            rows.append(row)
            logger.info("Processado: %d — %s (sk=%d)", year, circuit, sk)

    if not rows:
        logger.error("Nenhuma corrida processada.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    _save(df, years)
    return df


def _save(df: pd.DataFrame, years: Sequence[int]) -> None:
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = _PROCESSED_DIR / f"race_context_{min(years)}_{max(years)}.csv"
    df.to_csv(path, index=False)
    logger.info("Contexto salvo em %s (%d corridas)", path, len(df))


def load_race_context(years: Sequence[int]) -> pd.DataFrame:
    """
    Carrega a tabela processada do disco se existir,
    caso contrário chama build_race_context.
    """
    path = _PROCESSED_DIR / f"race_context_{min(years)}_{max(years)}.csv"
    if path.exists():
        logger.info("Carregando contexto de %s", path.name)
        return pd.read_csv(path)
    logger.info("Arquivo de contexto não encontrado. Construindo via API...")
    return build_race_context(years)
