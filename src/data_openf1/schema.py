"""
src/data_openf1/schema.py
==========================
Responsabilidade única: normalizar os DataFrames retornados pelos
endpoints da OpenF1, garantindo que as colunas esperadas pelo
feature_builder existam independente de mudanças de schema da API.

Schemas reais observados nos dados baixados (maio/2025):

  starting_grid:
    position, driver_number, lap_duration, meeting_key, session_key

  session_result:
    position, driver_number, number_of_laps, points,
    dnf, dns, dsq, duration, gap_to_leader, meeting_key, session_key

  race_control:
    date, driver_number, flag, category, message, scope,
    sector, lap_number, meeting_key, session_key

  drivers:
    driver_number, broadcast_name, full_name, name_acronym,
    team_name, meeting_key, session_key
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ── starting_grid ─────────────────────────────────────────────────────────────

def normalize_starting_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o DataFrame do endpoint starting_grid.

    Garante a coluna 'grid_position' independente do nome real da API:
    - Schema atual  : 'position'
    - Schema antigo : 'grid_position'

    Após normalização, a coluna 'grid_position' conterá inteiros ou NaN.
    """
    if df.empty:
        return df

    # Schema atual da API (maio/2025): coluna 'position'
    if "position" in df.columns and "grid_position" not in df.columns:
        df = df.copy()
        df["grid_position"] = pd.to_numeric(df["position"], errors="coerce")

    elif "grid_position" in df.columns:
        df = df.copy()
        df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce")

    else:
        logger.warning(
            "starting_grid: nenhuma coluna de posição encontrada. "
            "Colunas disponíveis: %s", list(df.columns)
        )
        df = df.copy()
        df["grid_position"] = pd.NA

    return df


# ── session_result ─────────────────────────────────────────────────────────────

def normalize_session_result(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o DataFrame do endpoint session_result.

    Garante a coluna 'dnf_flag' combinando os campos booleanos da API:
    - Schema atual  : colunas 'dnf', 'dns', 'dsq' (booleanos)
    - Schema antigo : coluna 'status' (string como "Finished", "DNF", etc.)

    Após normalização:
    - 'dnf_flag' = 1 se o piloto não completou (dnf|dns|dsq) ou tem status não-finalizador
    - 'dnf_flag' = 0 caso contrário
    - 'position'  = posição final como inteiro (ou NaN)
    """
    if df.empty:
        return df

    df = df.copy()

    # ── posição final ─────────────────────────────────────────────────────────
    if "position" in df.columns:
        df["position"] = pd.to_numeric(df["position"], errors="coerce")

    # ── dnf_flag — schema atual (booleanos) ───────────────────────────────────
    has_bool_cols = any(c in df.columns for c in ("dnf", "dns", "dsq"))

    if has_bool_cols:
        dnf_series = pd.Series(False, index=df.index)
        for col in ("dnf", "dns", "dsq"):
            if col in df.columns:
                # Normaliza: True/False, 1/0, "true"/"false", "True"/"False"
                series = df[col]
                if series.dtype == object:
                    series = series.astype(str).str.strip().str.lower()
                    series = series.map({"true": True, "1": True,
                                         "false": False, "0": False}).fillna(False)
                else:
                    series = series.fillna(False).astype(bool)
                dnf_series = dnf_series | series

        df["dnf_flag"] = dnf_series.astype(int)

    # ── dnf_flag — schema antigo (coluna 'status') ────────────────────────────
    elif "status" in df.columns:
        _FINISHED = frozenset({
            "finished", "1 lap", "+1 lap", "2 laps", "+2 laps",
            "1lap", "2laps", "+1lap", "+2laps",
        })
        df["dnf_flag"] = df["status"].apply(
            lambda s: 0 if str(s).strip().lower() in _FINISHED else 1
        )

    else:
        logger.warning(
            "session_result: sem colunas de DNF reconhecidas. "
            "Colunas disponíveis: %s", list(df.columns)
        )
        df["dnf_flag"] = 0   # conservador: não marcar como DNF sem evidência

    return df


# ── race_control ──────────────────────────────────────────────────────────────

def normalize_race_control(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o DataFrame do endpoint race_control.
    Garante que as colunas 'message', 'flag' e 'category' existam
    (como strings uppercase), mesmo que ausentes no response.
    """
    if df.empty:
        return df

    df = df.copy()
    for col in ("message", "flag", "category"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str).str.upper()

    return df


# ── drivers ────────────────────────────────────────────────────────────────────

def normalize_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o DataFrame do endpoint drivers.
    Garante que 'driver_number' seja inteiro e 'name_acronym' seja string.
    """
    if df.empty:
        return df

    df = df.copy()
    if "driver_number" in df.columns:
        df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")
    if "name_acronym" in df.columns:
        df["name_acronym"] = df["name_acronym"].fillna("").astype(str).str.strip().str.upper()

    return df
