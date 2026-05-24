"""
src/data_openf1/client.py
=====================================
Responsabilidade única: encapsular todas as chamadas HTTP à API OpenF1.

Cada função faz uma requisição GET, trata erros de rede e retorna
um DataFrame. Nenhuma função escreve em disco — isso é responsabilidade
do openf1_cache.py.

Referência da API: https://openf1.org/
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

BASE_URL     = "https://api.openf1.org/v1"
TIMEOUT      = 20      # segundos por tentativa
MAX_RETRIES  = 3
RETRY_WAIT   = 2.0     # segundos entre tentativas

logger = logging.getLogger(__name__)


# ── utilitário interno ────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Executa GET no endpoint e retorna DataFrame.
    Faz até MAX_RETRIES tentativas em caso de erro de rede.
    Retorna DataFrame vazio em caso de falha definitiva.
    """
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return pd.DataFrame(data) if data else pd.DataFrame()
        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "HTTP %s — %s (tentativa %d/%d)",
                exc.response.status_code, url, attempt, MAX_RETRIES
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("Erro de rede — %s (tentativa %d/%d): %s",
                           url, attempt, MAX_RETRIES, exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_WAIT)

    logger.error("Falha após %d tentativas: %s", MAX_RETRIES, url)
    return pd.DataFrame()


# ── endpoints públicos ────────────────────────────────────────────────────────

def fetch_meetings(year: int) -> pd.DataFrame:
    """
    Grandes Prêmios de uma temporada.
    Colunas: meeting_key, meeting_name, circuit_short_name,
             country_name, date_start, year.
    """
    return _get("meetings", {"year": year})


def fetch_sessions(meeting_key: int) -> pd.DataFrame:
    """
    Sessões de um Grande Prêmio (treinos, quali, sprint, corrida).
    Colunas: session_key, session_name, session_type,
             date_start, date_end, meeting_key.
    """
    return _get("sessions", {"meeting_key": meeting_key})


def fetch_drivers(session_key: int) -> pd.DataFrame:
    """
    Pilotos inscritos em uma sessão.
    Colunas: driver_number, name_acronym, full_name,
             broadcast_name, team_name, session_key.
    """
    return _get("drivers", {"session_key": session_key})


def fetch_session_result(session_key: int) -> pd.DataFrame:
    """
    Resultado oficial de uma sessão de corrida.
    Colunas: driver_number, position, status, points, session_key.
    """
    return _get("session_result", {"session_key": session_key})


def fetch_starting_grid(session_key: int) -> pd.DataFrame:
    """
    Grade de largada da corrida.
    Colunas: driver_number, grid_position, session_key.
    """
    return _get("starting_grid", {"session_key": session_key})


def fetch_laps(session_key: int,
               driver_number: int | None = None) -> pd.DataFrame:
    """
    Dados volta a volta de uma sessão.
    Colunas: driver_number, lap_number, lap_duration,
             duration_sector_1/2/3, st_speed, is_pit_out_lap.
    """
    params: dict[str, Any] = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    return _get("laps", params)


def fetch_stints(session_key: int) -> pd.DataFrame:
    """
    Stints de pneu de uma sessão.
    Colunas: driver_number, stint_number, lap_start, lap_end,
             compound, tyre_age_at_start, session_key.
    """
    return _get("stints", {"session_key": session_key})


def fetch_pit(session_key: int) -> pd.DataFrame:
    """
    Pit stops de uma sessão.
    Colunas: driver_number, lap_number, pit_duration, date, session_key.
    """
    return _get("pit", {"session_key": session_key})


def fetch_position(session_key: int,
                   driver_number: int | None = None) -> pd.DataFrame:
    """
    Evolução de posições em pista durante a sessão.
    Colunas: driver_number, position, date, session_key.
    """
    params: dict[str, Any] = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    return _get("position", params)


def fetch_intervals(session_key: int) -> pd.DataFrame:
    """
    Intervalos (gaps) entre carros durante a sessão.
    Colunas: driver_number, gap_to_leader, interval, date, session_key.
    """
    return _get("intervals", {"session_key": session_key})


def fetch_race_control(session_key: int) -> pd.DataFrame:
    """
    Mensagens de controle de corrida (safety car, bandeiras, punições).
    Colunas: date, driver_number, flag, category, message,
             scope, session_key.
    """
    return _get("race_control", {"session_key": session_key})


def fetch_weather(session_key: int) -> pd.DataFrame:
    """
    Dados climáticos amostrados ao longo da sessão.
    Colunas: date, air_temperature, track_temperature,
             humidity, rainfall, wind_speed, session_key.
    """
    return _get("weather", {"session_key": session_key})


def fetch_championship_drivers(session_key: int) -> pd.DataFrame:
    """
    Classificação do campeonato de pilotos após a sessão.
    Colunas: driver_number, position, points, session_key.
    """
    return _get("championship_drivers", {"session_key": session_key})


def fetch_championship_teams(session_key: int) -> pd.DataFrame:
    """
    Classificação do campeonato de construtores após a sessão.
    Colunas: team_name, position, points, session_key.
    """
    return _get("championship_teams", {"session_key": session_key})
