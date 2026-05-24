"""
src/data_openf1/cache.py
=====================================
Responsabilidade única: gerenciar o cache local dos dados da OpenF1.

Antes de chamar a API, verifica se o arquivo já existe em disco.
Se existir, carrega diretamente. Se não existir, faz a requisição
via openf1_client e salva o resultado.

Estrutura de cache:
    data/openf1/raw/<endpoint>_<chave>.csv

Exemplos:
    data/openf1/raw/meetings_2024.csv
    data/openf1/raw/sessions_meeting_1234.csv
    data/openf1/raw/weather_session_9158.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data_openf1 import client

logger = logging.getLogger(__name__)

# Raiz do repositório: dois níveis acima de src/pipeline_openf1/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR    = _PROJECT_ROOT / "data" / "openf1" / "raw"


def _cache_path(endpoint: str, key: str | int) -> Path:
    """Retorna o Path do arquivo de cache para um endpoint e chave."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{endpoint}_{key}.csv"


def _load_or_fetch(
    endpoint:      str,
    key:           str | int,
    fetch_fn:      Callable[[], pd.DataFrame],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Carrega do cache ou faz a requisição se necessário.

    Parameters
    ----------
    endpoint : str
        Nome lógico do endpoint (usado no nome do arquivo).
    key : str | int
        Chave que diferencia requisições do mesmo endpoint
        (ex: ano, meeting_key, session_key).
    fetch_fn : Callable[[], pd.DataFrame]
        Função sem argumentos que executa a requisição.
    force_refresh : bool
        Se True, ignora o cache e refaz a requisição.
    """
    path = _cache_path(endpoint, key)

    if path.exists() and not force_refresh:
        logger.debug("Cache hit: %s", path.name)
        return pd.read_csv(path)

    logger.info("Requisitando %s (key=%s)...", endpoint, key)
    df = fetch_fn()

    if df.empty:
        logger.warning("Resposta vazia — %s key=%s. Nada salvo em cache.", endpoint, key)
        return df

    df.to_csv(path, index=False)
    logger.info("Salvo em cache: %s (%d linhas)", path.name, len(df))
    return df


# ── API pública do cache ──────────────────────────────────────────────────────

def get_meetings(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Meetings (GPs) de uma temporada, com cache local."""
    return _load_or_fetch("meetings", year,
                          lambda: client.fetch_meetings(year), force_refresh)


def get_sessions(meeting_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Sessões de um GP, com cache local."""
    return _load_or_fetch("sessions_meeting", meeting_key,
                          lambda: client.fetch_sessions(meeting_key), force_refresh)


def get_drivers(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Pilotos de uma sessão, com cache local."""
    return _load_or_fetch("drivers_session", session_key,
                          lambda: client.fetch_drivers(session_key), force_refresh)


def get_session_result(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Resultado oficial de uma sessão de corrida, com cache local."""
    return _load_or_fetch("session_result", session_key,
                          lambda: client.fetch_session_result(session_key), force_refresh)


def get_starting_grid(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Grade de largada de uma corrida, com cache local."""
    return _load_or_fetch("starting_grid", session_key,
                          lambda: client.fetch_starting_grid(session_key), force_refresh)


def get_race_control(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Mensagens de race control, com cache local."""
    return _load_or_fetch("race_control", session_key,
                          lambda: client.fetch_race_control(session_key), force_refresh)


def get_weather(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Dados climáticos de uma sessão, com cache local."""
    return _load_or_fetch("weather", session_key,
                          lambda: client.fetch_weather(session_key), force_refresh)


def get_stints(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Dados de stint de uma sessão, com cache local."""
    return _load_or_fetch("stints", session_key,
                          lambda: client.fetch_stints(session_key), force_refresh)


def get_pit(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Pit stops de uma sessão, com cache local."""
    return _load_or_fetch("pit", session_key,
                          lambda: client.fetch_pit(session_key), force_refresh)


def get_championship_drivers(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Classificação do campeonato de pilotos, com cache local."""
    return _load_or_fetch("championship_drivers", session_key,
                          lambda: client.fetch_championship_drivers(session_key),
                          force_refresh)
