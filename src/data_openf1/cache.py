"""
src/data_openf1/cache.py
=====================================
Gerencia o cache local dos dados da OpenF1.

Politica principal:
    1. Se o arquivo existe e force_refresh=False, usa cache.
    2. Se precisa chamar API e a resposta vem vazia, reaproveita cache antigo
       se existir. Isso evita perder uma temporada inteira por HTTP 429.
    3. Resposta vazia sem cache antigo nao e salva.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data_openf1 import client

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _PROJECT_ROOT / "data" / "openf1" / "raw"


def _cache_path(endpoint: str, key: str | int) -> Path:
    """Retorna o Path do arquivo de cache para um endpoint e chave."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{endpoint}_{key}.csv"


def _read_cache(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df.attrs["cache_path"] = str(path)
        df.attrs["cache_hit"] = True
        return df
    except Exception as exc:  # pragma: no cover - seguranca operacional
        logger.warning("Falha ao ler cache %s: %s", path, exc)
        return pd.DataFrame()


def _load_or_fetch(
    endpoint: str,
    key: str | int,
    fetch_fn: Callable[[], pd.DataFrame],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Carrega do cache ou faz a requisicao se necessario."""
    path = _cache_path(endpoint, key)

    if path.exists() and not force_refresh:
        logger.debug("Cache hit: %s", path.name)
        return _read_cache(path)

    logger.info("Requisitando %s (key=%s)...", endpoint, key)
    df = fetch_fn()

    if df.empty:
        if path.exists():
            logger.warning(
                "Resposta vazia para %s key=%s. Reaproveitando cache antigo: %s",
                endpoint, key, path.name,
            )
            stale = _read_cache(path)
            stale.attrs["stale_cache_used"] = True
            return stale

        logger.warning("Resposta vazia - %s key=%s. Nada salvo em cache.", endpoint, key)
        return df

    df.to_csv(path, index=False)
    df.attrs["cache_path"] = str(path)
    df.attrs["cache_hit"] = False
    logger.info("Salvo em cache: %s (%d linhas)", path.name, len(df))
    return df


# -- API publica do cache -----------------------------------------------------

def get_meetings(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Meetings de uma temporada."""
    return _load_or_fetch("meetings", year, lambda: client.fetch_meetings(year), force_refresh)


def get_sessions(meeting_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Sessoes de um GP."""
    return _load_or_fetch("sessions_meeting", meeting_key, lambda: client.fetch_sessions(meeting_key), force_refresh)


def get_drivers(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Pilotos de uma sessao."""
    return _load_or_fetch("drivers_session", session_key, lambda: client.fetch_drivers(session_key), force_refresh)


def get_session_result(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Resultado oficial de uma sessao."""
    return _load_or_fetch("session_result", session_key, lambda: client.fetch_session_result(session_key), force_refresh)


def get_starting_grid(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Grade de largada de uma sessao de classificacao."""
    return _load_or_fetch("starting_grid", session_key, lambda: client.fetch_starting_grid(session_key), force_refresh)


def get_race_control(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Mensagens de race control."""
    return _load_or_fetch("race_control", session_key, lambda: client.fetch_race_control(session_key), force_refresh)


def get_weather(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Dados climaticos."""
    return _load_or_fetch("weather", session_key, lambda: client.fetch_weather(session_key), force_refresh)


def get_stints(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Dados de stint."""
    return _load_or_fetch("stints", session_key, lambda: client.fetch_stints(session_key), force_refresh)


def get_pit(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Pit stops."""
    return _load_or_fetch("pit", session_key, lambda: client.fetch_pit(session_key), force_refresh)


def get_championship_drivers(session_key: int, force_refresh: bool = False) -> pd.DataFrame:
    """Classificacao do campeonato de pilotos."""
    return _load_or_fetch("championship_drivers", session_key, lambda: client.fetch_championship_drivers(session_key), force_refresh)
