"""
src/data_openf1/client.py
=====================================
Cliente HTTP da API OpenF1.

Principais cuidados implementados:
    - 404 e 400 sao tratados como ausencia permanente de dado e nao sao
      repetidos inutilmente.
    - 429 recebe backoff exponencial e respeita Retry-After quando a API
      retornar esse cabecalho.
    - erros 5xx e falhas de rede sao repetidos com espera progressiva.
    - todas as funcoes retornam DataFrame; nenhuma funcao escreve em disco.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
TIMEOUT = int(os.getenv("OPENF1_TIMEOUT", "20"))
MAX_RETRIES = int(os.getenv("OPENF1_MAX_RETRIES", "6"))
BASE_RETRY_WAIT = float(os.getenv("OPENF1_BASE_RETRY_WAIT", "3"))
MAX_RETRY_WAIT = float(os.getenv("OPENF1_MAX_RETRY_WAIT", "120"))

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Projeto-Formula1-TCC/1.0 (+https://github.com/Guili-Lopes/Projeto-Formula1)",
    "Accept": "application/json",
})


def _sleep_seconds(resp: requests.Response | None, attempt: int, rate_limited: bool) -> float:
    """Calcula espera antes da proxima tentativa."""
    if resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), MAX_RETRY_WAIT)
            except ValueError:
                pass

    multiplier = 3.0 if rate_limited else 1.0
    wait = BASE_RETRY_WAIT * multiplier * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, min(1.0, wait * 0.10))
    return min(wait + jitter, MAX_RETRY_WAIT)


def _format_params(params: dict[str, Any] | None) -> str:
    if not params:
        return ""
    return "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))


# -- utilitario interno -------------------------------------------------------

def _get(endpoint: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Executa GET no endpoint e retorna DataFrame.

    Politica de retry:
        200 -> DataFrame com JSON ou vazio.
        400/404 -> vazio imediato, pois geralmente indica dado inexistente
                   para a chave pedida.
        429 -> retry com backoff longo.
        5xx/rede -> retry com backoff progressivo.
    """
    url = f"{BASE_URL}/{endpoint}"
    full_url = f"{url}{_format_params(params)}"

    # Etapa 2 da reestruturação: com OPENF1_OFFLINE=1, nenhuma chamada de rede
    # é permitida. Somente o sincronizador central (src.data_openf1.sync) pode
    # acessar a API; pipelines devem consumir dados já armazenados em disco.
    if os.getenv("OPENF1_OFFLINE") == "1":
        logger.warning(
            "OPENF1_OFFLINE=1 — chamada bloqueada: %s. Use "
            "'python -m src.data_openf1.sync' para sincronizar os dados.",
            full_url,
        )
        df = pd.DataFrame()
        df.attrs["openf1_status_code"] = None
        df.attrs["openf1_endpoint"] = endpoint
        df.attrs["openf1_params"] = params or {}
        df.attrs["openf1_offline_blocked"] = True
        return df

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _SESSION.get(url, params=params, timeout=TIMEOUT)
            status = resp.status_code

            if status == 200:
                data = resp.json()
                df = pd.DataFrame(data) if data else pd.DataFrame()
                df.attrs["openf1_status_code"] = 200
                df.attrs["openf1_endpoint"] = endpoint
                df.attrs["openf1_params"] = params or {}
                return df

            if status in {400, 404}:
                logger.warning("HTTP %s - %s. Dado ausente; sem novas tentativas.", status, full_url)
                df = pd.DataFrame()
                df.attrs["openf1_status_code"] = status
                df.attrs["openf1_endpoint"] = endpoint
                df.attrs["openf1_params"] = params or {}
                return df

            if status == 429:
                wait = _sleep_seconds(resp, attempt, rate_limited=True)
                logger.warning(
                    "HTTP 429 - %s (tentativa %d/%d). Aguardando %.1fs.",
                    full_url, attempt, MAX_RETRIES, wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                    continue
                break

            if 500 <= status < 600:
                wait = _sleep_seconds(resp, attempt, rate_limited=False)
                logger.warning(
                    "HTTP %s - %s (tentativa %d/%d). Aguardando %.1fs.",
                    status, full_url, attempt, MAX_RETRIES, wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                    continue
                break

            logger.warning("HTTP %s - %s. Status nao recuperavel.", status, full_url)
            df = pd.DataFrame()
            df.attrs["openf1_status_code"] = status
            df.attrs["openf1_endpoint"] = endpoint
            df.attrs["openf1_params"] = params or {}
            return df

        except requests.exceptions.RequestException as exc:
            wait = _sleep_seconds(None, attempt, rate_limited=False)
            logger.warning(
                "Erro de rede - %s (tentativa %d/%d): %s. Aguardando %.1fs.",
                full_url, attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
                continue
            break

    logger.error("Falha apos %d tentativas: %s", MAX_RETRIES, full_url)
    df = pd.DataFrame()
    df.attrs["openf1_status_code"] = 429
    df.attrs["openf1_endpoint"] = endpoint
    df.attrs["openf1_params"] = params or {}
    return df


# -- endpoints publicos -------------------------------------------------------

def fetch_meetings(year: int) -> pd.DataFrame:
    """Grandes Premios de uma temporada."""
    return _get("meetings", {"year": year})


def fetch_sessions(meeting_key: int) -> pd.DataFrame:
    """Sessoes de um Grande Premio."""
    return _get("sessions", {"meeting_key": meeting_key})


def fetch_drivers(session_key: int) -> pd.DataFrame:
    """Pilotos inscritos em uma sessao."""
    return _get("drivers", {"session_key": session_key})


def fetch_session_result(session_key: int) -> pd.DataFrame:
    """
    Resultado oficial de uma sessao.

    Campos relevantes na OpenF1 atual incluem: driver_number, position,
    dnf, dns, dsq, session_key.
    """
    return _get("session_result", {"session_key": session_key})


def fetch_starting_grid(session_key: int) -> pd.DataFrame:
    """
    Grade de largada.

    Na OpenF1 atual, a posicao vem na coluna ``position``. O codigo do
    feature_builder tambem aceita ``grid_position`` por compatibilidade.
    """
    return _get("starting_grid", {"session_key": session_key})


def fetch_laps(session_key: int, driver_number: int | None = None) -> pd.DataFrame:
    """Dados volta a volta de uma sessao."""
    params: dict[str, Any] = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    return _get("laps", params)


def fetch_stints(session_key: int) -> pd.DataFrame:
    """Stints de pneu de uma sessao."""
    return _get("stints", {"session_key": session_key})


def fetch_pit(session_key: int) -> pd.DataFrame:
    """Pit stops de uma sessao."""
    return _get("pit", {"session_key": session_key})


def fetch_position(session_key: int, driver_number: int | None = None) -> pd.DataFrame:
    """Evolucao de posicoes em pista durante a sessao."""
    params: dict[str, Any] = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    return _get("position", params)


def fetch_intervals(session_key: int) -> pd.DataFrame:
    """Intervalos entre carros durante a sessao."""
    return _get("intervals", {"session_key": session_key})


def fetch_race_control(session_key: int) -> pd.DataFrame:
    """Mensagens de controle de corrida."""
    return _get("race_control", {"session_key": session_key})


def fetch_weather(session_key: int) -> pd.DataFrame:
    """Dados climaticos amostrados ao longo da sessao."""
    return _get("weather", {"session_key": session_key})


def fetch_championship_drivers(session_key: int) -> pd.DataFrame:
    """Classificacao do campeonato de pilotos apos a sessao."""
    return _get("championship_drivers", {"session_key": session_key})


def fetch_championship_teams(session_key: int) -> pd.DataFrame:
    """Classificacao do campeonato de construtores apos a sessao."""
    return _get("championship_teams", {"session_key": session_key})
