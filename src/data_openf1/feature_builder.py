"""
src/data_openf1/feature_builder.py
====================================
Construcao da tabela de contexto OpenF1 por corrida.

A tabela gerada tem uma linha por corrida e deve ser usada com cuidado:
    - grid_* vem de informacao pre-corrida e pode ser usada em previsao.
    - dnf_* e race_control sao informacoes observadas depois da corrida;
      servem para avaliacao/diagnostico e para atualizar historico, mas nao
      devem ser usadas como variaveis preditivas da mesma corrida.

Correcoes implementadas:
    - starting_grid e buscado diretamente pela session_key da classificacao.
    - starting_grid aceita coluna position (schema atual) e grid_position
      (compatibilidade antiga).
    - DNF usa colunas booleanas dnf/dns/dsq; status e apenas fallback.
    - cada corrida recebe race_key canonico para casar nomes historicos e OpenF1.
    - contexto incompleto por temporada nao e salvo silenciosamente, a menos que
      allow_partial=True.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.data_openf1 import cache
from src.data_openf1.race_mapping import add_race_key_from_columns, canonical_race_key

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "openf1" / "processed"
CONTEXT_SCHEMA_VERSION = 2

_FINISHED_STATUSES = frozenset({
    "FINISHED", "FINISH", "CLASSIFIED", "1 LAP", "+1 LAP", "2 LAPS", "+2 LAPS",
    "3 LAPS", "+3 LAPS", "1LAP", "+1LAP", "2LAPS", "+2LAPS",
})


def _text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(["" for _ in range(len(df))], index=df.index, dtype=str)
    return df[col].fillna("").astype(str).str.strip()


def _sort_sessions(df: pd.DataFrame) -> pd.DataFrame:
    if "date_start" in df.columns:
        tmp = df.copy()
        tmp["_date_start_sort"] = pd.to_datetime(tmp["date_start"], errors="coerce")
        return tmp.sort_values("_date_start_sort").drop(columns=["_date_start_sort"])
    return df


# -- helpers de sessao --------------------------------------------------------

def _filter_race_session(sessions: pd.DataFrame) -> pd.DataFrame:
    """Filtra a sessao da corrida principal, excluindo sprint."""
    if sessions.empty:
        return pd.DataFrame()

    sessions = _sort_sessions(sessions)
    name = _text_series(sessions, "session_name").str.upper()
    stype = _text_series(sessions, "session_type").str.upper()

    exact = sessions[name == "RACE"]
    if not exact.empty:
        return exact

    filtered = sessions[
        stype.str.contains("RACE", na=False)
        & ~name.str.contains("SPRINT", na=False)
    ]
    return filtered if not filtered.empty else pd.DataFrame()


def _filter_qualifying_session(sessions: pd.DataFrame) -> pd.DataFrame:
    """Filtra a classificacao principal, excluindo sprint shootout/qualifying."""
    if sessions.empty:
        return pd.DataFrame()

    sessions = _sort_sessions(sessions)
    name = _text_series(sessions, "session_name").str.upper()
    stype = _text_series(sessions, "session_type").str.upper()

    exact = sessions[name == "QUALIFYING"]
    if not exact.empty:
        return exact

    filtered = sessions[
        stype.str.contains("QUALIFYING", na=False)
        & ~name.str.contains("SPRINT|SHOOTOUT", na=False)
    ]
    return filtered if not filtered.empty else pd.DataFrame()


# -- helpers de dados ---------------------------------------------------------

def _driver_map(session_key: int, force_refresh: bool = False) -> dict[int, str]:
    """Retorna {driver_number: name_acronym}."""
    df = cache.get_drivers(session_key, force_refresh=force_refresh)
    if df.empty or "driver_number" not in df.columns:
        return {}

    result: dict[int, str] = {}
    for _, row in df.iterrows():
        number = row.get("driver_number")
        acronym = row.get("name_acronym")
        if pd.notna(number) and pd.notna(acronym):
            try:
                result[int(number)] = str(acronym).strip().upper()
            except (TypeError, ValueError):
                continue
    return result


def _summarize_race_control(session_key: int, force_refresh: bool = False) -> dict[str, int]:
    """Conta eventos de race control para diagnostico pos-corrida."""
    df = cache.get_race_control(session_key, force_refresh=force_refresh)
    if df.empty:
        return {
            "sc_count": 0,
            "vsc_count": 0,
            "red_flag_count": 0,
            "yellow_flag_count": 0,
            "has_race_control": 0,
        }

    msg = _text_series(df, "message").str.upper()
    flag = _text_series(df, "flag").str.upper()
    cat = _text_series(df, "category").str.upper()

    sc_mask = (
        msg.str.contains("SAFETY CAR", na=False)
        & ~msg.str.contains("VIRTUAL", na=False)
        & ~msg.str.contains("CLEAR|END|ENDING|ENDS|IN THIS LAP", na=False)
    )
    vsc_mask = (
        msg.str.contains("VIRTUAL", na=False)
        & ~msg.str.contains("ENDING|ENDED|CLEAR|END", na=False)
    )
    red_mask = (cat == "FLAG") & flag.str.contains("RED", na=False)
    yellow_mask = flag.str.contains("YELLOW", na=False)

    return {
        "sc_count": int(sc_mask.sum()),
        "vsc_count": int(vsc_mask.sum()),
        "red_flag_count": int(red_mask.sum()),
        "yellow_flag_count": int(yellow_mask.sum()),
        "has_race_control": 1,
    }


def _safe_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _grid_from_starting_grid(
    session_key: int,
    drv_map: dict[int, str],
    force_refresh: bool = False,
) -> dict[str, int]:
    """Obtem grid via endpoint starting_grid."""
    df = cache.get_starting_grid(session_key, force_refresh=force_refresh)
    if df.empty or "driver_number" not in df.columns:
        return {}

    pos_col = "position" if "position" in df.columns else "grid_position" if "grid_position" in df.columns else None
    if pos_col is None:
        logger.warning("starting_grid %s sem coluna position/grid_position.", session_key)
        return {}

    result: dict[str, int] = {}
    for _, row in df.iterrows():
        number = _safe_int(row.get("driver_number"))
        pos = _safe_int(row.get(pos_col))
        if number is None or pos is None:
            continue
        sigla = drv_map.get(number)
        if sigla:
            result[sigla] = pos
    return result


def _grid_from_qualifying_result(
    session_key: int,
    drv_map: dict[int, str],
    force_refresh: bool = False,
) -> dict[str, int]:
    """Fallback: posicao final da classificacao como aproximacao do grid."""
    df = cache.get_session_result(session_key, force_refresh=force_refresh)
    if df.empty or "driver_number" not in df.columns or "position" not in df.columns:
        return {}

    result: dict[str, int] = {}
    for _, row in df.iterrows():
        number = _safe_int(row.get("driver_number"))
        pos = _safe_int(row.get("position"))
        if number is None or pos is None:
            continue
        sigla = drv_map.get(number)
        if sigla:
            result[sigla] = pos
    return result


def _get_grid_with_fallback(
    quali_sk: int | None,
    drv_map: dict[int, str],
    force_refresh: bool = False,
) -> tuple[dict[str, int], str]:
    """
    Obtem grid com politica segura:
        1. starting_grid da Qualifying.
        2. session_result da Qualifying como fallback.

    Nao usa starting_grid com session_key da Race, pois esse padrao gerava 404
    repetidos e desnecessarios.
    """
    if quali_sk is None:
        return {}, "unavailable"

    grid = _grid_from_starting_grid(quali_sk, drv_map, force_refresh=force_refresh)
    if grid:
        return grid, "starting_grid_quali"

    grid = _grid_from_qualifying_result(quali_sk, drv_map, force_refresh=force_refresh)
    if grid:
        return grid, "qualifying_fallback"

    return {}, "unavailable"


def _bool_like(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value))
    return str(value).strip().lower() in {"true", "1", "yes", "y", "sim"}


def _get_dnf_flags_from_result(
    result_df: pd.DataFrame,
    drv_map: dict[int, str],
) -> dict[str, int]:
    """Retorna {sigla: 1 se DNF/DNS/DSQ, 0 caso contrario}."""
    if result_df.empty or "driver_number" not in result_df.columns:
        return {}

    has_bool_schema = any(col in result_df.columns for col in ("dnf", "dns", "dsq"))
    result: dict[str, int] = {}

    for _, row in result_df.iterrows():
        number = _safe_int(row.get("driver_number"))
        if number is None:
            continue
        sigla = drv_map.get(number)
        if not sigla:
            continue

        if has_bool_schema:
            result[sigla] = int(
                _bool_like(row.get("dnf", False))
                or _bool_like(row.get("dns", False))
                or _bool_like(row.get("dsq", False))
            )
        else:
            status = str(row.get("status", "")).strip().upper()
            result[sigla] = 0 if status in _FINISHED_STATUSES else 1

    return result


def _expected_is_satisfied(df: pd.DataFrame, min_races_by_year: Mapping[int, int] | None) -> bool:
    if not min_races_by_year:
        return True
    if df.empty or "season" not in df.columns:
        return False
    for year, expected in min_races_by_year.items():
        if expected <= 0:
            continue
        found = int((df["season"] == year).sum())
        if found < expected:
            logger.warning("Contexto OpenF1 incompleto para %d: %d/%d corridas.", year, found, expected)
            return False
    return True


# -- funcao principal ---------------------------------------------------------

def build_race_context(
    years: Sequence[int],
    force_refresh: bool = False,
    allow_partial: bool = False,
    min_races_by_year: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Constroi a tabela de features contextuais para os anos informados."""
    rows: list[dict] = []
    failed_years: list[int] = []

    for year in years:
        meetings = cache.get_meetings(year, force_refresh=force_refresh)
        if meetings.empty:
            logger.warning("Sem meetings para %d.", year)
            failed_years.append(year)
            continue

        processed_this_year = 0

        for _, mtg in meetings.iterrows():
            meeting_key_raw = mtg.get("meeting_key")
            if pd.isna(meeting_key_raw):
                continue
            meeting_key = int(meeting_key_raw)
            circuit = str(mtg.get("circuit_short_name", "")).strip()
            meeting_name = str(mtg.get("meeting_name", "")).strip()
            race_key = add_race_key_from_columns(mtg)

            sessions = cache.get_sessions(meeting_key, force_refresh=force_refresh)
            if sessions.empty:
                logger.warning("Sem sessions para %d - %s (meeting_key=%s).", year, circuit, meeting_key)
                continue

            race_sess = _filter_race_session(sessions)
            if race_sess.empty:
                logger.debug("Corrida principal nao encontrada: %d - %s", year, circuit)
                continue
            race_sk = int(race_sess.iloc[0]["session_key"])

            race_result = cache.get_session_result(race_sk, force_refresh=force_refresh)
            if race_result.empty:
                logger.info("Ignorado %d - %s (sk=%s): sem session_result de Race", year, circuit, race_sk)
                continue

            quali_sess = _filter_qualifying_session(sessions)
            quali_sk = int(quali_sess.iloc[0]["session_key"]) if not quali_sess.empty else None

            drv_map = _driver_map(race_sk, force_refresh=force_refresh)
            if not drv_map and quali_sk is not None:
                drv_map = _driver_map(quali_sk, force_refresh=force_refresh)

            row: dict = {
                "context_schema_version": CONTEXT_SCHEMA_VERSION,
                "season": year,
                "race": circuit,
                "race_key": canonical_race_key(race_key or circuit or meeting_name),
                "meeting_name": meeting_name,
                "meeting_key": meeting_key,
                "session_key": race_sk,
                "quali_session_key": quali_sk,
                "has_session_result": 1,
            }

            row.update(_summarize_race_control(race_sk, force_refresh=force_refresh))

            grid, grid_source = _get_grid_with_fallback(quali_sk, drv_map, force_refresh=force_refresh)
            row["grid_source"] = grid_source
            row["grid_driver_count"] = len(grid)
            for sigla, pos in grid.items():
                row[f"grid_{sigla}"] = pos

            dnf_flags = _get_dnf_flags_from_result(race_result, drv_map)
            row["dnf_driver_count"] = len(dnf_flags)
            for sigla, flag in dnf_flags.items():
                row[f"dnf_{sigla}"] = flag

            rows.append(row)
            processed_this_year += 1
            logger.info(
                "Processado: %d - %s (race_key=%s, Race sk=%d, Quali sk=%s, grid=%s)",
                year,
                circuit,
                row["race_key"],
                race_sk,
                str(quali_sk) if quali_sk is not None else "N/A",
                grid_source,
            )

        expected = min_races_by_year.get(year) if min_races_by_year else None
        if expected is not None and processed_this_year < expected:
            logger.warning(
                "Ano %d processado parcialmente: %d/%d corridas esperadas.",
                year,
                processed_this_year,
                expected,
            )

    if not rows:
        logger.error("Nenhuma corrida processada.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    incomplete = failed_years or not _expected_is_satisfied(df, min_races_by_year)
    if incomplete and not allow_partial:
        detail = f"failed_years={failed_years}" if failed_years else "cobertura abaixo do esperado"
        raise RuntimeError(
            "Contexto OpenF1 incompleto; arquivo processado nao foi salvo. "
            f"Detalhe: {detail}. Use allow_partial=True apenas para diagnostico."
        )

    _save(df, years)
    return df


def _save(df: pd.DataFrame, years: Sequence[int]) -> None:
    """Salva a tabela processada em disco."""
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = _PROCESSED_DIR / f"race_context_{min(years)}_{max(years)}.csv"
    df.to_csv(path, index=False)
    logger.info("Contexto salvo em %s (%d corridas)", path, len(df))


def load_race_context(
    years: Sequence[int],
    force_refresh: bool = False,
    allow_partial: bool = False,
    min_races_by_year: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """
    Carrega a tabela processada do disco se ela existir e passar na validacao.
    Caso contrario, reconstrui via API/cache bruto.
    """
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = _PROCESSED_DIR / f"race_context_{min(years)}_{max(years)}.csv"

    if path.exists() and not force_refresh:
        logger.info("Carregando contexto de %s", path.name)
        df = pd.read_csv(path)
        if "race_key" not in df.columns and "race" in df.columns:
            df["race_key"] = df["race"].map(canonical_race_key)

        schema_ok = (
            "context_schema_version" in df.columns
            and set(df["context_schema_version"].dropna().astype(int).unique()) == {CONTEXT_SCHEMA_VERSION}
        )
        if not schema_ok:
            logger.warning("Contexto processado usa schema antigo. Reconstruindo.")
        elif _expected_is_satisfied(df, min_races_by_year) or allow_partial:
            return df
        else:
            logger.warning("Contexto processado existe, mas esta incompleto. Reconstruindo.")

    logger.info("Arquivo nao encontrado, antigo ou incompleto. Construindo via API/cache bruto...")
    return build_race_context(
        years=years,
        force_refresh=force_refresh,
        allow_partial=allow_partial,
        min_races_by_year=min_races_by_year,
    )
