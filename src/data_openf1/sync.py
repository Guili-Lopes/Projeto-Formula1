"""
src/data_openf1/sync.py
========================
Sincronizador central da OpenF1 (Etapa 2 da reestruturação).

Regra da reestruturação:

    Nenhum pipeline deverá chamar a API OpenF1 diretamente.
    Somente esta camada de sincronização pode acessar a API.

Fluxo:

    OpenF1 API → sincronizador central → data/openf1/raw/<ano>/
        → processamento e padronização → data/openf1/processed/
        → carregador compartilhado (src.data.repository) → pipelines

Uso:

    # Sincronização completa (perfil full)
    python -m src.data_openf1.sync --start-year 2023 --end-year 2026 --profile full

    # Atualizar apenas 2025, ignorando arquivos existentes
    python -m src.data_openf1.sync --start-year 2025 --end-year 2025 --force-refresh

    # Atualização incremental de 2026 (baixa somente o que falta)
    python -m src.data_openf1.sync --start-year 2026 --end-year 2026 --incremental

    # Importar o cache local legado (sem rede) e gerar processed + manifests
    python -m src.data_openf1.sync --start-year 2023 --end-year 2025 --from-local-cache

Perfis:

    core : dados essenciais para os pipelines atuais
    full : core + endpoints de alto volume (laps, position, intervals)

Saídas:

    data/openf1/raw/<ano>/<endpoint>/<endpoint>_<chave>.csv
    data/openf1/processed/<tabela>/<tabela>_<ano>.parquet (+ espelho .csv)
    data/openf1/manifests/sync_<ano>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_openf1 import client
from src.data_openf1.race_mapping import canonical_race_key

logger = logging.getLogger("openf1.sync")

OPENF1_DIR = _PROJECT_ROOT / "data" / "openf1"
RAW_DIR = OPENF1_DIR / "raw"
PROCESSED_DIR = OPENF1_DIR / "processed"
MANIFESTS_DIR = OPENF1_DIR / "manifests"

# Tipos de sessão sincronizados por padrão. Qualifying entra porque o grid de
# largada usa o resultado da classificação como fallback.
SESSION_NAME_TARGETS = {
    "Race",
    "Sprint",
    "Qualifying",
    "Sprint Qualifying",
    "Sprint Shootout",
}

# Endpoints por escopo. Nomes seguem src/data_openf1/client.py.
_SESSION_FETCHERS: dict[str, Callable[[int], pd.DataFrame]] = {
    "drivers": client.fetch_drivers,
    "session_result": client.fetch_session_result,
    "starting_grid": client.fetch_starting_grid,
    "race_control": client.fetch_race_control,
    "weather": client.fetch_weather,
    "pit": client.fetch_pit,
    "stints": client.fetch_stints,
    "championship_drivers": client.fetch_championship_drivers,
    "championship_teams": client.fetch_championship_teams,
    "laps": client.fetch_laps,
    "position": client.fetch_position,
    "intervals": client.fetch_intervals,
}

PROFILES: dict[str, list[str]] = {
    "core": [
        "drivers", "session_result", "starting_grid", "race_control",
        "weather", "pit", "stints",
        "championship_drivers", "championship_teams",
    ],
    "full": [
        "drivers", "session_result", "starting_grid", "race_control",
        "weather", "pit", "stints",
        "championship_drivers", "championship_teams",
        "laps", "position", "intervals",
    ],
}

# Endpoints do plano ainda não expostos pelo cliente atual (car_data,
# location, team_radio). Registrados aqui para extensão futura.
UNSUPPORTED_ENDPOINTS = ["car_data", "location", "team_radio"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de armazenamento
# ─────────────────────────────────────────────────────────────────────────────

def _raw_path(year: int, endpoint: str, key: object) -> Path:
    path = RAW_DIR / str(year) / endpoint / f"{endpoint}_{key}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _legacy_flat_path(endpoint_prefix: str, key: object) -> Path:
    """Caminho do cache plano legado (pré-reestruturação)."""
    return RAW_DIR / f"{endpoint_prefix}_{key}.csv"


def _write_processed(table: str, year: int, df: pd.DataFrame) -> dict[str, str]:
    out_dir = PROCESSED_DIR / table
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / f"{table}_{year}.parquet"
    csv = out_dir / f"{table}_{year}.csv"
    try:
        df.to_parquet(parquet, index=False, engine="pyarrow")
    except Exception:
        # Colunas 'object' com tipos mistos (ex.: gap_to_leader com floats e
        # "+1 LAP") quebram a conversão para Arrow. Normaliza para string
        # preservando nulos e tenta de novo.
        safe = df.copy()
        for col in safe.columns:
            if safe[col].dtype == object:
                safe[col] = safe[col].map(
                    lambda v: v if (v is None or (isinstance(v, float) and pd.isna(v)))
                    else str(v)
                )
        safe.to_parquet(parquet, index=False, engine="pyarrow")
        df = safe
    df.to_csv(csv, index=False)
    return {"parquet": str(parquet.relative_to(_PROJECT_ROOT)),
            "csv": str(csv.relative_to(_PROJECT_ROOT))}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# Aquisição (online ou importação do cache legado)
# ─────────────────────────────────────────────────────────────────────────────

class _YearStats:
    """Acumula estatísticas de uma sincronização para o manifest."""

    def __init__(self) -> None:
        self.endpoints: dict[str, int] = {}
        self.files_created: list[str] = []
        self.files_updated: list[str] = []
        self.files_reused: int = 0
        self.empty_responses: list[str] = []
        self.errors: list[str] = []

    def record(self, endpoint: str, n_rows: int) -> None:
        self.endpoints[endpoint] = self.endpoints.get(endpoint, 0) + int(n_rows)


def _acquire(
    *,
    year: int,
    endpoint: str,
    key: object,
    fetch_fn: Callable[[], pd.DataFrame],
    stats: _YearStats,
    force_refresh: bool,
    incremental: bool,
    from_local_cache: bool,
    legacy_prefix: str | None = None,
) -> pd.DataFrame:
    """
    Obtém um dataset e garante que ele exista no layout novo
    (raw/<ano>/<endpoint>/). Fontes, em ordem:

        1. arquivo já existente no layout novo (se não for force_refresh);
        2. cache plano legado (importação offline);
        3. API (somente quando não estamos em --from-local-cache).
    """
    target = _raw_path(year, endpoint, key)

    if target.exists() and not force_refresh:
        stats.files_reused += 1
        df = pd.read_csv(target)
        stats.record(endpoint, len(df))
        return df

    legacy = _legacy_flat_path(legacy_prefix or endpoint, key)
    if legacy.exists() and not force_refresh:
        df = pd.read_csv(legacy)
        df.to_csv(target, index=False)
        stats.files_created.append(str(target.relative_to(_PROJECT_ROOT)))
        stats.record(endpoint, len(df))
        return df

    if from_local_cache:
        stats.empty_responses.append(f"{endpoint}:{key} (ausente no cache local)")
        return pd.DataFrame()

    if incremental and target.exists():
        stats.files_reused += 1
        df = pd.read_csv(target)
        stats.record(endpoint, len(df))
        return df

    try:
        df = fetch_fn()
    except Exception as exc:  # segurança operacional: registra e segue
        stats.errors.append(f"{endpoint}:{key} → {exc}")
        return pd.DataFrame()

    if df.empty:
        stats.empty_responses.append(f"{endpoint}:{key}")
        return df

    existed = target.exists()
    df.to_csv(target, index=False)
    rel = str(target.relative_to(_PROJECT_ROOT))
    (stats.files_updated if existed else stats.files_created).append(rel)
    stats.record(endpoint, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sincronização de um ano
# ─────────────────────────────────────────────────────────────────────────────

def sync_year(
    year: int,
    *,
    profile: str = "core",
    force_refresh: bool = False,
    incremental: bool = False,
    from_local_cache: bool = False,
) -> dict:
    """Sincroniza um ano e devolve o manifesto gerado."""
    stats = _YearStats()
    endpoints = PROFILES[profile]

    meetings = _acquire(
        year=year, endpoint="meetings", key=year,
        fetch_fn=lambda: client.fetch_meetings(year),
        stats=stats, force_refresh=force_refresh,
        incremental=incremental, from_local_cache=from_local_cache,
    )

    sessions_frames: list[pd.DataFrame] = []
    session_targets: list[dict] = []

    if not meetings.empty:
        for _, mtg in meetings.iterrows():
            if pd.isna(mtg.get("meeting_key")):
                continue
            mk = int(mtg["meeting_key"])
            sessions = _acquire(
                year=year, endpoint="sessions", key=mk,
                fetch_fn=lambda mk=mk: client.fetch_sessions(mk),
                stats=stats, force_refresh=force_refresh,
                incremental=incremental, from_local_cache=from_local_cache,
                legacy_prefix="sessions_meeting",
            )
            if sessions.empty:
                continue
            sessions_frames.append(sessions)
            for _, sess in sessions.iterrows():
                if str(sess.get("session_name", "")) in SESSION_NAME_TARGETS:
                    session_targets.append({
                        "session_key": int(sess["session_key"]),
                        "session_name": str(sess["session_name"]),
                        "meeting_key": mk,
                        "meeting_name": str(mtg.get("meeting_name", "")),
                        "circuit_short_name": str(mtg.get("circuit_short_name", "")),
                        "location": str(mtg.get("location", "")),
                        "date_start": str(sess.get("date_start", "")),
                    })

    raw_by_endpoint: dict[str, list[pd.DataFrame]] = {e: [] for e in endpoints}
    _legacy_prefixes = {"drivers": "drivers_session"}
    for tgt in session_targets:
        sk = tgt["session_key"]
        for endpoint in endpoints:
            fetch = _SESSION_FETCHERS[endpoint]
            df = _acquire(
                year=year, endpoint=endpoint, key=sk,
                fetch_fn=lambda fetch=fetch, sk=sk: fetch(sk),
                stats=stats, force_refresh=force_refresh,
                incremental=incremental, from_local_cache=from_local_cache,
                legacy_prefix=_legacy_prefixes.get(endpoint),
            )
            if not df.empty:
                df = df.copy()
                for col, val in (
                    ("session_key", sk),
                    ("meeting_key", tgt["meeting_key"]),
                    ("session_name", tgt["session_name"]),
                ):
                    if col not in df.columns:
                        df[col] = val
                raw_by_endpoint[endpoint].append(df)

    processed_files = build_processed_tables(
        year=year,
        meetings=meetings,
        sessions=pd.concat(sessions_frames, ignore_index=True)
        if sessions_frames else pd.DataFrame(),
        session_targets=session_targets,
        raw_by_endpoint=raw_by_endpoint,
    )

    manifest = _build_manifest(
        year=year, profile=profile, stats=stats, meetings=meetings,
        session_targets=session_targets, processed_files=processed_files,
        from_local_cache=from_local_cache,
    )
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFESTS_DIR / f"sync_{year}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Manifesto gravado: %s", manifest_path)
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Tabelas processadas
# ─────────────────────────────────────────────────────────────────────────────

def _acronym_map(drivers_df: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """(session_key, driver_number) → {acronym, full_name, team_name}."""
    out: dict[tuple[int, int], dict] = {}
    if drivers_df.empty:
        return out
    for _, row in drivers_df.iterrows():
        try:
            key = (int(row["session_key"]), int(row["driver_number"]))
        except (TypeError, ValueError, KeyError):
            continue
        out[key] = {
            "driver": str(row.get("name_acronym", "") or ""),
            "full_name": str(row.get("full_name", "") or ""),
            "team_name": str(row.get("team_name", "") or ""),
        }
    return out


def _summarize_race_control_df(df: pd.DataFrame) -> dict[str, int]:
    """Mesma semântica de contagem usada pelo feature_builder histórico."""
    if df.empty:
        return {"sc_count": 0, "vsc_count": 0,
                "red_flag_count": 0, "yellow_flag_count": 0,
                "has_race_control": 0}

    def _txt(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index)

    msg = _txt("message").str.upper()
    flag = _txt("flag").str.upper()
    cat = _txt("category").str.upper()

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


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return (
        df[col].astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes"})
    )


def build_processed_tables(
    *,
    year: int,
    meetings: pd.DataFrame,
    sessions: pd.DataFrame,
    session_targets: list[dict],
    raw_by_endpoint: dict[str, list[pd.DataFrame]],
) -> dict[str, dict[str, str]]:
    """Constrói as tabelas padronizadas de um ano a partir do raw."""
    files: dict[str, dict[str, str]] = {}
    tgt_by_sk = {t["session_key"]: t for t in session_targets}

    def _concat(endpoint: str) -> pd.DataFrame:
        frames = raw_by_endpoint.get(endpoint) or []
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not meetings.empty:
        m = meetings.copy()
        m["season"] = year
        files["meetings"] = _write_processed("meetings", year, m)
    if not sessions.empty:
        s = sessions.copy()
        s["season"] = year
        files["sessions"] = _write_processed("sessions", year, s)

    drivers_raw = _concat("drivers")
    if not drivers_raw.empty:
        d = drivers_raw.copy()
        d["season"] = year
        files["drivers"] = _write_processed("drivers", year, d)
    amap = _acronym_map(drivers_raw)

    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        """Anexa season, identificação da corrida e piloto (sigla/equipe)."""
        out = df.copy()
        out["season"] = year
        out["meeting_name"] = out["session_key"].map(
            lambda sk: tgt_by_sk.get(int(sk), {}).get("meeting_name", ""))
        out["circuit_short_name"] = out["session_key"].map(
            lambda sk: tgt_by_sk.get(int(sk), {}).get("circuit_short_name", ""))
        out["date_start"] = out["session_key"].map(
            lambda sk: tgt_by_sk.get(int(sk), {}).get("date_start", ""))
        out["race_key"] = out["circuit_short_name"].map(canonical_race_key)
        if "driver_number" in out.columns:
            info = [
                amap.get((int(sk), int(dn)) if pd.notna(sk) and pd.notna(dn)
                         else (None, None), {})
                for sk, dn in zip(out["session_key"], out["driver_number"])
            ]
            out["driver"] = [i.get("driver", "") for i in info]
            out["driver_full_name"] = [i.get("full_name", "") for i in info]
            out["team_name"] = [i.get("team_name", "") for i in info]
        return out

    results_raw = _concat("session_result")
    if not results_raw.empty:
        res = _enrich(results_raw)
        res["session_name"] = res["session_key"].map(
            lambda sk: tgt_by_sk.get(int(sk), {}).get("session_name", ""))
        res["position_num"] = pd.to_numeric(res.get("position"), errors="coerce")
        res["dnf"] = _bool_col(res, "dnf")
        res["dns"] = _bool_col(res, "dns")
        res["dsq"] = _bool_col(res, "dsq")
        res["classified"] = res["position_num"].notna() & ~(
            res["dnf"] | res["dns"] | res["dsq"])
        res["number_of_laps"] = pd.to_numeric(
            res.get("number_of_laps"), errors="coerce")
        # Campos de status capturados desde já (decisão complementar 5c),
        # mesmo que o modelo atual ainda não os utilize.
        race = res[res["session_name"] == "Race"].copy()
        sprint = res[res["session_name"] == "Sprint"].copy()
        quali = res[res["session_name"].isin(
            ["Qualifying", "Sprint Qualifying", "Sprint Shootout"])].copy()
        if not race.empty:
            files["race_results"] = _write_processed("race_results", year, race)
        if not sprint.empty:
            files["sprint_results"] = _write_processed("sprint_results", year, sprint)
        if not quali.empty:
            files["qualifying_results"] = _write_processed(
                "qualifying_results", year, quali)

    grid_raw = _concat("starting_grid")
    if not grid_raw.empty:
        files["starting_grid"] = _write_processed(
            "starting_grid", year, _enrich(grid_raw))

    rc_raw = _concat("race_control")
    if not rc_raw.empty:
        rows = []
        for sk, group in rc_raw.groupby("session_key"):
            tgt = tgt_by_sk.get(int(sk), {})
            if tgt.get("session_name") != "Race":
                continue
            summary = _summarize_race_control_df(group)
            summary.update({
                "season": year,
                "session_key": int(sk),
                "meeting_name": tgt.get("meeting_name", ""),
                "circuit_short_name": tgt.get("circuit_short_name", ""),
                "race_key": canonical_race_key(tgt.get("circuit_short_name", "")),
            })
            rows.append(summary)
        if rows:
            files["race_control_summary"] = _write_processed(
                "race_control_summary", year, pd.DataFrame(rows))
        files["race_control"] = _write_processed(
            "race_control", year, _enrich(rc_raw))

    for endpoint in ("weather", "pit", "stints", "laps", "position",
                     "intervals", "championship_drivers", "championship_teams"):
        raw = _concat(endpoint)
        if not raw.empty:
            files[endpoint] = _write_processed(endpoint, year, _enrich(raw))

    return files


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def _build_manifest(
    *,
    year: int,
    profile: str,
    stats: _YearStats,
    meetings: pd.DataFrame,
    session_targets: list[dict],
    processed_files: dict[str, dict[str, str]],
    from_local_cache: bool,
) -> dict:
    races = [t for t in session_targets if t["session_name"] == "Race"]
    race_results_ok = 0
    rr_dir = PROCESSED_DIR / "race_results"
    rr_path = rr_dir / f"race_results_{year}.parquet"
    if rr_path.exists():
        race_results_ok = int(
            pd.read_parquet(rr_path)["session_key"].nunique())

    now = datetime.now(timezone.utc)
    season_over = year < now.year
    complete = season_over and race_results_ok == len(races) and len(races) > 0
    status = "completo" if complete else "parcial"

    return {
        "year": year,
        "executed_at": _utc_now(),
        "mode": "from_local_cache" if from_local_cache else "online",
        "profile": profile,
        "endpoints_consulted": sorted(stats.endpoints.keys()),
        "records_by_endpoint": stats.endpoints,
        "meetings_found": 0 if meetings.empty else int(len(meetings)),
        "sessions_found": len(session_targets),
        "races_found": len(races),
        "races_with_results": race_results_ok,
        "files_created": stats.files_created,
        "files_updated": stats.files_updated,
        "files_reused": stats.files_reused,
        "empty_responses": stats.empty_responses,
        "errors": stats.errors,
        "processed_tables": processed_files,
        "coverage": {
            "races_expected_hint": len(races),
            "races_with_results": race_results_ok,
        },
        "status": status,
        "unsupported_endpoints": UNSUPPORTED_ENDPOINTS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.data_openf1.sync",
        description="Sincronizador central da OpenF1 (única camada com acesso à API).",
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="core")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Rebaixa tudo, ignorando arquivos existentes.")
    parser.add_argument("--incremental", action="store_true",
                        help="Baixa apenas o que ainda não existe em disco.")
    parser.add_argument("--from-local-cache", action="store_true",
                        help="Importa o cache plano legado sem chamar a API.")
    parser.add_argument("--rebuild-context", action="store_true",
                        help="Regenera data/openf1/processed/race_context_*.csv "
                             "a partir do cache (sem rede quando possível).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    if args.from_local_cache:
        os.environ["OPENF1_OFFLINE"] = "1"

    years = list(range(args.start_year, args.end_year + 1))
    for year in years:
        logger.info("── Sincronizando %d (profile=%s, mode=%s) ──",
                    year, args.profile,
                    "offline" if args.from_local_cache else "online")
        manifest = sync_year(
            year,
            profile=args.profile,
            force_refresh=args.force_refresh,
            incremental=args.incremental,
            from_local_cache=args.from_local_cache,
        )
        logger.info(
            "%d: %d corridas encontradas, %d com resultado, status=%s",
            year, manifest["races_found"], manifest["races_with_results"],
            manifest["status"],
        )

    if args.rebuild_context:
        from src.data_openf1.feature_builder import build_race_context
        ctx = build_race_context(years=years, allow_partial=True)
        logger.info("race_context regenerado: %d corridas", len(ctx))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
