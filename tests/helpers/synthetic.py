"""Geradores de dados sintéticos usados pela suíte de testes."""

from __future__ import annotations

import random
from pathlib import Path

from src.data.data_pipeline import DRIVER_ABBREV

TRACKS = ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Miami", "Monaco"]


def driver_names(n: int) -> list[str]:
    """Nomes reais (presentes em DRIVER_ABBREV) para o loader mapear siglas."""
    return list(DRIVER_ABBREV.keys())[:n]


def driver_abbrevs(n: int) -> list[str]:
    return [DRIVER_ABBREV[name] for name in driver_names(n)]


def make_legacy_seasons(
    data_dir: Path,
    *,
    years: list[int],
    races_per_year: int = 3,
    n_drivers: int = 8,
    seed: int = 11,
    dnf_per_race: int = 2,
) -> None:
    """
    Escreve Season<ano>/<ano>raceresults.csv no formato do dataset histórico:
    colunas Position, Driver, Track; DNFs com posição não numérica ('NC').
    """
    rng = random.Random(seed)
    names = driver_names(n_drivers)
    for year in years:
        season_dir = data_dir / f"Season{year}"
        season_dir.mkdir(parents=True, exist_ok=True)
        lines = ["Position,Driver,Track"]
        for race in TRACKS[:races_per_year]:
            order = names[:]
            rng.shuffle(order)
            n_classified = n_drivers - dnf_per_race
            for pos, name in enumerate(order[:n_classified], start=1):
                lines.append(f"{pos},{name},{race}")
            for name in order[n_classified:]:
                lines.append(f"NC,{name},{race}")
        csv_path = season_dir / f"{year}raceresults.csv"
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_openf1_flat_cache(
    raw_dir: Path,
    *,
    year: int = 3050,
    n_drivers: int = 6,
) -> dict:
    """
    Cache plano mínimo da OpenF1 (formato legado) com um meeting e uma
    corrida, para exercitar o sincronizador em modo --from-local-cache.
    Inclui gap_to_leader com tipos mistos (float e '+1 LAP') de propósito.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    meeting_key, session_key = 9001, 9101
    abbrevs = driver_abbrevs(n_drivers)

    (raw_dir / f"meetings_{year}.csv").write_text(
        "meeting_key,meeting_name,circuit_short_name,location,year\n"
        f"{meeting_key},Testville Grand Prix,Testville,Testville,{year}\n",
        encoding="utf-8",
    )
    (raw_dir / f"sessions_meeting_{meeting_key}.csv").write_text(
        "session_key,meeting_key,session_name,date_start,year\n"
        f"{session_key},{meeting_key},Race,{year}-03-01T15:00:00+00:00,{year}\n",
        encoding="utf-8",
    )
    drv_lines = ["driver_number,name_acronym,full_name,team_name,"
                 "session_key,meeting_key"]
    for i, ab in enumerate(abbrevs, start=1):
        drv_lines.append(f"{i},{ab},Piloto {ab},Equipe {ab},"
                         f"{session_key},{meeting_key}")
    (raw_dir / f"drivers_session_{session_key}.csv").write_text(
        "\n".join(drv_lines) + "\n", encoding="utf-8")

    res_lines = ["position,driver_number,number_of_laps,points,dnf,dns,dsq,"
                 "gap_to_leader,session_key,meeting_key"]
    n_classified = n_drivers - 2
    for pos in range(1, n_classified + 1):
        gap = "0.0" if pos == 1 else (f"{pos * 3.5}" if pos < n_classified
                                      else "+1 LAP")
        res_lines.append(f"{pos},{pos},50,{10 - pos},False,False,False,"
                         f"{gap},{session_key},{meeting_key}")
    # dois não classificados: um DNF (mais voltas) e um DNS
    res_lines.append(f",{n_classified + 1},31,0,True,False,False,,"
                     f"{session_key},{meeting_key}")
    res_lines.append(f",{n_classified + 2},0,0,False,True,False,,"
                     f"{session_key},{meeting_key}")
    (raw_dir / f"session_result_{session_key}.csv").write_text(
        "\n".join(res_lines) + "\n", encoding="utf-8")

    grid_lines = ["position,driver_number,session_key,meeting_key"]
    for pos, i in enumerate(range(1, n_drivers + 1), start=1):
        grid_lines.append(f"{pos},{i},{session_key},{meeting_key}")
    (raw_dir / f"starting_grid_{session_key}.csv").write_text(
        "\n".join(grid_lines) + "\n", encoding="utf-8")

    (raw_dir / f"race_control_{session_key}.csv").write_text(
        "category,flag,message,session_key,meeting_key\n"
        f"SafetyCar,,SAFETY CAR DEPLOYED,{session_key},{meeting_key}\n"
        f"Flag,YELLOW,YELLOW IN SECTOR 2,{session_key},{meeting_key}\n"
        f"Flag,CLEAR,SAFETY CAR IN THIS LAP,{session_key},{meeting_key}\n",
        encoding="utf-8",
    )

    return {
        "year": year,
        "meeting_key": meeting_key,
        "session_key": session_key,
        "abbrevs": abbrevs,
        "n_classified": n_classified,
    }
