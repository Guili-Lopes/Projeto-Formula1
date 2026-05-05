"""
src/data/data_pipeline.py
=========================
Responsabilidade única: carregar os CSVs brutos e transformá-los em
rankings parciais ordenados por corrida.

Compartilhado entre Pipeline 1 e Pipeline 2.
"""

import os
import pandas as pd
from dataclasses import dataclass


DRIVER_ABBREV: dict[str, str] = {
    'Max Verstappen':     'VER', 'Sergio Perez':       'PER',
    'Lewis Hamilton':     'HAM', 'George Russell':     'RUS',
    'Charles Leclerc':    'LEC', 'Carlos Sainz':       'SAI',
    'Lando Norris':       'NOR', 'Oscar Piastri':      'PIA',
    'Fernando Alonso':    'ALO', 'Lance Stroll':       'STR',
    'Esteban Ocon':       'OCO', 'Pierre Gasly':       'GAS',
    'Valtteri Bottas':    'BOT', 'Guanyu Zhou':        'ZHO',
    'Yuki Tsunoda':       'TSU', 'Daniel Ricciardo':   'RIC',
    'Liam Lawson':        'LAW', 'Kevin Magnussen':    'MAG',
    'Nico Hulkenberg':    'HUL', 'Alexander Albon':    'ALB',
    'Logan Sargeant':     'SAR', 'Franco Colapinto':   'COL',
    'Oliver Bearman':     'BEA', 'Jack Doohan':        'DOO',
    'Kimi Antonelli':     'ANT', 'Gabriel Bortoleto':  'BOR',
    'Isack Hadjar':       'HAD', 'Sebastian Vettel':   'VET',
    'Kimi Raikkönen':     'RAI', 'Kimi Räikkönen':     'RAI',
    'Romain Grosjean':    'GRO', 'Antonio Giovinazzi': 'GIO',
    'Daniil Kvyat':       'KVY', 'Nicholas Latifi':    'LAT',
    'Mick Schumacher':    'MSC', 'Nikita Mazepin':     'MAZ',
    'Robert Kubica':      'KUB', 'Nyck de Vries':      'DEV',
    'Nyck De Vries':      'DEV', 'Zhou Guanyu':        'ZHO',
    'Theo Pourchaire':    'POU', 'Jack Aitken':        'AIT',
    'Pietro Fittipaldi':  'FIT',
}


@dataclass
class RaceRecord:
    """Representa uma corrida processada com seu ranking."""
    season:       int
    race:         str
    ranking:      list[str]
    n_classified: int
    n_dnf:        int


def _find_race_file(data_dir: str, season: int) -> str | None:
    season_dir = os.path.join(data_dir, f'Season{season}')
    if not os.path.isdir(season_dir):
        return None
    for fname in sorted(os.listdir(season_dir)):
        lower = fname.lower()
        if lower.endswith('raceresults.csv') and 'sprint' not in lower:
            return os.path.join(season_dir, fname)
    for fname in sorted(os.listdir(season_dir)):
        lower = fname.lower()
        if 'raceresults' in lower and 'sprint' not in lower:
            return os.path.join(season_dir, fname)
    return None


def load_seasons(
    data_dir: str,
    seasons:  list[int],
    top_k:    int = 10,
) -> list[RaceRecord]:
    """
    Carrega todas as temporadas e retorna lista de RaceRecord
    em ordem cronológica.
    """
    records: list[RaceRecord] = []

    for season in sorted(seasons):
        filepath = _find_race_file(data_dir, season)
        if filepath is None:
            print(f"  [AVISO] Temporada {season}: arquivo não encontrado.")
            continue

        df = pd.read_csv(filepath)
        df.columns   = [c.strip() for c in df.columns]
        df['pos_norm'] = df['Position'].apply(
            lambda x: int(str(x).strip()) if str(x).strip().isdigit() else 'DNF'
        )
        df['dnf']    = df['pos_norm'] == 'DNF'
        df['abbrev'] = df['Driver'].map(DRIVER_ABBREV)

        unmapped = df[df['abbrev'].isna()]['Driver'].unique()
        if len(unmapped):
            print(f"  [AVISO] {season}: sem abreviação → {list(unmapped)}")

        races_ordered = list(dict.fromkeys(df['Track'].tolist()))

        for race in races_ordered:
            df_race     = df[df['Track'] == race].copy()
            classified  = (df_race[~df_race['dnf']]
                           .sort_values('pos_norm')
                           .dropna(subset=['abbrev']))
            dnf_drivers = df_race[df_race['dnf']].dropna(subset=['abbrev'])

            top_drivers = classified['abbrev'].tolist()[:top_k]
            dnf_list    = dnf_drivers['abbrev'].tolist()
            ranking     = top_drivers + dnf_list

            if len(ranking) >= 2:
                records.append(RaceRecord(
                    season       = season,
                    race         = race,
                    ranking      = ranking,
                    n_classified = len(top_drivers),
                    n_dnf        = len(dnf_list),
                ))

    return records


def get_all_drivers(records: list[RaceRecord]) -> list[str]:
    """Retorna lista ordenada de todos os pilotos únicos."""
    drivers = set()
    for r in records:
        drivers.update(r.ranking)
    return sorted(drivers)
