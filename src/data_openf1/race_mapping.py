"""
src/data_openf1/race_mapping.py
================================
Normalizacao e equivalencia de nomes entre:
    - CSV historico do projeto: Bahrain, Saudi Arabia, Great Britain, ...
    - OpenF1: Sakhir, Jeddah, Silverstone, ...

O objetivo e permitir match estavel entre resultados historicos e contexto
OpenF1 sem depender de contains/fragments de string.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(value: object) -> str:
    """Normaliza texto para comparacao: sem acentos, minusculo, espacos simples."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Chave canonica usada no ctx_df.
# Para as corridas do OpenF1, a propria cidade/circuito geralmente ja e a chave.
_ALIAS_TO_KEY = {
    # Historico -> OpenF1/circuit_short_name
    "bahrain": "sakhir",
    "sakhir": "sakhir",
    "saudi arabia": "jeddah",
    "jeddah": "jeddah",
    "australia": "melbourne",
    "melbourne": "melbourne",
    "azerbaijan": "baku",
    "baku": "baku",
    "miami": "miami",
    "emilia romagna": "imola",
    "emilia romagna gp": "imola",
    "emilia romagna grand prix": "imola",
    "emiliaromagna": "imola",
    "imola": "imola",
    "monaco": "monte carlo",
    "monte carlo": "monte carlo",
    "spain": "catalunya",
    "spanish": "catalunya",
    "catalunya": "catalunya",
    "canada": "montreal",
    "montreal": "montreal",
    "austria": "spielberg",
    "spielberg": "spielberg",
    "great britain": "silverstone",
    "britain": "silverstone",
    "united kingdom": "silverstone",
    "silverstone": "silverstone",
    "hungary": "hungaroring",
    "hungaroring": "hungaroring",
    "belgium": "spa francorchamps",
    "spa": "spa francorchamps",
    "spa francorchamps": "spa francorchamps",
    "netherlands": "zandvoort",
    "dutch": "zandvoort",
    "zandvoort": "zandvoort",
    "italy": "monza",
    "italian": "monza",
    "monza": "monza",
    "singapore": "singapore",
    "japan": "suzuka",
    "suzuka": "suzuka",
    "qatar": "lusail",
    "lusail": "lusail",
    "united states": "austin",
    "usa": "austin",
    "us": "austin",
    "austin": "austin",
    "mexico": "mexico city",
    "mexico city": "mexico city",
    "brazil": "interlagos",
    "sao paulo": "interlagos",
    "interlagos": "interlagos",
    "las vegas": "las vegas",
    "abu dhabi": "yas marina circuit",
    "yas marina": "yas marina circuit",
    "yas marina circuit": "yas marina circuit",
    "china": "shanghai",
    "chinese": "shanghai",
    "shanghai": "shanghai",
    # Corridas antigas possiveis nos CSVs historicos, sem OpenF1 2023+
    "france": "le castellet",
    "french": "le castellet",
    "le castellet": "le castellet",
    "russia": "sochi",
    "sochi": "sochi",
    "portugal": "portimao",
    "portimao": "portimao",
    "turkey": "istanbul",
    "istanbul": "istanbul",
    "eifel": "nurburgring",
    "nurburgring": "nurburgring",
    "tuscany": "mugello",
    "mugello": "mugello",
    "styrian": "spielberg",
    "70th anniversary": "silverstone",
}


def canonical_race_key(value: object) -> str:
    """Retorna a chave canonica de uma corrida/circuito."""
    norm = normalize_text(value)
    return _ALIAS_TO_KEY.get(norm, norm)


def add_race_key_from_columns(row: object) -> str:
    """
    Calcula race_key a partir de uma linha do DataFrame OpenF1.
    Prefere circuit_short_name; usa meeting_name como fallback.
    """
    for attr in ("circuit_short_name", "race", "meeting_name", "location"):
        try:
            value = row.get(attr, "")  # pandas Series
        except AttributeError:
            value = ""
        key = canonical_race_key(value)
        if key:
            return key
    return ""
