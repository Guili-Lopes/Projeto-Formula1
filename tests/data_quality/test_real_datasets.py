"""Data quality — verificações sobre os dados reais versionados (somente leitura)."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from src.data import repository
from src.data.validate_datasets import MANIFESTS_DIR, validate

pytestmark = pytest.mark.data_quality


@pytest.fixture(scope="module")
def result():
    return validate()


def test_transition_seasons_are_equivalent(result):
    c = result["comparison"]
    assert c["races_compared"] == 61
    # Núcleo da validação da Etapa 3: mesma ordem oficial de chegada
    assert c["classified_identical"] == c["races_compared"]
    # Única exceção conhecida de conjunto de DNFs: DNS omitido pela OpenF1
    # (Singapura 2023). Ordem da cauda pode variar (documentado no relatório).
    assert c["dnf_set_identical"] >= c["races_compared"] - 1


def test_only_known_quality_issues(result):
    allowed = re.compile(r"menos de 20 pilotos")
    unexpected = [i for i in result["issues"] if not allowed.search(i)]
    assert unexpected == [], f"novas ocorrências de qualidade: {unexpected}"


def test_openf1_season_coverage(result):
    years = result["years"]
    assert years[2023]["openf1_races"] == 22   # Ímola 2023 cancelada
    assert years[2024]["openf1_races"] == 24
    assert years[2025]["openf1_races"] == 24   # temporada completa na OpenF1
    assert years[2025]["legacy_races"] == 15   # CSV legado parcial (transição)


def test_manifests_exist_and_are_consistent():
    for year in (2023, 2024, 2025):
        payload = json.loads(
            (MANIFESTS_DIR / f"sync_{year}.json").read_text(encoding="utf-8"))
        assert payload["year"] == year
        assert payload["races_with_results"] <= payload["races_found"]
        assert payload["status"] in {"completo", "parcial"}


def test_processed_race_results_positions_are_valid():
    for year in (2023, 2024, 2025):
        df = repository.load_openf1_race_results(year)
        assert not df.empty
        for _, g in df.groupby("session_key"):
            pos = g["position_num"].dropna()
            assert not pos.duplicated().any()
            assert pos.min() == 1
            assert (g.loc[g["dns"], "position_num"].isna()).all()


def test_legacy_label_map_covers_openf1_circuits():
    repository._label_cache.clear()
    mapping = repository._legacy_label_map(repository.DEFAULT_DATA_DIR)
    from src.data_openf1.race_mapping import canonical_race_key
    assert mapping.get(canonical_race_key("Sakhir")) == "Bahrain"
    for year in (2023, 2024, 2025):
        df = repository.load_openf1_race_results(year)
        keys = {canonical_race_key(c) for c in df["circuit_short_name"]}
        missing = {k for k in keys if k not in mapping}
        assert missing == set(), f"{year}: circuitos sem rótulo histórico: {missing}"
