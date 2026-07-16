"""Unit — camada de dados: data_pipeline e repository (regra de fontes)."""

from __future__ import annotations

import pytest

from src.data import repository
from src.data.data_pipeline import DRIVER_ABBREV, load_seasons
from tests.helpers.synthetic import driver_abbrevs


def test_load_seasons_synthetic_structure(synthetic_data_dir):
    records = load_seasons(str(synthetic_data_dir), [3001], top_k=5)
    assert len(records) == 3  # 3 corridas por temporada sintética
    abbrevs = set(driver_abbrevs(8))
    for rec in records:
        assert rec.season == 3001
        # top_k corta os classificados (o 6º sai do ranking); DNFs vão à cauda
        assert rec.n_classified == 5
        assert rec.n_dnf == 2
        assert len(rec.ranking) == rec.n_classified + rec.n_dnf
        assert set(rec.ranking) <= abbrevs
        assert len(set(rec.ranking)) == len(rec.ranking)


def test_load_seasons_dnf_goes_to_tail(synthetic_data_dir):
    # posição não numérica ('NC') nunca aparece entre os classificados
    records = load_seasons(str(synthetic_data_dir), [3002], top_k=6)
    for rec in records:
        assert rec.n_classified == 6
        assert rec.n_dnf == 2


def test_repository_legacy_only_matches_load_seasons(synthetic_data_dir):
    direct = load_seasons(str(synthetic_data_dir), [3001, 3002], top_k=5)
    via_repo, provenance = repository.load_season_records(
        [3001, 3002], top_k=5,
        source_policy="legacy_only", data_dir=synthetic_data_dir)
    assert provenance == {3001: "legacy", 3002: "legacy"}
    assert [(r.season, r.race, tuple(r.ranking)) for r in direct] == \
           [(r.season, r.race, tuple(r.ranking)) for r in via_repo]


def test_repository_prefer_openf1_falls_back_with_warning(
        synthetic_data_dir, caplog):
    # 3003 >= OPENF1_FIRST_YEAR? Não — anos sintéticos são >2023, então a
    # política tenta OpenF1, não encontra processado e cai para o legado.
    assert 3003 >= repository.OPENF1_FIRST_YEAR
    with caplog.at_level("WARNING", logger="data.repository"):
        records, provenance = repository.load_season_records(
            [3003], top_k=5,
            source_policy="prefer_openf1", data_dir=synthetic_data_dir)
    assert provenance == {3003: "legacy"}
    assert len(records) == 3
    assert any("Sem dados OpenF1" in m for m in caplog.messages)


def test_repository_strict_openf1_raises_without_processed(synthetic_data_dir):
    with pytest.raises(FileNotFoundError):
        repository.load_season_records(
            [3004], top_k=5,
            source_policy="strict_openf1", data_dir=synthetic_data_dir)


def test_repository_rejects_unknown_policy(synthetic_data_dir):
    with pytest.raises(ValueError):
        repository.load_season_records(
            [3001], source_policy="qualquer", data_dir=synthetic_data_dir)


def test_driver_abbrev_values_are_valid_siglas():
    # nomes-alias podem apontar para a mesma sigla; toda sigla tem 3 letras
    assert all(len(v) == 3 and v.isupper() for v in DRIVER_ABBREV.values())
