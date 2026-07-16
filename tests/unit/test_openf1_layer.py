"""Unit — camada OpenF1: race_mapping, cliente offline, sync e repositório."""

from __future__ import annotations

import json

import pandas as pd

from src.data import repository
from src.data_openf1 import client, sync
from src.data_openf1.race_mapping import canonical_race_key
from tests.helpers.synthetic import make_openf1_flat_cache


def test_canonical_race_key_unifies_aliases():
    assert canonical_race_key("Bahrain") == canonical_race_key("Sakhir")
    assert canonical_race_key("Great Britain") == canonical_race_key("Silverstone")
    assert canonical_race_key("  MONACO ") == canonical_race_key("Monaco")


def test_client_blocks_network_when_offline(monkeypatch):
    monkeypatch.setenv("OPENF1_OFFLINE", "1")
    df = client.fetch_meetings(2099)
    assert df.empty
    assert df.attrs.get("openf1_offline_blocked") is True


def _patched_sync_dirs(monkeypatch, tmp_path):
    raw = tmp_path / "openf1" / "raw"
    processed = tmp_path / "openf1" / "processed"
    manifests = tmp_path / "openf1" / "manifests"
    monkeypatch.setattr(sync, "RAW_DIR", raw)
    monkeypatch.setattr(sync, "PROCESSED_DIR", processed)
    monkeypatch.setattr(sync, "MANIFESTS_DIR", manifests)
    return raw, processed, manifests


def test_sync_from_local_cache_builds_processed_and_manifest(
        monkeypatch, tmp_path):
    raw, processed, manifests = _patched_sync_dirs(monkeypatch, tmp_path)
    info = make_openf1_flat_cache(raw)
    year = info["year"]

    manifest = sync.sync_year(year, profile="core", from_local_cache=True)

    # layout novo por ano criado a partir do cache plano
    assert (raw / str(year) / "meetings" / f"meetings_{year}.csv").exists()
    assert (raw / str(year) / "session_result" /
            f"session_result_{info['session_key']}.csv").exists()

    # tabela processada com siglas, status e tipos mistos tratados
    rr = pd.read_parquet(
        processed / "race_results" / f"race_results_{year}.parquet")
    assert rr["session_key"].nunique() == 1
    assert set(rr["driver"]) == set(info["abbrevs"])
    assert rr["classified"].sum() == info["n_classified"]
    assert rr["dnf"].sum() == 1 and rr["dns"].sum() == 1
    assert (rr["team_name"] != "").all()

    # resumo de race control com a semântica de contagem do projeto
    rc = pd.read_parquet(
        processed / "race_control_summary" /
        f"race_control_summary_{year}.parquet")
    row = rc.iloc[0]
    assert row["sc_count"] == 1        # 'IN THIS LAP' não conta como deploy
    assert row["yellow_flag_count"] == 1
    assert row["vsc_count"] == 0

    # manifest com campos do plano
    saved = json.loads(
        (manifests / f"sync_{year}.json").read_text(encoding="utf-8"))
    for key in ("year", "mode", "profile", "records_by_endpoint",
                "races_found", "races_with_results", "files_created",
                "empty_responses", "errors", "status", "processed_tables"):
        assert key in saved
    assert saved["mode"] == "from_local_cache"
    assert saved["races_found"] == 1
    assert saved["races_with_results"] == 1
    assert manifest["races_with_results"] == 1


def test_repository_reads_openf1_records_with_legacy_semantics(
        monkeypatch, tmp_path, synthetic_data_dir):
    raw, processed, _ = _patched_sync_dirs(monkeypatch, tmp_path)
    info = make_openf1_flat_cache(raw)
    year = info["year"]
    sync.sync_year(year, profile="core", from_local_cache=True)

    monkeypatch.setattr(repository, "PROCESSED_DIR", processed)
    repository._label_cache.clear()

    records = repository._openf1_records_for_year(
        year, top_k=3, data_dir=synthetic_data_dir)
    assert len(records) == 1
    rec = records[0]
    # rótulo sem 'Grand Prix' quando não há equivalente no histórico
    assert rec.race == "Testville"
    # classificados = posição numérica, na ordem oficial, cortados no top_k
    assert rec.ranking[:3] == info["abbrevs"][:3]
    assert rec.n_classified == 3
    # cauda = sem posição numérica, DNF (mais voltas) antes do DNS (0 voltas)
    tail = rec.ranking[rec.n_classified:]
    assert tail == [info["abbrevs"][info["n_classified"]],
                    info["abbrevs"][info["n_classified"] + 1]]


def test_sync_profiles_are_consistent():
    assert set(sync.PROFILES["core"]) <= set(sync.PROFILES["full"])
    for endpoint in sync.PROFILES["full"]:
        assert endpoint in sync._SESSION_FETCHERS
