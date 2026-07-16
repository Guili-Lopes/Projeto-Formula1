"""Pipelines — validação das configurações padrão (splits históricos)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data.repository import SOURCE_POLICIES

CONFIGS = {
    "pipeline_mallows_plackett_luce": {
        "train": [2019, 2020, 2021, 2022],
        "validation": [2023],
        "test": [2024],
    },
    "pipeline_score_rules": {
        "train": [2019, 2020, 2021, 2022],
        "validation": [2023],
        "test": [2024],
    },
    "pipeline_openf1": {
        "train": [2019, 2020, 2021, 2022, 2023],
        "validation": [2024],
        "test": [2025],
    },
}


@pytest.mark.parametrize("pipeline,splits", CONFIGS.items())
def test_default_config_preserves_historical_split(
        project_root: Path, pipeline: str, splits: dict):
    path = project_root / "src" / pipeline / "configs" / "default.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert cfg["pipeline"] == pipeline
    assert cfg["splits"] == splits, "split histórico alterado (decisão nº 3)"
    assert cfg["seed"] == 42
    assert cfg["data"]["top_k"] == 10
    assert cfg["data"]["source_policy"] in SOURCE_POLICIES
    assert cfg["model"]["n_clusters"] == 2

    if pipeline != "pipeline_mallows_plackett_luce":
        mc = cfg["monte_carlo"]
        assert mc["n_simulations"] == 10000
        assert mc["n_positions"] == 20
        assert mc.get("seed") is None  # comportamento histórico preservado

    if pipeline == "pipeline_openf1":
        assert cfg["context"]["first_year"] == 2023


@pytest.mark.parametrize("pipeline", CONFIGS)
def test_pipeline_has_readme(project_root: Path, pipeline: str):
    readme = project_root / "src" / pipeline / "README.md"
    text = readme.read_text(encoding="utf-8")
    for section in ("Objetivo", "Como executar", "Configuração",
                    "Dados", "Saídas", "Notebook"):
        assert section in text, f"{pipeline}/README.md sem seção '{section}'"
