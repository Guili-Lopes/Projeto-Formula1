"""
Configuração global da suíte de testes (Etapa 8 da reestruturação).

Regras (plano, seção 9.2):
    - testes não modificam data/ real — tudo em diretórios temporários;
    - nenhum teste chama a API OpenF1 (OPENF1_OFFLINE=1 forçado);
    - seeds fixas para reprodutibilidade.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers.synthetic import make_legacy_seasons  # noqa: E402


@pytest.fixture(autouse=True)
def _offline_guard(monkeypatch):
    """Bloqueia qualquer chamada à API OpenF1 em todos os testes."""
    monkeypatch.setenv("OPENF1_OFFLINE", "1")


@pytest.fixture()
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> Path:
    """data/ sintético com quatro temporadas legadas (3001–3004)."""
    data_dir = tmp_path / "data"
    make_legacy_seasons(
        data_dir,
        years=[3001, 3002, 3003, 3004],
        races_per_year=3,
        n_drivers=8,
        seed=11,
        dnf_per_race=2,
    )
    return data_dir


@pytest.fixture()
def tmp_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir()
    return root
