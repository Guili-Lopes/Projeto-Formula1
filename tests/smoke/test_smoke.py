"""Smoke — imports dos módulos principais e CLIs respondendo a --help."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

MODULES = [
    "src.data.data_pipeline",
    "src.data.repository",
    "src.data.validate_datasets",
    "src.data_openf1.client",
    "src.data_openf1.sync",
    "src.data_openf1.race_mapping",
    "src.data_openf1.feature_builder",
    "src.engine.engine_trainer",
    "src.engine.engine_predictor",
    "src.evaluation.evaluation_metrics",
    "src.evaluation.additional_metrics",
    "src.experiments.run_store",
    "src.experiments.pipeline_config",
    "src.models.models_mallows",
    "src.models.models_plackett_luce",
    "src.models.models_weights",
    "src.pipeline_mallows_plackett_luce.run_experiment",
    "src.pipeline_score_rules.run_pipeline_score_rules",
    "src.pipeline_score_rules.monte_carlo",
    "src.pipeline_score_rules.scoring_rules",
    "src.pipeline_openf1.run_pipeline_openf1",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


CLIS = [
    "src.data_openf1.sync",
    "src.data.validate_datasets",
    "src.pipeline_mallows_plackett_luce.run_experiment",
    "src.pipeline_score_rules.run_pipeline_score_rules",
    "src.pipeline_openf1.run_pipeline_openf1",
]


@pytest.mark.parametrize("cli", CLIS)
def test_cli_help(cli, project_root):
    proc = subprocess.run(
        [sys.executable, "-m", cli, "--help"],
        capture_output=True, text=True, cwd=project_root, timeout=90,
        env={"OPENF1_OFFLINE": "1", "PATH": "/usr/bin:/bin",
             "HOME": "/root", "MPLBACKEND": "Agg"},
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "usage" in proc.stdout.lower()
