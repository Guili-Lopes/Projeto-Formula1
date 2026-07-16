"""Regression — compara execuções com os baselines capturados na migração.

Os baselines (tests/regression/baselines/*.json) guardam, por pipeline:
  - legacy_only: números da execução original pré-reestruturação;
  - prefer_openf1: números oficiais pós-migração.

Métricas determinísticas são comparadas com tolerância estrita; RPS usa
tolerância de ruído Monte Carlo (seed histórica nula) — regra 9 do plano.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

BASELINES_DIR = Path(__file__).parent / "baselines"
DET_TOL = 1e-6
RPS_TOL = 0.006

PIPELINES = ["pipeline_mallows_plackett_luce",
             "pipeline_score_rules",
             "pipeline_openf1"]


def _load_baseline(pipeline: str) -> dict:
    return json.loads(
        (BASELINES_DIR / f"{pipeline}.json").read_text(encoding="utf-8"))


def _latest_run(project_root: Path, pipeline: str):
    pointer = project_root / "artifacts" / pipeline / "latest_run.json"
    if not pointer.exists():
        pytest.skip(f"sem execuções em artifacts/{pipeline} — rode o pipeline")
    info = json.loads(pointer.read_text(encoding="utf-8"))
    run_dir = project_root / info["run_dir"]
    summary = json.loads(
        (run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    return summary, config


def _compare(summary: dict, expected: dict) -> list[str]:
    problems = []
    for phase in ("validation", "test"):
        exp = expected[phase]
        got = summary[phase]
        if got["n_races"] != exp["n_races"]:
            problems.append(
                f"{phase}.n_races: {got['n_races']} ≠ {exp['n_races']}")
        for key, tol in (("mean_top3", DET_TOL), ("mean_top5", DET_TOL),
                         ("mean_kendall", DET_TOL),
                         ("mean_rps_model", RPS_TOL),
                         ("mean_rps_baseline", RPS_TOL),
                         ("mean_gain", RPS_TOL)):
            if key not in exp:
                continue
            if abs(got[key] - exp[key]) > tol:
                problems.append(
                    f"{phase}.{key}: {got[key]:.4f} vs {exp[key]:.4f} "
                    f"(tol {tol})")
    return problems


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_latest_run_matches_mode_baseline(project_root, pipeline):
    """A última execução em artifacts/ bate com o baseline do seu modo."""
    summary, config = _latest_run(project_root, pipeline)
    mode = config["data"]["source_policy"]
    baseline = _load_baseline(pipeline)
    if mode not in baseline["modes"]:
        pytest.skip(f"sem baseline para o modo {mode}")
    problems = _compare(summary, baseline["modes"][mode])
    assert problems == [], "; ".join(problems)


@pytest.mark.slow
def test_pipeline1_legacy_only_reproduces_original_baseline(
        project_root, tmp_path):
    """Execução completa do P1 em legacy_only reproduz o baseline exato."""
    from src.pipeline_mallows_plackett_luce.run_experiment import main

    default = (project_root / "src" / "pipeline_mallows_plackett_luce" /
               "configs" / "default.yaml")
    cfg = yaml.safe_load(default.read_text(encoding="utf-8"))
    cfg["data"]["source_policy"] = "legacy_only"
    cfg["artifacts"] = {"root": str(tmp_path / "artifacts")}
    cfg_path = tmp_path / "p1_legacy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    main(["--config", str(cfg_path)])

    pointer = json.loads(
        (tmp_path / "artifacts" / "pipeline_mallows_plackett_luce" /
         "latest_run.json").read_text(encoding="utf-8"))
    summary = pointer["summary"]
    expected = _load_baseline(
        "pipeline_mallows_plackett_luce")["modes"]["legacy_only"]
    problems = _compare(summary, expected)
    assert problems == [], "; ".join(problems)
