"""E2E — os três pipelines completos sobre dados sintéticos reduzidos.

Cada teste executa o main() real do pipeline com uma configuração mínima
(temporadas sintéticas 3001–3004, Monte Carlo pequeno) escrevendo em um
artifacts/ temporário — sem tocar em data/ real nem chamar a API.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from tests.helpers.synthetic import driver_abbrevs

pytestmark = pytest.mark.e2e

VIZ_DRIVERS = driver_abbrevs(5)


def _write_config(tmp_path, data_dir, artifacts_root, *, pipeline,
                  extra: dict) -> str:
    cfg = {
        "pipeline": pipeline,
        "seed": 7,
        "data": {"source_policy": "legacy_only", "top_k": 5,
                 "dir": str(data_dir)},
        "splits": {"train": [3001, 3002], "validation": [3003],
                   "test": [3004]},
        "model": {"n_clusters": 2, "n_iter": 8, "alpha": 0.5},
        "artifacts": {"root": str(artifacts_root)},
    }
    cfg.update(extra)
    path = tmp_path / f"{pipeline}_e2e.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


def _check_run(artifacts_root, pipeline, *, expect_tables):
    pointer = json.loads(
        (artifacts_root / pipeline / "latest_run.json").read_text("utf-8"))
    run_dir = artifacts_root / pipeline / pointer["run_id"]
    summary = json.loads(
        (run_dir / "metrics_summary.json").read_text("utf-8"))
    for phase in ("validation", "test"):
        assert summary[phase]["n_races"] == 3
        assert 0.0 <= summary[phase]["mean_top3"] <= 1.0
    for stem in expect_tables:
        assert (run_dir / f"{stem}.parquet").exists(), stem
    rm = pd.read_parquet(run_dir / "race_metrics.parquet")
    assert len(rm) == 6 and set(rm["phase"]) == {"validation", "test"}
    assert (run_dir / "run.log").stat().st_size > 0
    assert (run_dir / "manifest.json").exists()
    assert list(run_dir.glob("plots/*.png")), "nenhuma figura gerada"
    assert not (run_dir / "error.json").exists()
    return run_dir, summary


def test_pipeline1_e2e(tmp_path, synthetic_data_dir, tmp_artifacts):
    from src.pipeline_mallows_plackett_luce.run_experiment import main
    cfg = _write_config(
        tmp_path, synthetic_data_dir, tmp_artifacts,
        pipeline="pipeline_mallows_plackett_luce",
        extra={"visualization": {"viz1_drivers": VIZ_DRIVERS,
                                 "skill_ranking_top_n": 5}})
    main(["--config", cfg])
    run_dir, _ = _check_run(
        tmp_artifacts, "pipeline_mallows_plackett_luce",
        expect_tables=["race_metrics", "predictions",
                       "skill_history", "regulatory_weights"])
    assert (run_dir / "nb_data.pkl").exists()


def test_pipeline2_e2e(tmp_path, synthetic_data_dir, tmp_artifacts):
    from src.pipeline_score_rules.run_pipeline_score_rules import main
    cfg = _write_config(
        tmp_path, synthetic_data_dir, tmp_artifacts,
        pipeline="pipeline_score_rules",
        extra={"monte_carlo": {"n_simulations": 60, "n_positions": 6,
                               "seed": 5},
               "visualization": {"top5_drivers": VIZ_DRIVERS}})
    main(["--config", cfg])
    run_dir, summary = _check_run(
        tmp_artifacts, "pipeline_score_rules",
        expect_tables=["race_metrics", "rps_metrics",
                       "predictions", "position_probabilities"])
    assert summary["validation"]["mean_rps_baseline"] > 0
    pp = pd.read_parquet(run_dir / "position_probabilities.parquet")
    assert set(pp["position"]) == set(range(1, 7))
    assert (run_dir / "nb_data_p2.pkl").exists()


def test_pipeline3_e2e_without_context(tmp_path, synthetic_data_dir,
                                       tmp_artifacts):
    from src.pipeline_openf1.run_pipeline_openf1 import main
    cfg = _write_config(
        tmp_path, synthetic_data_dir, tmp_artifacts,
        pipeline="pipeline_openf1",
        extra={"monte_carlo": {"n_simulations": 60, "n_positions": 6,
                               "seed": 5},
               # anos sintéticos ficam abaixo do first_year → sem contexto,
               # exercitando o caminho de contexto vazio/parcial
               "context": {"first_year": 9999, "allow_partial": True},
               "visualization": {"top5_drivers": VIZ_DRIVERS}})
    main(["--config", cfg])
    run_dir, _ = _check_run(
        tmp_artifacts, "pipeline_openf1",
        expect_tables=["race_metrics", "rps_metrics",
                       "predictions", "position_probabilities"])
    rps = pd.read_parquet(run_dir / "rps_metrics.parquet")
    assert (~rps["has_context"]).all()
    assert (run_dir / "dnf_analysis.json").exists()
    assert (run_dir / "nb_data_p3.pkl").exists()
