"""Integration — fluxo dados → engine → avaliação → artefatos."""

from __future__ import annotations

import json
import pickle

from src.data.repository import get_all_drivers, load_season_records
from src.engine.engine_predictor import predict
from src.engine.engine_trainer import incremental_update, initial_fit
from src.evaluation.evaluation_metrics import evaluate_race, season_summary
from src.experiments.run_store import PipelineRunStore, pipeline_run


def test_data_to_metrics_to_artifacts(synthetic_data_dir, tmp_artifacts):
    records, provenance = load_season_records(
        [3001, 3002, 3003], top_k=5,
        source_policy="legacy_only", data_dir=synthetic_data_dir)
    drivers = get_all_drivers(records)
    train = [r for r in records if r.season in (3001, 3002)]
    evalset = [r for r in records if r.season == 3003]

    state = initial_fit(records=train, all_drivers=drivers,
                        n_clusters=2, n_iter=8, alpha=0.5, verbose=False)

    store = PipelineRunStore("fluxo", artifacts_root=tmp_artifacts)
    with pipeline_run(store):
        evals = []
        for i, rec in enumerate(evalset):
            pred = predict(state, rec.season, rec.race)
            evals.append(evaluate_race(rec.season, rec.race,
                                       pred.predicted_order, rec.ranking,
                                       pred.cluster_used))
            state = incremental_update(
                state=state, new_record=rec,
                refit_mallows=(i == len(evalset) - 1), verbose=False)

        summary = season_summary(evals, 3003)
        store.write_json("metrics_summary.json", {
            "n_races": summary.n_races,
            "mean_top3": summary.mean_top3,
            "provenance": provenance,
        })
        store.write_pickle("state.pkl", state)
        store.finalize(summary={"mean_top3": summary.mean_top3})

    run_dir = store.run_dir
    saved = json.loads(
        (run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert saved["n_races"] == 3
    assert 0.0 <= saved["mean_top3"] <= 1.0

    with (run_dir / "state.pkl").open("rb") as fh:
        restored = pickle.load(fh)
    assert len(restored.seen_records) == len(train) + len(evalset)

    pointer = json.loads(
        (tmp_artifacts / "fluxo" / "latest_run.json").read_text("utf-8"))
    assert pointer["run_id"] == store.run_id
