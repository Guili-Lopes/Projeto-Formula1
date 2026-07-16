"""Unit — modelos, engine incremental e métricas de avaliação."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.repository import get_all_drivers, load_season_records
from src.engine.engine_predictor import predict
from src.engine.engine_trainer import incremental_update, initial_fit
from src.evaluation.additional_metrics import (
    mean_absolute_position_error,
    winner_correct,
)
from src.evaluation.evaluation_metrics import evaluate_race, season_summary
from src.models.models_weights import compute as compute_weights
from src.pipeline_score_rules.monte_carlo import simulate, uniform_baseline
from src.pipeline_score_rules.scoring_rules import compute_rps


@pytest.fixture()
def fitted_state(synthetic_data_dir):
    records, _ = load_season_records(
        [3001, 3002], top_k=5,
        source_policy="legacy_only", data_dir=synthetic_data_dir)
    drivers = get_all_drivers(records)
    state = initial_fit(records=records, all_drivers=drivers,
                        n_clusters=2, n_iter=8, alpha=0.5, verbose=False)
    return state, records, drivers


def test_initial_fit_and_predict(fitted_state):
    state, records, drivers = fitted_state
    assert state.n_clusters == 2
    assert len(state.assignments) == len(records)
    pred = predict(state, 3003, records[0].race)
    assert pred.cluster_used in (0, 1)
    assert sorted(pred.predicted_order) == sorted(drivers)
    assert len(set(pred.predicted_order)) == len(drivers)


def test_incremental_update_appends_record(fitted_state, synthetic_data_dir):
    state, _, _ = fitted_state
    new_records, _ = load_season_records(
        [3003], top_k=5, source_policy="legacy_only",
        data_dir=synthetic_data_dir)
    before = len(state.seen_records)
    state = incremental_update(state=state, new_record=new_records[0],
                               refit_mallows=False, verbose=False)
    assert len(state.seen_records) == before + 1


def test_evaluate_race_bounds_and_perfection():
    order = ["VER", "NOR", "LEC", "HAM", "PIA", "RUS"]
    ev = evaluate_race(3001, "Bahrain", order, order, cluster_used=0)
    assert ev.top3_acc == 1.0 and ev.top5_acc == 1.0
    assert ev.kendall_tau == pytest.approx(1.0)

    shuffled = ["RUS", "PIA", "HAM", "LEC", "NOR", "VER"]
    ev2 = evaluate_race(3001, "Bahrain", shuffled, order, cluster_used=0)
    assert ev2.kendall_tau == pytest.approx(-1.0)
    assert 0.0 <= ev2.top3_acc <= 1.0

    summary = season_summary([ev, ev2], 3001)
    assert summary.n_races == 2
    assert summary.mean_top3 == pytest.approx((ev.top3_acc + ev2.top3_acc) / 2)


def test_additional_metrics_ported_from_legacy_suite():
    assert winner_correct(["VER", "NOR"], ["VER", "LEC"])
    assert not winner_correct(["NOR", "VER"], ["VER", "NOR"])
    assert not winner_correct([], ["VER"])
    assert mean_absolute_position_error(
        ["VER", "NOR", "LEC"], ["VER", "LEC", "NOR"]) == pytest.approx(2 / 3)


def test_regulatory_weights_shape():
    seasons = [3001, 3001, 3002]
    races = ["Bahrain", "Monaco", "Bahrain"]
    weights = compute_weights(seasons, races)
    assert len(weights) == 3
    assert all(w.final_weight > 0 for w in weights)


def test_monte_carlo_is_seed_deterministic():
    scores = {"VER": 0.5, "NOR": 0.3, "LEC": 0.2}
    kwargs = dict(skill_scores=scores, season=3001, race="Bahrain",
                  cluster=0, n_positions=3, n_simulations=400)
    d1 = simulate(seed=123, **kwargs)
    d2 = simulate(seed=123, **kwargs)
    d3 = simulate(seed=321, **kwargs)
    for drv in scores:
        assert np.allclose(d1.vectors[drv].probs, d2.vectors[drv].probs)
    assert any(not np.allclose(d1.vectors[d].probs, d3.vectors[d].probs)
               for d in scores)
    # vetores são distribuições parciais válidas
    for drv in scores:
        probs = d1.vectors[drv].probs
        assert len(probs) == 3
        assert (probs >= 0).all() and probs.sum() <= 1.0 + 1e-9
    # colunas somam 1: alguém ocupa cada posição
    col_sums = np.sum([d1.vectors[d].probs for d in scores], axis=0)
    assert np.allclose(col_sums, 1.0)


def test_rps_model_beats_uniform_when_confident():
    drivers = ["VER", "NOR", "LEC"]
    actual = ["VER", "NOR", "LEC"]
    sharp = simulate(skill_scores={"VER": 0.90, "NOR": 0.09, "LEC": 0.01},
                     season=3001, race="Bahrain", cluster=0,
                     n_positions=3, n_simulations=3000, seed=7)
    base = uniform_baseline(drivers=drivers, season=3001,
                            race="Bahrain", n_positions=3)
    rps = compute_rps(sharp, base, actual)
    assert rps.rps_model < rps.rps_baseline
    assert rps.gain == pytest.approx(rps.rps_baseline - rps.rps_model)
    assert rps.rps_model >= 0
