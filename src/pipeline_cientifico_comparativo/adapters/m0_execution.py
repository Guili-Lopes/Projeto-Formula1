"""Execution of the configured Mallows + Plackett-Luce M0 baseline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.data_pipeline import RaceRecord, get_all_drivers, load_seasons
from src.engine.engine_predictor import Prediction, predict as make_prediction
from src.engine.engine_trainer import ModelState, incremental_update, initial_fit
from src.evaluation.additional_metrics import (
    mean_absolute_position_error,
    winner_correct,
)
from src.evaluation.evaluation_metrics import evaluate_race
from src.experiments.experiment_result import ExperimentResult
from src.experiments.reproducibility import derive_seed
from src.pipeline_cientifico_comparativo.adapters.m0_data import (
    evaluation_years_for_mode,
    record_rows,
)
from src.pipeline_cientifico_comparativo.adapters.m0_tables import (
    distribution_rows,
    parameter_rows,
    prediction_rows,
)
from src.pipeline_score_rules.monte_carlo import simulate, uniform_baseline
from src.pipeline_score_rules.scoring_rules import compute_rps


def _select_simulation_scores(
    state: ModelState,
    prediction: Prediction,
    score_source: str,
) -> dict[str, float]:
    if score_source == "cluster":
        return dict(
            state.pl.cluster_scores.get(
                prediction.cluster_used,
                state.pl.global_scores,
            )
        )
    if score_source == "combined":
        return dict(prediction.scores_combined)
    if score_source == "global":
        return dict(state.pl.global_scores)
    raise ValueError(f"Unsupported score_source: {score_source}")


def _should_refit_mallows(records: list[RaceRecord], index: int) -> bool:
    is_last = index == len(records) - 1
    season_turn = (
        index < len(records) - 1
        and records[index + 1].season != records[index].season
    )
    return is_last or season_turn


def _warmup_state(
    state: ModelState,
    records: list[RaceRecord],
    *,
    verbose: bool,
) -> ModelState:
    """Observe validation races without collecting final-test metrics."""
    if verbose and records:
        print(f"\n[Warm-up] Incorporando {len(records)} corridas antes do teste final...")
    for index, record in enumerate(records):
        state = incremental_update(
            state=state,
            new_record=record,
            refit_mallows=_should_refit_mallows(records, index),
            verbose=False,
        )
    return state


def run_m0(
    config: dict[str, Any],
    *,
    mode: str,
    allow_test: bool,
) -> ExperimentResult:
    """Execute the configured M0 baseline and return standardized artifacts."""
    model_id = str(config["experiment"]["model_id"])
    seed = int(config["experiment"]["random_seed"])
    verbose = bool(config["runtime"].get("verbose", True))
    data_dir = str(config["_meta"]["data_dir_resolved"])

    load_years, warmup_years, evaluation_years = evaluation_years_for_mode(
        config,
        mode=mode,
        allow_test=allow_test,
    )
    all_records = load_seasons(
        data_dir=data_dir,
        seasons=load_years,
        top_k=int(config["data"]["top_k"]),
    )

    available_years = {record.season for record in all_records}
    missing_years = [year for year in load_years if year not in available_years]
    if missing_years:
        raise FileNotFoundError(
            f"No valid race records were loaded for seasons: {missing_years}"
        )

    train_year_set = set(config["data"]["train_years"])
    warmup_year_set = set(warmup_years)
    evaluation_year_set = set(evaluation_years)

    train_records = [
        record for record in all_records if record.season in train_year_set
    ]
    warmup_records = [
        record for record in all_records if record.season in warmup_year_set
    ]
    evaluation_records = [
        record for record in all_records if record.season in evaluation_year_set
    ]

    if not train_records:
        raise ValueError("No training records were loaded")
    if not evaluation_records:
        raise ValueError("No evaluation records were loaded")

    # In validation mode, 2025 is absent from this universe by construction.
    all_drivers = get_all_drivers(all_records)

    if verbose:
        print("\n[Dados]")
        print(f"  Anos carregados: {load_years}")
        print(f"  Treino: {len(train_records)} corridas")
        print(f"  Warm-up: {len(warmup_records)} corridas")
        print(f"  Avaliação: {len(evaluation_records)} corridas")
        print(f"  Pilotos no universo da execução: {len(all_drivers)}")

    state = initial_fit(
        records=train_records,
        all_drivers=all_drivers,
        n_clusters=int(config["mallows"]["n_clusters"]),
        n_iter=int(config["mallows"]["n_iterations"]),
        alpha=float(config["mallows"]["alpha"]),
        verbose=verbose,
        pl_n_iter=int(config["plackett_luce"]["n_iterations"]),
        refit_mallows_n_iter=int(config["mallows"]["refit_iterations"]),
    )
    state = _warmup_state(state, warmup_records, verbose=verbose)

    race_rows: list[dict[str, Any]] = []
    predicted_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []

    if verbose:
        print(f"\n[Avaliação] {mode}")

    for race_index, record in enumerate(evaluation_records):
        skill_rows.extend(
            parameter_rows(
                state,
                model_id=model_id,
                mode=mode,
                seed=seed,
                season=record.season,
                race=record.race,
                race_index=race_index,
            )
        )

        prediction = make_prediction(
            state,
            record.season,
            record.race,
            cluster_weight=float(config["prediction"]["cluster_weight"]),
            min_cluster_size=int(config["prediction"]["min_cluster_size"]),
        )
        simulation_scores = _select_simulation_scores(
            state,
            prediction,
            str(config["simulation"]["score_source"]),
        )
        monte_carlo_seed = derive_seed(
            seed,
            model_id,
            mode,
            record.season,
            record.race,
            race_index,
            "monte_carlo",
        )
        distribution = simulate(
            skill_scores=simulation_scores,
            season=record.season,
            race=record.race,
            cluster=prediction.cluster_used,
            n_positions=int(config["simulation"]["n_positions"]),
            n_simulations=int(config["simulation"]["n_simulations"]),
            seed=monte_carlo_seed,
        )
        baseline = uniform_baseline(
            drivers=state.all_drivers,
            season=record.season,
            race=record.race,
            n_positions=int(config["simulation"]["n_positions"]),
        )

        evaluation = evaluate_race(
            season=record.season,
            race=record.race,
            predicted=prediction.predicted_order,
            actual=record.ranking,
            cluster_used=prediction.cluster_used,
        )
        rps_result = compute_rps(distribution, baseline, record.ranking)

        race_rows.append(
            {
                "model_id": model_id,
                "mode": mode,
                "seed": seed,
                "season": record.season,
                "race": record.race,
                "race_index": race_index,
                "cluster_used": prediction.cluster_used,
                "cluster_size_pre_prediction": state.assignments.count(
                    prediction.cluster_used
                ),
                "top3_accuracy": evaluation.top3_acc,
                "top5_accuracy": evaluation.top5_acc,
                "kendall_tau": evaluation.kendall_tau,
                "winner_correct": int(
                    winner_correct(prediction.predicted_order, record.ranking)
                ),
                "mean_absolute_position_error": mean_absolute_position_error(
                    prediction.predicted_order,
                    record.ranking,
                ),
                "rps_model": rps_result.rps_model,
                "rps_baseline": rps_result.rps_baseline,
                "rps_gain": rps_result.gain,
                "n_observed_ranked": len(record.ranking),
                "n_classified": record.n_classified,
                "n_dnf": record.n_dnf,
                "monte_carlo_seed": monte_carlo_seed,
                "score_source": str(config["simulation"]["score_source"]),
            }
        )
        predicted_rows.extend(
            prediction_rows(
                prediction,
                record,
                distribution,
                model_id=model_id,
                mode=mode,
                seed=seed,
                race_index=race_index,
            )
        )
        probability_rows.extend(
            distribution_rows(
                distribution,
                model_id=model_id,
                mode=mode,
                seed=seed,
                race_index=race_index,
            )
        )

        if verbose:
            print(
                f"  {record.season} {record.race:22s} | "
                f"Top-3={evaluation.top3_acc:.3f} | "
                f"Kendall={evaluation.kendall_tau:.3f} | "
                f"RPS={rps_result.rps_model:.4f} | "
                f"Ganho={rps_result.gain:.4f}"
            )

        state = incremental_update(
            state=state,
            new_record=record,
            refit_mallows=_should_refit_mallows(evaluation_records, race_index),
            verbose=False,
        )

    race_metrics = pd.DataFrame(race_rows)
    predictions = pd.DataFrame(predicted_rows)
    position_probabilities = pd.DataFrame(probability_rows)
    parameter_history = pd.DataFrame(skill_rows)

    summary: dict[str, Any] = {
        "model_id": model_id,
        "phase": int(config["experiment"]["phase"]),
        "mode": mode,
        "seed": seed,
        "train_years": list(config["data"]["train_years"]),
        "warmup_years": warmup_years,
        "evaluation_years": evaluation_years,
        "test_years_loaded": any(
            year in set(config["data"]["test_years"]) for year in load_years
        ),
        "n_train_races": len(train_records),
        "n_warmup_races": len(warmup_records),
        "n_races": len(race_metrics),
        "n_drivers": len(all_drivers),
        "n_clusters": state.n_clusters,
        "mean_top3": float(race_metrics["top3_accuracy"].mean()),
        "mean_top5": float(race_metrics["top5_accuracy"].mean()),
        "mean_kendall": float(race_metrics["kendall_tau"].mean()),
        "winner_accuracy": float(race_metrics["winner_correct"].mean()),
        "mean_absolute_position_error": float(
            race_metrics["mean_absolute_position_error"].mean()
        ),
        "mean_rps_model": float(race_metrics["rps_model"].mean()),
        "mean_rps_baseline": float(race_metrics["rps_baseline"].mean()),
        "mean_gain": float(race_metrics["rps_gain"].mean()),
    }
    summary["rps_improvement_percent"] = (
        float(100.0 * summary["mean_gain"] / summary["mean_rps_baseline"])
        if summary["mean_rps_baseline"] > 0
        else 0.0
    )

    records_table = pd.DataFrame(
        record_rows(train_records, "train")
        + record_rows(warmup_records, "warmup")
        + record_rows(evaluation_records, mode)
    )
    cluster_assignments = pd.DataFrame(
        [
            {
                "season": record.season,
                "race": record.race,
                "assignment": assignment,
                "weight": weight,
            }
            for record, assignment, weight in zip(
                state.seen_records,
                state.assignments,
                state.seen_weights,
            )
        ]
    )

    return ExperimentResult(
        model_id=model_id,
        mode=mode,
        seed=seed,
        summary=summary,
        race_metrics=race_metrics,
        predictions=predictions,
        position_probabilities=position_probabilities,
        parameter_history=parameter_history,
        state=state,
        extra_tables={
            "records": records_table,
            "cluster_assignments": cluster_assignments,
        },
    )
