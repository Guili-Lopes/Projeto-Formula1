"""Fase 0: reproduzir e congelar o baseline científico M0."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

from src.experiments import ArtifactStore, RunManifest, set_global_seed
from src.experiments.console_capture import capture_console
from src.pipeline_cientifico_comparativo.adapters.m0_adapter import (
    run_m0,
    validate_environment,
)


def _reference_comparison(
    summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compara a execução reproduzível do M0 com a referência histórica."""
    reference = config.get("reference_validation", {})
    expected = reference.get("metrics", {})
    tolerances = reference.get("tolerances", {})
    deterministic_tolerance = float(tolerances.get("deterministic", 0.001))
    probabilistic_tolerance = float(tolerances.get("probabilistic", 0.010))

    probabilistic_metrics = {
        "mean_rps_model",
        "mean_rps_baseline",
        "mean_gain",
    }
    comparisons: dict[str, Any] = {}
    all_within_tolerance = True

    expected_races = reference.get("expected_races")
    if expected_races is not None:
        observed_races = int(summary.get("n_races", -1))
        race_ok = observed_races == int(expected_races)
        comparisons["n_races"] = {
            "expected": int(expected_races),
            "observed": observed_races,
            "within_tolerance": race_ok,
        }
        all_within_tolerance = all_within_tolerance and race_ok

    for metric, expected_value in expected.items():
        if metric not in summary:
            comparisons[metric] = {
                "expected": float(expected_value),
                "observed": None,
                "within_tolerance": False,
                "reason": "metric_missing",
            }
            all_within_tolerance = False
            continue

        observed = float(summary[metric])
        tolerance = (
            probabilistic_tolerance
            if metric in probabilistic_metrics
            else deterministic_tolerance
        )
        difference = abs(observed - float(expected_value))
        within_tolerance = difference <= tolerance
        comparisons[metric] = {
            "expected": float(expected_value),
            "observed": observed,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "within_tolerance": within_tolerance,
        }
        all_within_tolerance = all_within_tolerance and within_tolerance

    return {
        "status": "equivalent" if all_within_tolerance else "review_required",
        "all_within_tolerance": all_within_tolerance,
        "comparisons": comparisons,
        "note": (
            "A execução histórica do Pipeline 3 carregava o universo de pilotos "
            "de 2025 durante a validação de 2024 e não fixava uma semente por "
            "corrida no Monte Carlo. O M0 científico mantém 2025 intocado e "
            "usa sementes estáveis por corrida; por isso a referência "
            "probabilística é aproximada, e não bit a bit."
        ),
    }


def _phase_audit_text(project_root: Path) -> str:
    audit_path = (
        project_root
        / "src"
        / "pipeline_cientifico_comparativo"
        / "docs"
        / "PHASE_00_AUDIT.md"
    )
    if audit_path.exists():
        return audit_path.read_text(encoding="utf-8")
    return "# Fase 0\n\nRelatório de auditoria não encontrado no checkout.\n"


def _persist_tables(
    *,
    store: ArtifactStore,
    result: Any,
    config: dict[str, Any],
) -> dict[str, str]:
    """Persiste tabelas oficiais e retorna os caminhos preferenciais."""
    csv_mirrors = bool(config["artifacts"].get("csv_mirrors", True))
    parquet_required = bool(
        config["artifacts"].get("parquet_required", True)
    )
    artifact_paths: dict[str, str] = {}

    tables = [
        ("race_metrics", result.race_metrics, csv_mirrors),
        ("predictions", result.predictions, csv_mirrors),
        ("position_probabilities", result.position_probabilities, False),
        ("parameter_history", result.parameter_history, False),
    ]
    for stem, dataframe, mirror in tables:
        written = store.write_dataframe(
            stem,
            dataframe,
            csv_mirror=mirror,
            parquet_required=parquet_required,
        )
        preferred = written.get("parquet") or written.get("csv")
        if preferred is not None:
            artifact_paths[stem] = str(preferred)

    for stem, dataframe in result.extra_tables.items():
        written = store.write_dataframe(
            stem,
            dataframe,
            csv_mirror=csv_mirrors,
            parquet_required=parquet_required,
        )
        preferred = written.get("parquet") or written.get("csv")
        if preferred is not None:
            artifact_paths[stem] = str(preferred)

    return artifact_paths


def run_phase_00(
    config: dict[str, Any],
    *,
    mode: str = "validation",
    allow_test: bool = False,
) -> dict[str, Any]:
    """Executa o M0, salva todos os artefatos e retorna um registro compacto."""
    model_id = str(config["experiment"]["model_id"])
    seed = int(config["experiment"]["random_seed"])
    phase = int(config["experiment"]["phase"])
    project_root = Path(config["_meta"]["project_root"])
    artifact_root = Path(config["_meta"]["artifact_root_resolved"])

    # A trava temporal é avaliada antes de qualquer leitura de corrida.
    environment = validate_environment(
        config,
        mode=mode,
        allow_test=allow_test,
    )

    store = ArtifactStore(
        artifact_root=artifact_root,
        mode=mode,
        model_id=model_id,
        seed=seed,
    )
    manifest = RunManifest.start(
        run_id=store.run_id,
        model_id=model_id,
        phase=phase,
        mode=mode,
        seed=seed,
        schema_version=str(config["artifacts"]["schema_version"]),
        project_root=project_root,
        config_sources=list(config["_meta"]["config_sources"]),
    )

    store.write_yaml("config_resolved.yaml", config)
    store.write_json("environment.json", environment)
    store.write_text("phase_00_audit.md", _phase_audit_text(project_root))
    store.write_json("manifest.json", manifest.to_dict())

    started = time.perf_counter()
    captured_warnings: list[dict[str, Any]] = []
    seed_audit = set_global_seed(
        seed,
        include_torch=bool(config["runtime"].get("include_torch_seed", False)),
    )

    try:
        if not environment["ready"]:
            missing = [
                item["year"]
                for item in environment["seasons"]
                if not item["ready"]
            ]
            raise FileNotFoundError(
                f"Arquivos de resultados indisponíveis para as temporadas: {missing}"
            )

        with capture_console(store.path("run.log"), echo=True):
            print("=" * 72)
            print("FASE 0 - BASELINE CIENTÍFICO M0")
            print("=" * 72)
            print(f"Run ID: {store.run_id}")
            print(f"Modo: {mode}")
            print(f"Semente: {seed}")
            print(f"Artefatos: {store.run_dir}")
            print(f"Anos de teste carregados: {environment['test_years_loaded']}")

            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                result = run_m0(config, mode=mode, allow_test=allow_test)

            captured_warnings = [
                {
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "lineno": warning.lineno,
                }
                for warning in warning_records
            ]

            elapsed = time.perf_counter() - started
            result.summary["run_id"] = store.run_id
            result.summary["duration_seconds"] = elapsed
            result.summary["artifact_dir"] = str(store.run_dir)

            artifact_paths = _persist_tables(
                store=store,
                result=result,
                config=config,
            )

            cluster_consensus = {
                "n_clusters": result.state.n_clusters,
                "cluster_sizes": list(result.state.mallows.cluster_sizes),
                "consensus": {
                    str(cluster_id): list(
                        result.state.mallows.consensos[cluster_id]
                    )
                    for cluster_id in range(result.state.n_clusters)
                },
            }
            artifact_paths["cluster_consensus"] = str(
                store.write_json("cluster_consensus.json", cluster_consensus)
            )

            store.write_json("metrics_summary.json", result.summary)
            equivalence = _reference_comparison(result.summary, config)
            store.write_json("baseline_equivalence.json", equivalence)
            store.write_json("warnings.json", captured_warnings)
            store.write_json(
                "runtime.json",
                {
                    "duration_seconds": elapsed,
                    "seed_audit": seed_audit,
                    "n_warnings": len(captured_warnings),
                },
            )

            if bool(config["artifacts"].get("save_model", True)):
                artifact_paths["model"] = str(
                    store.write_pickle("model.pkl", result.state)
                )

            if bool(config["artifacts"].get("save_notebook_bundle", True)):
                artifact_paths["notebook_bundle"] = str(
                    store.write_pickle(
                        "notebook_bundle.pkl",
                        result.notebook_bundle(artifact_paths),
                    )
                )

            manifest.finish_success(elapsed)
            store.write_json("manifest.json", manifest.to_dict())
            store.update_latest_pointer(result.summary)

            print("\nResumo:")
            for key in [
                "n_races",
                "mean_top3",
                "mean_top5",
                "mean_kendall",
                "winner_accuracy",
                "mean_absolute_position_error",
                "mean_rps_model",
                "mean_rps_baseline",
                "mean_gain",
            ]:
                print(f"  {key}: {result.summary[key]}")
            print(f"Equivalência do baseline: {equivalence['status']}")
            print(f"Artefatos salvos em: {store.run_dir}")

        if (
            mode == "validation"
            and bool(
                config.get("reference_validation", {}).get(
                    "fail_on_mismatch",
                    False,
                )
            )
            and not equivalence["all_within_tolerance"]
        ):
            raise RuntimeError(
                "O M0 não reproduziu a referência dentro da tolerância. "
                "Consulte baseline_equivalence.json."
            )

        return {
            "model_id": model_id,
            "phase": phase,
            "mode": mode,
            "seed": seed,
            "run_id": store.run_id,
            "artifact_dir": str(store.run_dir),
            "summary": result.summary,
            "equivalence": equivalence,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - started
        manifest.finish_failure(elapsed, exc)
        store.write_json("manifest.json", manifest.to_dict())
        store.write_json(
            "error.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        store.write_json("warnings.json", captured_warnings)
        raise
