"""Validation rules for scientific-comparative YAML configurations."""

from __future__ import annotations

from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def _require(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required configuration field: {path}")
        current = current[part]
    return current


def _require_int(
    config: dict[str, Any],
    path: str,
    *,
    minimum: int | None = None,
) -> int:
    value = _require(config, path)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be >= {minimum}")
    return value


def _require_number(
    config: dict[str, Any],
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = _require(config, path)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{path} must be numeric")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ConfigError(f"{path} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ConfigError(f"{path} must be <= {maximum}")
    return numeric


def _year_list(config: dict[str, Any], path: str) -> list[int]:
    value = _require(config, path)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty list")
    if any(not isinstance(year, int) or isinstance(year, bool) for year in value):
        raise ConfigError(f"{path} must contain integers only")
    if value != sorted(value):
        raise ConfigError(f"{path} must be sorted chronologically")
    if len(value) != len(set(value)):
        raise ConfigError(f"{path} must not contain duplicate years")
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Validate required fields, ranges and temporal separation."""
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a mapping")

    model_id = _require(config, "experiment.model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("experiment.model_id must be a non-empty string")

    _require_int(config, "experiment.phase", minimum=0)
    _require_int(config, "experiment.random_seed", minimum=0)

    train_years = _year_list(config, "data.train_years")
    validation_years = _year_list(config, "data.validation_years")
    test_years = _year_list(config, "data.test_years")

    if set(train_years) & set(validation_years):
        raise ConfigError("Training and validation years overlap")
    if set(train_years) & set(test_years):
        raise ConfigError("Training and test years overlap")
    if set(validation_years) & set(test_years):
        raise ConfigError("Validation and test years overlap")
    if max(train_years) >= min(validation_years):
        raise ConfigError("All training years must precede validation years")
    if max(validation_years) >= min(test_years):
        raise ConfigError("All validation years must precede test years")

    _require_int(config, "data.top_k", minimum=2)
    _require_int(config, "mallows.n_clusters", minimum=1)
    _require_int(config, "mallows.n_iterations", minimum=1)
    _require_int(config, "mallows.refit_iterations", minimum=1)
    _require_number(config, "mallows.alpha", minimum=0.0)
    _require_int(config, "plackett_luce.n_iterations", minimum=1)
    _require_number(
        config,
        "prediction.cluster_weight",
        minimum=0.0,
        maximum=1.0,
    )
    _require_int(config, "prediction.min_cluster_size", minimum=1)
    _require_int(config, "simulation.n_simulations", minimum=1)
    _require_int(config, "simulation.n_positions", minimum=2)

    score_source = _require(config, "simulation.score_source")
    if score_source not in {"cluster", "combined", "global"}:
        raise ConfigError(
            "simulation.score_source must be 'cluster', 'combined' or 'global'"
        )

    for path in [
        "paths.data_dir",
        "paths.artifact_root",
        "artifacts.schema_version",
    ]:
        value = _require(config, path)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{path} must be a non-empty string")

    for path in [
        "artifacts.save_model",
        "artifacts.save_notebook_bundle",
        "artifacts.csv_mirrors",
        "artifacts.parquet_required",
        "development.test_years_locked",
    ]:
        value = _require(config, path)
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be boolean")
