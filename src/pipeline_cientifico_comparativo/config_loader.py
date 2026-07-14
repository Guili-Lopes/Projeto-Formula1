"""Load, merge, resolve and validate YAML experiment configurations."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.pipeline_cientifico_comparativo.config_schema import validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file and require a mapping at its root."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return payload


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_resolved_config(
    *,
    model_config: str | Path = "m0_baseline.yaml",
    common_config: str | Path = "common.yaml",
    seed_override: int | None = None,
    artifact_root_override: str | Path | None = None,
) -> dict[str, Any]:
    """Load common + model YAML and add resolved path metadata."""
    common_path = Path(common_config)
    if not common_path.is_absolute():
        common_path = DEFAULT_CONFIG_DIR / common_path

    model_path = Path(model_config)
    if not model_path.is_absolute():
        model_path = DEFAULT_CONFIG_DIR / model_path

    config = deep_merge(load_yaml(common_path), load_yaml(model_path))

    if seed_override is not None:
        config.setdefault("experiment", {})["random_seed"] = int(seed_override)
    if artifact_root_override is not None:
        config.setdefault("paths", {})["artifact_root"] = str(artifact_root_override)

    validate_config(config)

    data_dir = _resolve_path(PROJECT_ROOT, config["paths"]["data_dir"])
    artifact_root = _resolve_path(PROJECT_ROOT, config["paths"]["artifact_root"])

    config["_meta"] = {
        "loaded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "data_dir_resolved": str(data_dir),
        "artifact_root_resolved": str(artifact_root),
        "config_sources": [str(common_path), str(model_path)],
    }
    return config
