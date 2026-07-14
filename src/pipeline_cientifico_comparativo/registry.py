"""Central registry of comparable scientific models."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    phase: int
    config_file: str
    runner: str
    description: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "M0": ModelSpec(
        model_id="M0",
        phase=0,
        config_file="m0_baseline.yaml",
        runner=(
            "src.pipeline_cientifico_comparativo.phases.phase_00_baseline:"
            "run_phase_00"
        ),
        description="Mallows + Plackett-Luce original, Monte Carlo and RPS",
    ),
}


def get_model_by_phase(phase: int) -> ModelSpec:
    matches = [spec for spec in MODEL_REGISTRY.values() if spec.phase == phase]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one registered model for phase {phase}")
    return matches[0]


def load_runner(spec: ModelSpec) -> Callable[..., dict[str, Any]]:
    module_name, function_name = spec.runner.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    runner = getattr(module, function_name)
    if not callable(runner):
        raise TypeError(f"Registered runner is not callable: {spec.runner}")
    return runner
