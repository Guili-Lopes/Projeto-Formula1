"""Public M0 adapter API kept stable for CLI and tests."""

from src.pipeline_cientifico_comparativo.adapters.m0_data import (
    evaluation_years_for_mode,
    validate_environment,
)
from src.pipeline_cientifico_comparativo.adapters.m0_execution import run_m0

__all__ = ["evaluation_years_for_mode", "run_m0", "validate_environment"]
