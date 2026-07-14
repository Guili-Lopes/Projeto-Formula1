"""Reusable deterministic metrics added for comparative experiments."""

from __future__ import annotations

import numpy as np


def winner_correct(predicted: list[str], actual: list[str]) -> bool:
    """Return whether the predicted and observed winner are equal."""
    return bool(predicted and actual and predicted[0] == actual[0])


def mean_absolute_position_error(
    predicted: list[str],
    actual: list[str],
) -> float:
    """Mean absolute position error over drivers present in both rankings."""
    predicted_position = {
        driver: position for position, driver in enumerate(predicted, start=1)
    }
    errors = [
        abs(predicted_position[driver] - actual_position)
        for actual_position, driver in enumerate(actual, start=1)
        if driver in predicted_position
    ]
    return float(np.mean(errors)) if errors else 0.0
