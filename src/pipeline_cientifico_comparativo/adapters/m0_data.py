"""Data selection and audit helpers for the M0 baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.data.data_pipeline import RaceRecord


def evaluation_years_for_mode(
    config: dict[str, Any],
    *,
    mode: str,
    allow_test: bool,
) -> tuple[list[int], list[int], list[int]]:
    """Return loaded, warm-up and evaluated years for one execution mode."""
    train_years = list(config["data"]["train_years"])
    validation_years = list(config["data"]["validation_years"])
    test_years = list(config["data"]["test_years"])

    if mode == "validation":
        # The final test set is not even loaded while models are developed.
        return train_years + validation_years, [], validation_years

    if mode != "final_test":
        raise ValueError("mode must be 'validation' or 'final_test'")

    if config["development"]["test_years_locked"] and not allow_test:
        raise PermissionError(
            "The 2025 test set is locked. Use --unlock-final-test only after "
            "all model and hyperparameter decisions have been frozen."
        )

    # Validation races are incorporated sequentially before untouched 2025 is
    # evaluated, preserving the project's incremental learning protocol.
    return (
        train_years + validation_years + test_years,
        validation_years,
        test_years,
    )


def _find_race_result_files(data_dir: Path, year: int) -> list[Path]:
    season_dir = data_dir / f"Season{year}"
    if not season_dir.is_dir():
        return []
    return sorted(
        path
        for path in season_dir.iterdir()
        if path.is_file()
        and "raceresults" in path.name.lower()
        and "sprint" not in path.name.lower()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_environment(
    config: dict[str, Any],
    *,
    mode: str,
    allow_test: bool,
) -> dict[str, Any]:
    """Inspect required inputs and return an auditable data inventory."""
    data_dir = Path(config["_meta"]["data_dir_resolved"])
    load_years, warmup_years, evaluation_years = evaluation_years_for_mode(
        config,
        mode=mode,
        allow_test=allow_test,
    )

    seasons: list[dict[str, Any]] = []
    for year in load_years:
        season_dir = data_dir / f"Season{year}"
        files = _find_race_result_files(data_dir, year)
        seasons.append(
            {
                "year": year,
                "directory": str(season_dir),
                "directory_exists": season_dir.is_dir(),
                "files": [
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in files
                ],
                "ready": bool(files),
            }
        )

    return {
        "data_dir": str(data_dir),
        "mode": mode,
        "load_years": load_years,
        "warmup_years": warmup_years,
        "evaluation_years": evaluation_years,
        "test_years_loaded": any(
            year in set(config["data"]["test_years"]) for year in load_years
        ),
        "seasons": seasons,
        "ready": all(item["ready"] for item in seasons),
    }


def record_rows(records: list[RaceRecord], split: str) -> list[dict[str, Any]]:
    """Convert race records to a stable tabular representation."""
    return [
        {
            "split": split,
            "split_index": index,
            "season": record.season,
            "race": record.race,
            "ranking": "|".join(record.ranking),
            "n_ranked": len(record.ranking),
            "n_classified": record.n_classified,
            "n_dnf": record.n_dnf,
        }
        for index, record in enumerate(records)
    ]
