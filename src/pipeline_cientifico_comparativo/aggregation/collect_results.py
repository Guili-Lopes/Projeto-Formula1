"""Consolidate summaries produced by the central multi-model runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def write_model_summary(
    *,
    artifact_root: Path,
    mode: str,
    executions: list[dict[str, Any]],
) -> Path:
    """Write the comparison table for all models executed in one batch."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = artifact_root / "comparison" / mode / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = [execution["summary"] for execution in executions]
    summary_df = pd.DataFrame(rows)
    try:
        summary_df.to_parquet(
            output_dir / "model_summary.parquet",
            index=False,
            engine="pyarrow",
        )
    except (ImportError, ModuleNotFoundError):
        # CSV remains available even in a minimal environment.
        pass
    summary_df.to_csv(output_dir / "model_summary.csv", index=False)

    with (output_dir / "experiment_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "mode": mode,
                "created_at": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "models": [
                    {
                        "model_id": execution["model_id"],
                        "run_id": execution["run_id"],
                        "artifact_dir": execution["artifact_dir"],
                    }
                    for execution in executions
                ],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return output_dir
