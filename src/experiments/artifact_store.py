"""Versioned and non-destructive artifact storage for experiments."""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class ArtifactStore:
    """Write all outputs of one run into an immutable run directory."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        mode: str,
        model_id: str,
        seed: int,
        run_id: str | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = run_id or f"{model_id}_{mode}_seed{seed}_{timestamp}"
        self.seed_root = artifact_root / mode / model_id / f"seed_{seed}"
        self.run_dir = self.seed_root / self.run_id

        if self.run_dir.exists():
            raise FileExistsError(
                "Artifact directory already exists and will not be overwritten: "
                f"{self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def path(self, relative_name: str) -> Path:
        """Return a path inside the immutable run directory."""
        target = (self.run_dir / relative_name).resolve()
        run_dir_resolved = self.run_dir.resolve()
        if target != run_dir_resolved and run_dir_resolved not in target.parents:
            raise ValueError(f"Artifact path escapes run directory: {relative_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _atomic_replace(temp_path: Path, final_path: Path) -> None:
        os.replace(temp_path, final_path)

    def write_json(self, relative_name: str, payload: Any) -> Path:
        path = self.path(relative_name)
        temp = path.with_name(path.name + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        self._atomic_replace(temp, path)
        return path

    def write_yaml(self, relative_name: str, payload: Any) -> Path:
        path = self.path(relative_name)
        temp = path.with_name(path.name + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
        self._atomic_replace(temp, path)
        return path

    def write_text(self, relative_name: str, text: str) -> Path:
        path = self.path(relative_name)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(text, encoding="utf-8")
        self._atomic_replace(temp, path)
        return path

    def write_pickle(self, relative_name: str, payload: Any) -> Path:
        path = self.path(relative_name)
        temp = path.with_name(path.name + ".tmp")
        with temp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._atomic_replace(temp, path)
        return path

    def write_dataframe(
        self,
        stem: str,
        dataframe: pd.DataFrame,
        *,
        csv_mirror: bool = False,
        parquet_required: bool = True,
    ) -> dict[str, Path]:
        """Persist a DataFrame as Parquet and optionally as CSV.

        If ``parquet_required`` is false, a missing Parquet engine results in a
        CSV-only artifact rather than an interrupted experiment.
        """
        written: dict[str, Path] = {}
        parquet_path = self.path(f"{stem}.parquet")
        temp_parquet = parquet_path.with_name(parquet_path.name + ".tmp")
        try:
            dataframe.to_parquet(temp_parquet, index=False, engine="pyarrow")
            self._atomic_replace(temp_parquet, parquet_path)
            written["parquet"] = parquet_path
        except (ImportError, ModuleNotFoundError) as exc:
            if temp_parquet.exists():
                temp_parquet.unlink()
            if parquet_required:
                raise RuntimeError(
                    "Parquet output requires pyarrow. Run "
                    "'pip install -r requirements.txt'."
                ) from exc

        if csv_mirror or not written:
            csv_path = self.path(f"{stem}.csv")
            temp_csv = csv_path.with_name(csv_path.name + ".tmp")
            dataframe.to_csv(temp_csv, index=False)
            self._atomic_replace(temp_csv, csv_path)
            written["csv"] = csv_path

        return written

    def update_latest_pointer(self, summary: dict[str, Any] | None = None) -> Path:
        """Write a mutable pointer; immutable run directories remain untouched."""
        self.seed_root.mkdir(parents=True, exist_ok=True)
        pointer = self.seed_root / "latest_run.json"
        temp = pointer.with_name(pointer.name + ".tmp")
        payload = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "summary": summary or {},
        }
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        self._atomic_replace(temp, pointer)
        return pointer
