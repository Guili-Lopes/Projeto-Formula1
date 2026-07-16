"""
src/experiments/run_store.py
=============================
Armazenamento de execuções dos pipelines históricos (Etapas 4–6 da
reestruturação).

Layout gerado (plano, seção 5):

    artifacts/<pipeline>/
    ├── latest_run.json                 # ponteiro mutável para a última execução
    └── <run_id>/                       # pasta imutável da execução
        ├── config_resolved.yaml
        ├── manifest.json
        ├── metrics_summary.json
        ├── *.parquet (+ espelho .csv)
        ├── nb_data*.pkl                # PKL mantido como formato auxiliar
        ├── run.log
        ├── runtime.json
        ├── warnings.json               # quando houver avisos
        ├── error.json                  # somente quando a execução falha
        └── plots/*.png

Complementa (sem alterar) o ArtifactStore do pipeline científico, que usa a
hierarquia mode/model/seed.
"""

from __future__ import annotations

import json
import os
import pickle
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml

from src.experiments.console_capture import capture_console
from src.experiments.manifest import (
    dependency_versions,
    detect_git_state,
    utc_now_iso,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_ROOT = _PROJECT_ROOT / "artifacts"


class PipelineRunStore:
    """Grava todos os produtos de uma execução em uma pasta imutável."""

    def __init__(
        self,
        pipeline_name: str,
        *,
        artifacts_root: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        root = Path(artifacts_root) if artifacts_root else DEFAULT_ARTIFACTS_ROOT
        if not root.is_absolute():
            root = _PROJECT_ROOT / root
        self.pipeline_name = pipeline_name
        self.pipeline_dir = root / pipeline_name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = run_id or f"run_{timestamp}"
        self.run_dir = self.pipeline_dir / self.run_id
        if self.run_dir.exists():
            raise FileExistsError(
                f"Execução já existe e não será sobrescrita: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self._started = datetime.now(timezone.utc)
        self.warnings: list[str] = []

    # ── caminhos ─────────────────────────────────────────────────────────
    def path(self, relative_name: str) -> Path:
        target = (self.run_dir / relative_name).resolve()
        run_dir = self.run_dir.resolve()
        if target != run_dir and run_dir not in target.parents:
            raise ValueError(f"Caminho escapa da pasta da execução: {relative_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def plot_path(self, filename: str) -> str:
        return str(self.plots_dir / filename)

    # ── escrita atômica ──────────────────────────────────────────────────
    @staticmethod
    def _replace(temp: Path, final: Path) -> None:
        os.replace(temp, final)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.path(name)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._replace(temp, path)
        return path

    def write_yaml(self, name: str, payload: Any) -> Path:
        path = self.path(name)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._replace(temp, path)
        return path

    def write_pickle(self, name: str, payload: Any) -> Path:
        path = self.path(name)
        temp = path.with_name(path.name + ".tmp")
        with temp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._replace(temp, path)
        return path

    def write_dataframe(
        self, stem: str, df: pd.DataFrame, *, csv_mirror: bool = True
    ) -> dict[str, Path]:
        written: dict[str, Path] = {}
        parquet = self.path(f"{stem}.parquet")
        temp = parquet.with_name(parquet.name + ".tmp")
        try:
            df.to_parquet(temp, index=False, engine="pyarrow")
            self._replace(temp, parquet)
            written["parquet"] = parquet
        except Exception:
            if temp.exists():
                temp.unlink()
            safe = df.copy()
            for col in safe.columns:
                if safe[col].dtype == object:
                    safe[col] = safe[col].map(
                        lambda v: v
                        if (v is None or (isinstance(v, float) and pd.isna(v)))
                        else json.dumps(v, ensure_ascii=False, default=str)
                        if isinstance(v, (list, dict))
                        else str(v)
                    )
            safe.to_parquet(temp, index=False, engine="pyarrow")
            self._replace(temp, parquet)
            written["parquet"] = parquet
            df = safe
        if csv_mirror:
            csv = self.path(f"{stem}.csv")
            temp_csv = csv.with_name(csv.name + ".tmp")
            df.to_csv(temp_csv, index=False)
            self._replace(temp_csv, csv)
            written["csv"] = csv
        return written

    # ── metadados padrão ─────────────────────────────────────────────────
    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def write_manifest(self, *, config: dict, extra: dict | None = None) -> Path:
        commit, dirty = detect_git_state(_PROJECT_ROOT)
        payload = {
            "pipeline": self.pipeline_name,
            "run_id": self.run_id,
            "created_at": utc_now_iso(),
            "git_commit": commit,
            "git_dirty": dirty,
            "dependencies": dependency_versions(),
            "config": config,
        }
        if extra:
            payload.update(extra)
        return self.write_json("manifest.json", payload)

    def write_error(self, exc: BaseException) -> Path:
        return self.write_json("error.json", {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "occurred_at": utc_now_iso(),
        })

    def _relative_run_dir(self) -> str:
        try:
            return str(self.run_dir.relative_to(_PROJECT_ROOT))
        except ValueError:
            return str(self.run_dir)

    def finalize(self, summary: dict | None = None) -> Path:
        finished = datetime.now(timezone.utc)
        self.write_json("runtime.json", {
            "started_at": self._started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "duration_seconds": round(
                (finished - self._started).total_seconds(), 3
            ),
        })
        if self.warnings:
            self.write_json("warnings.json", {"warnings": self.warnings})
        pointer = self.pipeline_dir / "latest_run.json"
        temp = pointer.with_name(pointer.name + ".tmp")
        temp.write_text(
            json.dumps({
                "run_id": self.run_id,
                "run_dir": self._relative_run_dir(),
                "created_at": utc_now_iso(),
                "summary": summary or {},
            }, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._replace(temp, pointer)
        return pointer


@contextmanager
def pipeline_run(store: PipelineRunStore) -> Iterator[PipelineRunStore]:
    """
    Contexto padrão de execução: captura o console em run.log e, em caso de
    falha, grava error.json (mantendo o ponteiro latest_run.json intocado).
    """
    try:
        with capture_console(store.path("run.log")):
            yield store
    except BaseException as exc:  # noqa: BLE001 — registrar qualquer falha
        store.write_error(exc)
        raise
