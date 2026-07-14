"""Run manifest primitives shared by scientific pipelines."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_git_state(project_root: Path) -> tuple[str | None, bool | None]:
    """Return current commit and dirty status, if executed in a Git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return commit or None, bool(dirty_result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def dependency_versions() -> dict[str, str | None]:
    """Collect versions of packages that materially affect experiments."""
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "pyarrow",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


@dataclass
class RunManifest:
    """Metadata required to audit and reproduce one experiment run."""

    run_id: str
    model_id: str
    phase: int
    mode: str
    seed: int
    schema_version: str
    status: str
    started_at: str
    finished_at: str | None
    duration_seconds: float | None
    git_commit: str | None
    git_dirty: bool | None
    python_version: str
    platform: str
    dependencies: dict[str, str | None]
    config_sources: list[str]
    error: dict[str, Any] | None = None

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        model_id: str,
        phase: int,
        mode: str,
        seed: int,
        schema_version: str,
        project_root: Path,
        config_sources: list[str],
    ) -> "RunManifest":
        git_commit, git_dirty = detect_git_state(project_root)
        return cls(
            run_id=run_id,
            model_id=model_id,
            phase=phase,
            mode=mode,
            seed=seed,
            schema_version=schema_version,
            status="running",
            started_at=utc_now_iso(),
            finished_at=None,
            duration_seconds=None,
            git_commit=git_commit,
            git_dirty=git_dirty,
            python_version=sys.version.replace("\n", " "),
            platform=platform.platform(),
            dependencies=dependency_versions(),
            config_sources=config_sources,
        )

    def finish_success(self, duration_seconds: float) -> None:
        self.status = "success"
        self.finished_at = utc_now_iso()
        self.duration_seconds = round(float(duration_seconds), 6)
        self.error = None

    def finish_failure(self, duration_seconds: float, exc: BaseException) -> None:
        self.status = "failed"
        self.finished_at = utc_now_iso()
        self.duration_seconds = round(float(duration_seconds), 6)
        self.error = {"type": type(exc).__name__, "message": str(exc)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
