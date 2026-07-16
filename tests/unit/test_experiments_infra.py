"""Unit — infraestrutura de execução: run_store, pipeline_config, seeds."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.experiments.manifest import dependency_versions, detect_git_state
from src.experiments.pipeline_config import load_pipeline_config
from src.experiments.reproducibility import derive_seed, set_global_seed
from src.experiments.run_store import PipelineRunStore, pipeline_run


def test_run_store_layout_and_pointer(tmp_artifacts):
    store = PipelineRunStore("demo", artifacts_root=tmp_artifacts,
                             run_id="run_A")
    store.write_json("metrics_summary.json", {"ok": 1})
    df = pd.DataFrame({"x": [1, 2], "mixed": [1.5, "+1 LAP"]})
    written = store.write_dataframe("tabela", df)
    assert written["parquet"].exists() and written["csv"].exists()
    store.write_pickle("obj.pkl", {"a": np.arange(3)})
    (store.plots_dir / "fig.png").write_bytes(b"png")
    pointer = store.finalize(summary={"m": 0.5})

    run_dir = tmp_artifacts / "demo" / "run_A"
    assert (run_dir / "runtime.json").exists()
    data = json.loads(pointer.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_A"
    assert data["summary"] == {"m": 0.5}

    # imutabilidade: mesma execução não pode ser recriada
    with pytest.raises(FileExistsError):
        PipelineRunStore("demo", artifacts_root=tmp_artifacts, run_id="run_A")


def test_run_store_rejects_path_escape(tmp_artifacts):
    store = PipelineRunStore("demo", artifacts_root=tmp_artifacts)
    with pytest.raises(ValueError):
        store.path("../fora.txt")


def test_pipeline_run_writes_error_and_log_without_pointer(tmp_artifacts):
    store = PipelineRunStore("demo", artifacts_root=tmp_artifacts,
                             run_id="run_err")
    with pytest.raises(RuntimeError):
        with pipeline_run(store):
            print("mensagem capturada no run.log")
            raise RuntimeError("falha simulada")

    run_dir = tmp_artifacts / "demo" / "run_err"
    err = json.loads((run_dir / "error.json").read_text(encoding="utf-8"))
    assert err["type"] == "RuntimeError"
    assert "falha simulada" in err["message"]
    assert "mensagem capturada" in (run_dir / "run.log").read_text(
        encoding="utf-8")
    assert not (tmp_artifacts / "demo" / "latest_run.json").exists()


def test_run_store_warnings_file(tmp_artifacts):
    store = PipelineRunStore("demo", artifacts_root=tmp_artifacts,
                             run_id="run_W")
    store.add_warning("atenção")
    store.finalize()
    payload = json.loads(
        (tmp_artifacts / "demo" / "run_W" / "warnings.json")
        .read_text(encoding="utf-8"))
    assert payload["warnings"] == ["atenção"]


def test_load_pipeline_config_default_and_override(tmp_path):
    pipe = tmp_path / "pipe"
    (pipe / "configs").mkdir(parents=True)
    (pipe / "configs" / "default.yaml").write_text(
        "seed: 1\nsplits:\n  train: [3001]\n", encoding="utf-8")
    cfg = load_pipeline_config(pipe, argv=[])
    assert cfg["seed"] == 1 and cfg["splits"]["train"] == [3001]
    assert cfg["_config_path"].endswith("default.yaml")

    alt = tmp_path / "outra.yaml"
    alt.write_text("seed: 99\n", encoding="utf-8")
    cfg2 = load_pipeline_config(pipe, argv=["--config", str(alt)])
    assert cfg2["seed"] == 99

    with pytest.raises(FileNotFoundError):
        load_pipeline_config(pipe, argv=["--config", str(tmp_path / "x.yaml")])


def test_reproducibility_helpers():
    a = derive_seed(42, 3001, "Bahrain")
    assert a == derive_seed(42, 3001, "Bahrain")
    assert a != derive_seed(42, 3001, "Monaco")
    assert a != derive_seed(43, 3001, "Bahrain")

    set_global_seed(123)
    first = np.random.rand(3)
    set_global_seed(123)
    assert np.allclose(first, np.random.rand(3))


def test_manifest_helpers(project_root):
    commit, dirty = detect_git_state(project_root)
    assert commit is None or len(commit) >= 7
    deps = dependency_versions()
    assert "numpy" in deps and "pandas" in deps
