"""
src/experiments/pipeline_config.py
===================================
Carregamento de configuração YAML dos pipelines históricos (Etapas 4–6).

Cada pipeline mantém seu próprio `configs/default.yaml`, preservando o split
temporal histórico dele (decisão complementar nº 3). O arquivo pode ser
substituído via linha de comando:

    python -m src.<pipeline>.<entry> --config caminho/para/outra_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_pipeline_config(
    pipeline_dir: str | Path,
    argv: list[str] | None = None,
    *,
    description: str = "",
) -> dict:
    """
    Resolve a configuração de um pipeline.

    Ordem de precedência: --config da CLI > configs/default.yaml do pipeline.
    O caminho do arquivo usado fica registrado em config['_config_path'].
    """
    pipeline_dir = Path(pipeline_dir)
    default_path = pipeline_dir / "configs" / "default.yaml"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=str(default_path),
        help=f"Arquivo YAML de configuração (padrão: {default_path})",
    )
    args, _ = parser.parse_known_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config["_config_path"] = str(config_path)
    return config
