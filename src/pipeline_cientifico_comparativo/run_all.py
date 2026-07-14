"""Central orchestrator that executes every registered comparable model."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline_cientifico_comparativo.aggregation import write_model_summary
from src.pipeline_cientifico_comparativo.config_loader import load_resolved_config
from src.pipeline_cientifico_comparativo.registry import MODEL_REGISTRY, load_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute all registered scientific-comparative models."
    )
    parser.add_argument(
        "--mode",
        choices=["validation", "final_test"],
        default="validation",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model IDs. Default: every registered model.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--artifact-root", type=str, default=None)
    parser.add_argument(
        "--unlock-final-test",
        "--allow-test",
        dest="allow_test",
        action="store_true",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_ids = args.models or list(MODEL_REGISTRY)
    unknown = [model_id for model_id in model_ids if model_id not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown model IDs: {unknown}")

    executions: list[dict] = []
    artifact_root: Path | None = None

    for model_id in model_ids:
        spec = MODEL_REGISTRY[model_id]
        config = load_resolved_config(
            model_config=spec.config_file,
            seed_override=args.seed,
            artifact_root_override=args.artifact_root,
        )
        if artifact_root is None:
            artifact_root = Path(config["_meta"]["artifact_root_resolved"])
        runner = load_runner(spec)
        executions.append(
            runner(config, mode=args.mode, allow_test=args.allow_test)
        )

    if artifact_root is None:
        raise RuntimeError("No models were selected")

    comparison_dir = write_model_summary(
        artifact_root=artifact_root,
        mode=args.mode,
        executions=executions,
    )
    print(f"\nConsolidated comparison saved in: {comparison_dir}")


if __name__ == "__main__":
    main()
