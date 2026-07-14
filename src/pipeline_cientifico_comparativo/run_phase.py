"""CLI to execute one scientific-comparative phase."""

from __future__ import annotations

import argparse

import yaml

from src.pipeline_cientifico_comparativo.adapters.m0_adapter import (
    validate_environment,
)
from src.pipeline_cientifico_comparativo.config_loader import load_resolved_config
from src.pipeline_cientifico_comparativo.registry import (
    get_model_by_phase,
    load_runner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute one phase of the Formula 1 comparative pipeline."
    )
    parser.add_argument("--phase", type=int, default=0, help="Phase number")
    parser.add_argument(
        "--mode",
        choices=["validation", "final_test"],
        default="validation",
        help="final_test remains locked unless explicitly unlocked",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--artifact-root", type=str, default=None)
    parser.add_argument(
        "--unlock-final-test",
        "--allow-test",
        dest="allow_test",
        action="store_true",
        help="Unlock 2025 only after every development decision is frozen",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and data paths without training",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = get_model_by_phase(args.phase)
    config = load_resolved_config(
        model_config=args.config or spec.config_file,
        seed_override=args.seed,
        artifact_root_override=args.artifact_root,
    )

    if args.dry_run:
        environment = validate_environment(
            config,
            mode=args.mode,
            allow_test=args.allow_test,
        )
        print("Resolved configuration:\n")
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        print("Environment check:\n")
        print(yaml.safe_dump(environment, allow_unicode=True, sort_keys=False))
        if not environment["ready"]:
            raise SystemExit(
                "Dry-run failed: one or more season files are unavailable."
            )
        print("Dry-run successful.")
        return

    runner = load_runner(spec)
    execution = runner(config, mode=args.mode, allow_test=args.allow_test)
    print(f"\nCompleted {execution['model_id']} -> {execution['artifact_dir']}")


if __name__ == "__main__":
    main()
