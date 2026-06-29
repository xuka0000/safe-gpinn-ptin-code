from __future__ import annotations

import argparse
import json
from pathlib import Path

import ptin_sim.closed_loop.training as training


def parse_method_subset(text: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in text.split(",") if item.strip())
    valid_methods = set(training.EVAL_METHODS)
    unknown = [method for method in methods if method not in valid_methods]
    if unknown:
        raise ValueError(f"unknown evaluation method(s): {', '.join(unknown)}")
    if not methods:
        raise ValueError("at least one evaluation method is required")
    return methods


def run_subset_checkpoint_evaluation(
    *,
    output_dir: Path,
    config_path: Path,
    seed: int,
    rollout_dataset: Path,
    checkpoint_dir: Path,
    methods: tuple[str, ...],
    uncertainty_scale: float | None = None,
) -> dict[str, object]:
    training.EVAL_METHODS = methods
    manifest = training.run_closed_loop_policy_checkpoint_evaluation(
        output_dir=output_dir,
        config_path=config_path,
        adapter_mode=None,
        scenario_dir=None,
        seed=seed,
        rollout_dataset=rollout_dataset,
        checkpoint_dir=checkpoint_dir,
        evaluation_mode="env",
    )
    manifest_path = output_dir / "manifest.json"
    manifest = dict(manifest)
    manifest["method_subset"] = list(methods)
    manifest["scaled_rollout_dataset"] = str(rollout_dataset)
    if uncertainty_scale is not None:
        manifest["uncertainty_scale"] = float(uncertainty_scale)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run closed-loop checkpoint evaluation for a method subset."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rollout-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        default="two_stage_exact,rolling_mpc,safe_gpinn,mappo,diffusion_policy",
    )
    parser.add_argument("--uncertainty-scale", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = parse_method_subset(args.methods)
    manifest = run_subset_checkpoint_evaluation(
        output_dir=args.output_dir,
        config_path=args.config,
        seed=args.seed,
        rollout_dataset=args.rollout_dataset,
        checkpoint_dir=args.checkpoint_dir,
        methods=methods,
        uncertainty_scale=args.uncertainty_scale,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(args.output_dir),
                "method_subset": list(methods),
                "rollout_row_count": manifest.get("rollout_row_count"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
